"""Deterministic reading of what the user is asking for.

The LLM analysis step returns an ``action`` for each turn, but some
intents are too important to leave to it — asking to start over, asking why
a field is needed, or asking to apply. :func:`detect_action_override`
recognises those from the text itself and wins over the LLM's answer.

Everything here works on the raw message. Decisions that also need session
or profile state live in :mod:`turn_policy`.
"""

import re

from src.models.session import ConversationState
from src.services.conversation import scheme_reference
from src.services.conversation.language import detect_explicit_language_request

TOPIC_SWITCH_PATTERNS = (
    r"\bnow i need\b",
    r"\bi need .* instead\b",
    r"\binstead\b",
    r"\bchange (?:the )?topic\b",
    r"\bswitch (?:the )?topic\b",
    r"\bnew topic\b",
    r"\bnot .* anymore\b",
    r"\bactually i need\b",
    r"\blooking for .* instead\b",
    r"\bdifferent help\b",
    r"\bother help\b",
    r"\belse instead\b",
    r"\bab mujhe\b",
    r"\biske bajay\b",
    r"\btopic badal\b",
    r"\bab .* chahiye\b",
    r"अब मुझे",
    r"इसके बजाय",
    r"बजाय",
    r"विषय बदल",
    r"टॉपिक बदल",
    r"अब .* चाहिए",
)
DOCUMENT_REQUEST_PATTERNS = (
    r"\bdocument\b",
    r"\bdocuments\b",
    r"\bdoc\b",
    r"\bdocs\b",
    r"\bcertificate\b",
    r"\bcertificates\b",
    r"दस्तावेज",
    r"कागज",
    r"document guidance",
)
REJECTION_REQUEST_PATTERNS = (
    r"\breject(?:ion)?\b",
    r"\bwarning\b",
    r"\bmistake\b",
    r"\bavoid\b",
    r"\berror\b",
    r"अस्वीकृति",
    r"रिजेक्शन",
    r"गलती",
)
APPLICATION_REQUEST_PATTERNS = (
    r"\bapply\b",
    r"\bapplication\b",
    r"\bapplication steps?\b",
    r"\bapplication process\b",
    r"\bapplication procedure\b",
    r"\bprocedure\b",
    r"\bprocess\b",
    r"\bhow to apply\b",
    r"\bhow do i apply\b",
    r"\bsteps?\b",
    r"अवेदन",
    r"आवेदन",
    r"आवेदन प्रक्रिया",
    r"प्रक्रिया",
    r"कदम",
)
SCHEME_LIST_PATTERNS = (
    r"\bshow .*scheme list\b",
    r"\bshow .*options\b",
    r"\bother schemes\b",
    r"\banother scheme\b",
    r"\bback to schemes\b",
    r"\bscheme list again\b",
    r"\boptions again\b",
    r"फिर से योजनाएं",
    r"दूसरी योजना",
)
JUSTIFICATION_PATTERNS = (
    r"\bjustify\b",
    r"\bwhy (?:this|that) scheme\b",
    r"\bwhy did you suggest\b",
    r"\bwhy did you recommend\b",
    r"\bexplain why\b",
    r"क्यों सुझा",
    r"क्यों recommend",
)
SCHEME_QUESTION_PATTERNS = (
    r"\bwhat\b",
    r"\bwhy\b",
    r"\bhow\b",
    r"\bmean(?:ing)?\b",
    r"\bexplain\b",
    r"\bjustify\b",
    r"\bclarify\b",
    r"\bcan you\b",
    r"क्या",
    r"क्यों",
    r"कैसे",
)

COMMAND_ALIASES = {
    "/start": "start",
    "/shuru": "start",
    "/help": "help",
    "/madad": "help",
    "/language": "language",
    "/lang": "language",
    "/bhasha": "language",
}

_REASON_REQUEST_PATTERNS = (
    r"\bwhy\b",
    r"\breason\b",
    r"\bwhy do you need\b",
    r"\bwhat(?:'s| is) the matter\b",
    r"\bkyu\b",
    r"\bkyon\b",
    r"क्यों",
)

_FIELD_HELP_PATTERNS = (
    r"\bwhat does\b",
    r"\bwhat is\b",
    r"\bmeaning of\b",
    r"\bmatlab\b",
    r"\bexample\b",
    r"\bhow (?:do|should|can) i\b",
    r"\bhow to\b",
    r"\bestimate\b",
    r"\bcalculate\b",
    r"\bwhich one\b",
    r"\bexplain\b",
    r"\bclarify\b",
)

_FIELD_HELP_KEYWORDS = {
    "age": ("age", "years", "year", "saal", "उम्र"),
    "category": ("category", "caste", "obc", "sc", "st", "ews", "general", "श्रेणी"),
    "annual_income": ("income", "salary", "earn", "monthly", "yearly", "annual", "mahina", "आय"),
    "gender": ("gender", "male", "female", "woman", "man", "लिंग"),
    "life_event": ("assistance", "help", "scheme", "support", "situation", "मदद"),
}

# Questions that need a real answer rather than a view switch, even when they
# mention documents or the application process.
_ANALYTICAL_QUESTION_PATTERNS = (
    r"^which\b",
    r"^what\b",
    r"^how\b",
    r"^why\b",
    r"^is\b",
    r"^are\b",
    r"^do\b",
    r"^does\b",
    r"^can\b",
    r"^will\b",
    r"\bif\b",
    r"\bwithout\b",
    r"\bneed\b",
    r"\brequire",
    r"\bकौन(?:सा|सी|से)\b",
    r"\bक्या\b",
    r"\bकैसे\b",
    r"\bअगर\b",
    r"\bकिस\b",
)

# Above this length a pattern hit is treated as a question rather than a
# terse navigation command like "documents?".
_MAX_NAVIGATION_TOKENS = 4

_TOKEN_RE = re.compile(r"[A-Za-z0-9ऀ-ॿ]+")

_START_OVER_RE = re.compile(
    r"\b(start over|restart|reset|begin again|from scratch|new search)\b|फिर से|शुरू से"
)
_APPLY_RE = re.compile(r"\b(apply|application|apply kar|apply kare|अवेदन|आवेदन)\b")
_HANDOFF_RE = re.compile(
    r"\b(csc|human help|service center|service centre|nearest center|operator|contact center)\b"
)
_DETAILS_RE = re.compile(
    r"\b(detail|details|document|documents|eligibility|benefit|explain|translate|translation)\b"
)

_AFFIRMATIVE_RE = re.compile(
    r"(?i)\b("
    r"yes|yeah|yep|yup|sure|ok|okay|alright|"
    r"absolutely|definitely|of course|please|go ahead|"
    r"let'?s do|proceed|I want|I'?d like|"
    r"haan|haa|ha|ji|bilkul|zaroor|chalo|thik|"
    r"हां|हाँ|जी|ठीक|ज़रूर|बिल्कुल|चलो"
    r")\b"
)

# Word boundaries work for the Latin patterns; Devanagari has no \b, so the
# Hindi alternatives are matched bare.
_SKIP_RE = re.compile(
    "|".join(
        (
            r"\bdon'?t know\b",
            r"\bdo not know\b",
            r"\bno idea\b",
            r"\bnot sure\b",
            r"\bunsure\b",
            r"\bskip\b",
            r"\bpass\b",
            r"\bnext\b",
            r"\bmove on\b",
            r"\bnahi pata\b",
            r"\bpata nahi\b",
            r"\bmaloom nahi\b",
            r"\bnahi maloom\b",
            r"पता नहीं",
            r"नहीं पता",
            r"मालूम नहीं",
            r"छोड़ो",
            r"अगला",
        )
    ),
    re.IGNORECASE,
)

_BENEFICIARY_MARKERS = (
    "daughter",
    "son",
    "beti",
    "beta",
    "my daughter",
    "my son",
    "meri beti",
    "mera beta",
    "applicant",
    "beneficiary",
)
_SCOPE_MARKERS = (
    "both",
    "also",
    "too",
    "choose one",
    "one person",
    "one applicant",
    "first",
    "kiske liye",
    "kis ke liye",
    "scholarship",
    "education",
    "college",
    "bhi",
)


def matches_any_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    """Return True when the text matches any regex pattern in the tuple."""
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def is_explicit_topic_switch(text: str) -> bool:
    """Return True when the user is clearly asking to change topics."""
    return matches_any_pattern(text.lower(), TOPIC_SWITCH_PATTERNS)


def wants_scheme_list_again(text: str) -> bool:
    """Return True when the user wants to go back to the candidate list."""
    return matches_any_pattern(text, SCHEME_LIST_PATTERNS)


def is_affirmative(text: str) -> bool:
    """Check if the user's message is a short affirmative/confirmation."""
    words = text.strip().split()
    return len(words) <= 5 and bool(_AFFIRMATIVE_RE.search(text))


def wants_to_skip(text: str) -> bool:
    """Check if the user wants to skip the current question."""
    return bool(_SKIP_RE.search(text))


def detect_reason_request(text: str) -> bool:
    """Detect when the user asks why a field is needed."""
    return matches_any_pattern(text.lower(), _REASON_REQUEST_PATTERNS)


def extract_supported_command(text: str) -> str | None:
    """Return the normalized Telegram command when the turn starts with one."""
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None

    command_token = stripped.split(maxsplit=1)[0].lower()
    base_command = command_token.split("@", maxsplit=1)[0]
    return COMMAND_ALIASES.get(base_command)


def detect_field_help_request(text: str, field: str | None) -> bool:
    """Detect clarification questions about the field currently being asked."""
    if not field:
        return False

    text_lower = text.lower()
    asks_something = (
        any(re.search(pattern, text_lower) for pattern in _FIELD_HELP_PATTERNS)
        or "?" in text
    )
    if not asks_something:
        return False

    return any(
        keyword in text_lower for keyword in _FIELD_HELP_KEYWORDS.get(field, ())
    )


def looks_like_scheme_question(text: str) -> bool:
    """Return True when the user is asking a natural-language question."""
    if "?" in text:
        return True
    return matches_any_pattern(text.lower(), SCHEME_QUESTION_PATTERNS)


def is_navigation_only_scheme_followup(text: str) -> bool:
    """Return True for short view-switch commands, not substantive questions.

    "documents?" is navigation even though it ends in a question mark, while
    "which document do I need if I have no ration card?" is a real question
    that deserves an answer instead of a document card.
    """
    stripped = text.strip()
    if not stripped:
        return False

    nav_patterns = (
        DOCUMENT_REQUEST_PATTERNS
        + REJECTION_REQUEST_PATTERNS
        + APPLICATION_REQUEST_PATTERNS
    )
    if not matches_any_pattern(stripped, nav_patterns):
        return False

    text_lower = stripped.lower()
    if any(re.search(pattern, text_lower) for pattern in _ANALYTICAL_QUESTION_PATTERNS):
        return False

    return len(_TOKEN_RE.findall(stripped)) <= _MAX_NAVIGATION_TOKENS


def is_multi_beneficiary_scope_followup(text: str, currently_asking: str | None) -> bool:
    """Detect scope questions about self vs child while a field is pending."""
    if currently_asking is None:
        return False
    if detect_reason_request(text):
        return False
    if detect_field_help_request(text, currently_asking):
        return False
    if not looks_like_scheme_question(text):
        return False

    text_lower = text.lower()
    return any(marker in text_lower for marker in _BENEFICIARY_MARKERS) and any(
        marker in text_lower for marker in _SCOPE_MARKERS
    )


def detect_action_override(
    text: str,
    current_state: ConversationState,
    currently_asking: str | None,
    resolved_scheme_id: str | None,
    active_scheme_id: str | None,
) -> str | None:
    """Infer high-signal actions deterministically.

    Order matters: the checks run most-specific first, and the first hit
    wins over whatever action the LLM proposed for this turn.
    """
    text_lower = text.lower().strip()

    if detect_explicit_language_request(text):
        return "change_language"
    if _START_OVER_RE.search(text_lower):
        return "start_over"
    if currently_asking and detect_reason_request(text):
        return "ask_field_reason"
    if detect_field_help_request(text, currently_asking):
        return "clarify_field"
    if wants_to_skip(text):
        return "skip_field"
    if _APPLY_RE.search(text_lower):
        return "request_application"
    if current_state in scheme_reference.SCHEME_CONTEXT_STATES and matches_any_pattern(
        text_lower,
        APPLICATION_REQUEST_PATTERNS,
    ):
        return "request_application"
    if _HANDOFF_RE.search(text_lower):
        return "request_handoff"
    if _DETAILS_RE.search(text_lower):
        return "request_details"
    if resolved_scheme_id and not scheme_reference.resolved_scheme_matches_active_scheme(
        current_state,
        resolved_scheme_id,
        active_scheme_id,
    ):
        return (
            "switch_scheme"
            if current_state in scheme_reference.SCHEME_DETAIL_STATES
            else "select_scheme"
        )
    return None
