"""Main conversation orchestrator.

This is the brain of Delhi Scheme Saathi. It:
1. Loads/creates session
2. Analyzes user input via LLM
3. Updates user profile
4. Executes FSM transitions
5. Runs state-specific logic (matching, document resolution, etc.)
6. Generates structured or LLM-grounded responses depending on the turn
7. Saves session state
"""

import logging
import re
from typing import Any

import asyncpg

from src.config import get_settings
from src.db import office_repo, scheme_repo
from src.models.api import ChatRequest, ChatResponse
from src.models.document import DocumentChain
from src.models.scheme import SchemeMatch
from src.models.session import ConversationState, Session, UserProfile
from src.prompts.loader import get_system_prompt
from src.services import (
    document_resolver,
    fsm,
    life_event_classifier,
    profile_extractor,
    rejection_engine,
    response_generator,
    scheme_matcher,
    scheme_relevance,
    session_manager,
)
from src.services.ai_background import enqueue_memory_refresh
from src.services.ai_orchestrator import get_ai_orchestrator
from src.services.conversation_memory import should_refresh_working_memory
from src.utils.formatters import (
    format_inline_keyboard,
    format_language_keyboard,
    format_presented_scheme_keyboard,
)
from src.utils.validators import sanitize_input

logger = logging.getLogger(__name__)

# Icons for life event categories
LIFE_EVENT_ICONS = {
    "HOUSING": "🏠",
    "HEALTH_CRISIS": "🏥",
    "EDUCATION": "📚",
    "DEATH_IN_FAMILY": "🙏",
    "MARITAL_DISTRESS": "🙏",
    "BUSINESS_STARTUP": "💼",
    "JOB_LOSS": "💼",
    "WOMEN_EMPOWERMENT": "👩",
    "CHILDBIRTH": "👶",
    "MARRIAGE": "💒",
}

MATCH_RELEVANT_FIELDS = {"life_event", "age", "category", "annual_income", "gender"}
SCHEME_CONTEXT_STATES = {
    ConversationState.SCHEME_PRESENTATION,
    ConversationState.SCHEME_DETAILS,
    ConversationState.DOCUMENT_GUIDANCE,
    ConversationState.REJECTION_WARNINGS,
    ConversationState.APPLICATION_HELP,
}
SCHEME_SELECTION_CUES = (
    "scheme",
    "yojana",
    "योजना",
    "option",
    "number",
    "no",
    "select",
    "choose",
    "pick",
    "details",
    "detail",
    "about",
    "show",
    "explain",
)
SCHEME_NAME_STOPWORDS = {
    "scheme",
    "yojana",
    "योजना",
    "the",
    "a",
    "an",
    "this",
    "that",
    "please",
    "detail",
    "details",
    "about",
    "show",
    "explain",
    "tell",
    "me",
    "for",
    "of",
    "apply",
    "application",
    "option",
    "number",
    "no",
    "select",
    "choose",
    "pick",
    "open",
    "want",
    "need",
    "ki",
    "ke",
    "ka",
    "ko",
    "mein",
    "mai",
}
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


def _text_variant(language: str, hi: str, en: str, hinglish: str | None = None) -> str:
    """Select a language-specific string."""
    if language == "hi":
        return hi
    if language == "hinglish":
        return hinglish or en
    return en


def _normalize_language(language: str | None) -> str:
    """Normalize language codes to supported response variants."""
    if language in {"hi", "en", "hinglish"}:
        return language
    return "hi"


def _infer_text_language(text: str) -> str:
    """Infer the user's language from the raw text when no session lock exists."""
    devanagari_chars = sum(1 for char in text if "\u0900" <= char <= "\u097F")
    alpha_chars = sum(1 for char in text if char.isalpha())
    if alpha_chars and devanagari_chars / alpha_chars > 0.3:
        return "hi"

    text_lower = text.lower()
    hinglish_markers = (
        "mujhe", "chahiye", "batao", "batayiye", "kyu", "kaise",
        "sahayata", "madad", "mera", "meri", "mere", "kripya",
    )
    marker_hits = sum(
        1 for marker in hinglish_markers
        if re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", text_lower)
    )
    if marker_hits >= 2:
        return "hinglish"
    return "en"


def _detect_explicit_language_request(text: str) -> str | None:
    """Detect when the user explicitly asks for a specific language."""
    text_lower = text.lower()
    language_patterns = {
        "en": [
            r"\benglish\b",
            r"\buse english\b",
            r"\bplease use english\b",
            r"\bi don't understand hindi\b",
        ],
        "hi": [
            r"\bhindi\b",
            r"हिंदी",
            r"\buse hindi\b",
            r"\bhindi language\b",
        ],
        "hinglish": [
            r"\bhinglish\b",
            r"\buse hinglish\b",
            r"\broman hindi\b",
        ],
    }

    for language, patterns in language_patterns.items():
        if any(re.search(pattern, text_lower) for pattern in patterns):
            return language
    return None


def _detect_reason_request(text: str) -> bool:
    """Detect when the user asks why a field is needed."""
    text_lower = text.lower()
    patterns = [
        r"\bwhy\b",
        r"\breason\b",
        r"\bwhy do you need\b",
        r"\bwhat(?:'s| is) the matter\b",
        r"\bkyu\b",
        r"\bkyon\b",
        r"क्यों",
    ]
    return any(re.search(pattern, text_lower) for pattern in patterns)


def _extract_supported_command(text: str) -> str | None:
    """Return the normalized Telegram command when the turn starts with one."""
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None

    command_token = stripped.split(maxsplit=1)[0].lower()
    base_command = command_token.split("@", maxsplit=1)[0]
    return COMMAND_ALIASES.get(base_command)


def _command_response_language(session: Session) -> str:
    """Choose the safest deterministic command response language."""
    return session.language_preference if session.language_preference != "auto" else "en"


def _build_presented_scheme_selection_text(
    presented_schemes: list[dict[str, str]],
    language: str,
) -> str | None:
    """Render stored presented schemes without needing full match payloads."""
    if not presented_schemes:
        return None

    header = _text_variant(
        language,
        "🎯 आपने ये योजना विकल्प देखे थे:",
        "🎯 You were viewing these scheme options:",
        "🎯 Aap ye scheme options dekh rahe the:",
    )
    footer = _text_variant(
        language,
        "नीचे बटन दबाकर योजना चुनें।",
        "Tap a button below to open a scheme.",
        "Neeche button dabakar scheme kholiye.",
    )
    lines = [header, ""]
    for index, scheme in enumerate(presented_schemes[:5], 1):
        name = scheme.get("name_hindi") if language == "hi" else scheme.get("name")
        display_name = name or scheme.get("name") or scheme.get("name_hindi") or "Scheme"
        lines.append(f"{index}. {display_name}")
    lines.extend(["", footer])
    return "\n".join(lines)


def _detect_field_help_request(text: str, field: str | None) -> bool:
    """Detect clarification questions about the field currently being asked."""
    if not field:
        return False

    text_lower = text.lower()
    generic_help_patterns = (
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
    if not any(re.search(pattern, text_lower) for pattern in generic_help_patterns) and "?" not in text:
        return False

    field_keywords = {
        "age": ("age", "years", "year", "saal", "उम्र"),
        "category": ("category", "caste", "obc", "sc", "st", "ews", "general", "श्रेणी"),
        "annual_income": ("income", "salary", "earn", "monthly", "yearly", "annual", "mahina", "आय"),
        "gender": ("gender", "male", "female", "woman", "man", "लिंग"),
        "life_event": ("assistance", "help", "scheme", "support", "situation", "मदद"),
    }
    return any(keyword in text_lower for keyword in field_keywords.get(field, ()))


def _looks_like_scheme_question(text: str) -> bool:
    """Return True when the user is asking a natural-language question."""
    if "?" in text:
        return True
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in SCHEME_QUESTION_PATTERNS)


def _is_navigation_only_scheme_followup(text: str) -> bool:
    """Return True for short view-switch commands, not substantive follow-up questions."""
    stripped = text.strip()
    if not stripped or "?" in stripped:
        return False

    token_count = len(re.findall(r"[A-Za-z0-9\u0900-\u097F]+", stripped))
    if token_count > 4:
        return False

    return _matches_any_pattern(
        stripped,
        DOCUMENT_REQUEST_PATTERNS
        + REJECTION_REQUEST_PATTERNS
        + APPLICATION_REQUEST_PATTERNS,
    )


def _resolved_scheme_matches_active_scheme(
    current_state: ConversationState,
    resolved_scheme_id: str | None,
    active_scheme_id: str | None,
) -> bool:
    """Return True when a resolved scheme only repeats the already-open scheme."""
    return bool(
        resolved_scheme_id
        and active_scheme_id
        and resolved_scheme_id == active_scheme_id
        and current_state in SCHEME_CONTEXT_STATES
    )


def _should_answer_scheme_question(
    text: str,
    current_state: ConversationState,
    action: str | None,
    resolved_scheme_id: str | None,
    active_scheme_id: str | None,
    has_scheme_context: bool,
) -> bool:
    """Detect scheme follow-up questions that deserve an answer, not a card replay."""
    if current_state not in SCHEME_CONTEXT_STATES:
        return False
    if not has_scheme_context:
        return False
    if resolved_scheme_id and not _resolved_scheme_matches_active_scheme(
        current_state,
        resolved_scheme_id,
        active_scheme_id,
    ):
        return False
    if (
        current_state in {
            ConversationState.DOCUMENT_GUIDANCE,
            ConversationState.REJECTION_WARNINGS,
            ConversationState.APPLICATION_HELP,
        }
        and _looks_like_scheme_question(text)
        and not _is_navigation_only_scheme_followup(text)
    ):
        return True
    if action in {
        "start_over",
        "goodbye",
        "skip_field",
        "ask_field_reason",
        "clarify_field",
        "request_application",
        "request_handoff",
        "select_scheme",
        "switch_scheme",
    }:
        return False
    if _wants_scheme_list_again(text):
        return False
    if _matches_any_pattern(text, DOCUMENT_REQUEST_PATTERNS):
        return False
    if _matches_any_pattern(text, REJECTION_REQUEST_PATTERNS):
        return False
    return _looks_like_scheme_question(text)


def _is_low_context_matching_turn(session: Session, user_message: str) -> bool:
    """Return True when matching was triggered by a field reply or short confirmation.

    These turns usually contain bare values like "5 lakhs" or confirmations like
    "yes", so the active profile is a better signal than the raw message text.
    The AI relevance gate is also more likely to over-clarify on these inputs.
    """
    if session.currently_asking is not None:
        return True
    return bool(session.user_profile.life_event and _is_affirmative(user_message))


def _build_matching_focus_text(profile: UserProfile, user_message: str) -> str:
    """Build a stable intent summary for AI relevance judging.

    The judge should evaluate the active scheme search goal, not only the latest
    collection turn such as a bare income answer.
    """
    focus_parts = []
    if profile.life_event:
        focus_parts.append(f"Need area: {profile.life_event}")
    if profile.marital_status:
        focus_parts.append(f"Marital status: {profile.marital_status}")
    if profile.gender:
        focus_parts.append(f"Gender: {profile.gender}")
    if profile.age is not None:
        focus_parts.append(f"Age: {profile.age}")
    if profile.category:
        focus_parts.append(f"Category: {profile.category}")
    if profile.annual_income is not None:
        focus_parts.append(f"Annual income: ₹{profile.annual_income}")
    if user_message.strip():
        focus_parts.append(f"Latest reply: {user_message.strip()}")
    return " | ".join(focus_parts) if focus_parts else user_message


def _detect_action_override(
    text: str,
    current_state: ConversationState,
    currently_asking: str | None,
    resolved_scheme_id: str | None,
    active_scheme_id: str | None,
) -> str | None:
    """Infer high-signal actions deterministically."""
    text_lower = text.lower().strip()

    if _detect_explicit_language_request(text):
        return "change_language"
    if re.search(
        r"\b(start over|restart|reset|begin again|from scratch|new search)\b|फिर से|शुरू से",
        text_lower,
    ):
        return "start_over"
    if currently_asking and _detect_reason_request(text):
        return "ask_field_reason"
    if _detect_field_help_request(text, currently_asking):
        return "clarify_field"
    if _wants_to_skip(text):
        return "skip_field"
    if re.search(r"\b(apply|application|apply kar|apply kare|अवेदन|आवेदन)\b", text_lower):
        return "request_application"
    if current_state in SCHEME_CONTEXT_STATES and _matches_any_pattern(
        text_lower,
        APPLICATION_REQUEST_PATTERNS,
    ):
        return "request_application"
    if re.search(r"\b(csc|human help|service center|service centre|nearest center|operator|contact center)\b", text_lower):
        return "request_handoff"
    if re.search(r"\b(detail|details|document|documents|eligibility|benefit|explain|translate|translation)\b", text_lower):
        return "request_details"
    if resolved_scheme_id and not _resolved_scheme_matches_active_scheme(
        current_state,
        resolved_scheme_id,
        active_scheme_id,
    ):
        return "switch_scheme" if current_state in {
            ConversationState.SCHEME_DETAILS,
            ConversationState.DOCUMENT_GUIDANCE,
            ConversationState.REJECTION_WARNINGS,
            ConversationState.APPLICATION_HELP,
        } else "select_scheme"
    return None


def _matches_any_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    """Return True when the text matches any regex pattern in the tuple."""
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _wants_scheme_list_again(text: str) -> bool:
    """Return True when the user wants to go back to the candidate list."""
    return _matches_any_pattern(text, SCHEME_LIST_PATTERNS)


def _requested_scheme_view(
    text: str,
    action: str | None,
    current_state: ConversationState,
    has_selected_scheme: bool,
    resolved_scheme_id: str | None,
    active_scheme_id: str | None,
) -> ConversationState | None:
    """Resolve scheme-area navigation within the 10-state FSM."""
    same_active_scheme = _resolved_scheme_matches_active_scheme(
        current_state,
        resolved_scheme_id,
        active_scheme_id,
    )
    if _wants_scheme_list_again(text):
        return ConversationState.SCHEME_PRESENTATION
    if action == "request_handoff":
        return ConversationState.CSC_HANDOFF
    if action == "request_application" or _matches_any_pattern(text, APPLICATION_REQUEST_PATTERNS):
        return ConversationState.APPLICATION_HELP
    if _matches_any_pattern(text, DOCUMENT_REQUEST_PATTERNS):
        return ConversationState.DOCUMENT_GUIDANCE
    if _matches_any_pattern(text, REJECTION_REQUEST_PATTERNS):
        return ConversationState.REJECTION_WARNINGS
    if action == "answer_scheme_question":
        return (
            current_state
            if current_state in SCHEME_CONTEXT_STATES
            else ConversationState.SCHEME_DETAILS
        )
    if action == "request_details" or _matches_any_pattern(text, JUSTIFICATION_PATTERNS):
        return ConversationState.SCHEME_DETAILS
    if action in {"select_scheme", "switch_scheme"} or (
        resolved_scheme_id and not same_active_scheme
    ):
        return ConversationState.SCHEME_DETAILS
    if not has_selected_scheme and current_state not in {ConversationState.SCHEME_PRESENTATION, ConversationState.CSC_HANDOFF}:
        return None
    return None


def _matching_field_changes(
    before_profile: UserProfile,
    after_profile: UserProfile,
) -> set[str]:
    """Return search-relevant profile fields that changed value."""
    changed_fields = set()
    for field in MATCH_RELEVANT_FIELDS:
        if getattr(before_profile, field) != getattr(after_profile, field):
            changed_fields.add(field)
    return changed_fields


def _should_refresh_matches_after_profile_change(
    *,
    session: Session,
    profile: UserProfile,
    matching_inputs_changed: bool,
    action: str | None,
    requested_state: ConversationState | None,
) -> bool:
    """Decide when updated profile facts should trigger a fresh scheme match."""
    if not matching_inputs_changed or not profile.is_complete_for_matching:
        return False
    if session.state not in SCHEME_CONTEXT_STATES:
        return False
    if requested_state in SCHEME_CONTEXT_STATES | {ConversationState.CSC_HANDOFF}:
        return False
    return not _should_preserve_scheme_context_action(action)


def _should_preserve_scheme_context_action(action: str | None) -> bool:
    """Return True when an explicit scheme-flow action should keep the active scheme."""
    return action in {
        "answer_scheme_question",
        "request_details",
        "request_application",
        "request_handoff",
        "select_scheme",
        "switch_scheme",
    }


def _collection_state_for_profile(profile: UserProfile) -> ConversationState:
    """Return the active collection state based on whether the topic is known."""
    if profile.life_event:
        return ConversationState.PROFILE_COLLECTION
    return ConversationState.SITUATION_UNDERSTANDING


def _is_explicit_topic_switch(text: str) -> bool:
    """Return True when the user is clearly asking to change topics."""
    text_lower = text.lower()
    return any(re.search(pattern, text_lower) for pattern in TOPIC_SWITCH_PATTERNS)


def _should_update_life_event(
    session: Session,
    detected_life_event: str | None,
    extracted_fields: dict[str, Any],
    action: str | None,
    user_message: str,
) -> bool:
    """Decide whether the current turn is allowed to replace the active topic."""
    current_life_event = session.user_profile.life_event
    if not detected_life_event or detected_life_event == current_life_event:
        return False
    if current_life_event is None:
        return True
    if session.currently_asking == "life_event":
        return True
    if _is_explicit_topic_switch(user_message):
        return True

    if (
        session.state in SCHEME_CONTEXT_STATES | {ConversationState.CSC_HANDOFF}
        and (
            action in {
            "request_details",
            "request_application",
            "request_handoff",
            "select_scheme",
            "switch_scheme",
            }
            or session.selected_scheme_id
            or session.presented_schemes
        )
    ):
        return False

    # When the bot is collecting a specific field, treat this turn as a field
    # answer unless the user explicitly changes topic.
    if session.currently_asking and session.currently_asking != "life_event":
        if session.currently_asking in extracted_fields:
            return False
        if extracted_fields:
            return False
        if action in {"answer_field", "skip_field", "ask_field_reason"}:
            return False

    return (
        session.state in {
            ConversationState.GREETING,
            ConversationState.SITUATION_UNDERSTANDING,
            ConversationState.PROFILE_COLLECTION,
        }
        and session.currently_asking is None
        and not extracted_fields
    )


def _tokenize_scheme_reference(text: str) -> set[str]:
    """Tokenize user text for safe scheme-name matching."""
    tokens = set()
    for token in re.findall(r"[a-z0-9\u0900-\u097F]+", text.lower()):
        if len(token) <= 1:
            continue
        if token in SCHEME_NAME_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _is_selection_phrase(text: str) -> bool:
    """Return True when the message looks like a scheme selection request."""
    stripped = text.strip().lower()
    if re.fullmatch(r"\d+", stripped):
        return True
    if len(stripped.split()) <= 4:
        return True
    return any(cue in stripped for cue in SCHEME_SELECTION_CUES)


# ---------------------------------------------------------------------------
# Plain-text formatting helpers (no MarkdownV2 — sent as plain Telegram text)
# ---------------------------------------------------------------------------

def _format_currency_plain(amount: int | float | None, language: str = "hi") -> str:
    """Format amount as Indian currency with lakh/crore notation."""
    if amount is None:
        return ""
    if amount >= 10000000:
        crores = amount / 10000000
        lbl = "करोड़" if language == "hi" else "Cr"
        return f"₹{crores:.1f} {lbl}" if crores != int(crores) else f"₹{int(crores)} {lbl}"
    if amount >= 100000:
        lakhs = amount / 100000
        lbl = "लाख" if language == "hi" else "lakh"
        return f"₹{lakhs:.1f} {lbl}" if lakhs != int(lakhs) else f"₹{int(lakhs)} {lbl}"
    return f"₹{amount:,.0f}"


def _truncate_at_word(text: str, max_len: int, ellipsis: str = "...") -> str:
    """Truncate text at word boundary to avoid cutting words mid-way.

    Example: _truncate_at_word("pursuing professional and technical", 32)
             returns "pursuing professional and..." (not "pursuing professional and techni...")
    """
    if len(text) <= max_len:
        return text
    # Leave room for ellipsis
    truncated = text[: max_len - len(ellipsis)]
    # Find last space to avoid cutting mid-word
    last_space = truncated.rfind(" ")
    if last_space > max_len * 0.5:  # Only use space if we keep >50% of text
        truncated = truncated[:last_space]
    return truncated.rstrip() + ellipsis


def _truncate_at_sentence(text: str, max_len: int, ellipsis: str = "...") -> str:
    """Prefer sentence boundaries when shortening long text for chat output."""
    if len(text) <= max_len:
        return text

    sentence_end = max(
        (
            text.rfind(marker, 0, max_len) + 1
            for marker in (". ", "! ", "? ", ".\n", "!\n", "?\n", "। ", "।\n", "।")
            if text.rfind(marker, 0, max_len) != -1
        ),
        default=-1,
    )
    if sentence_end >= max_len * 0.5:
        return text[:sentence_end].rstrip()
    return _truncate_at_word(text, max_len, ellipsis)


def _wrap_long_text(text: str, max_line_len: int = 70, prefix: str = "     ") -> str:
    """Wrap long text into multiple lines for readability.

    Used for long database fields like issuing_authority.
    """
    if len(text) <= max_line_len:
        return text

    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        if len(test_line) <= max_line_len:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    # Join with prefix for subsequent lines
    if len(lines) <= 1:
        return text
    return lines[0] + "\n" + "\n".join(f"{prefix}{line}" for line in lines[1:])


def _build_scheme_list_text(
    schemes: list[SchemeMatch],
    profile: UserProfile,
    language: str,
) -> str:
    """Build a numbered, plain-text scheme list with eligibility info."""
    if not schemes:
        return _text_variant(
            language,
            "कोई योजना नहीं मिली।",
            "No matching schemes found.",
            "Koi matching scheme nahi mili.",
        )

    header = _text_variant(
        language,
        "🎯 आपके लिए ये योजनाएं मिली हैं:",
        "🎯 Found these schemes for you:",
        "🎯 Aapke liye ye schemes mili hain:",
    )
    lines = [header, ""]

    for i, match in enumerate(schemes[:5], 1):
        scheme = match.scheme

        icon = "📋"
        preferred_events = list(scheme.life_events)
        if profile.life_event and profile.life_event in preferred_events:
            preferred_events = [profile.life_event] + [
                event for event in preferred_events if event != profile.life_event
            ]
        for event in preferred_events:
            if event in LIFE_EVENT_ICONS:
                icon = LIFE_EVENT_ICONS[event]
                break

        name = scheme.name_hindi if language == "hi" else scheme.name
        lines.append(f"{i}. {icon} {name}")

        # Benefits
        if scheme.benefits_amount:
            amount_str = _format_currency_plain(scheme.benefits_amount, language)
            freq_map = {
                "monthly": _text_variant(language, "मासिक", "/month", "per month"),
                "yearly": _text_variant(language, "वार्षिक", "/year", "per year"),
                "one-time": _text_variant(language, "एकमुश्त", "one-time", "one-time"),
                "installments": _text_variant(language, "किश्तों में", "in installments", "installments mein"),
            }
            freq_display = freq_map.get(scheme.benefits_frequency or "", "")
            benefit_label = _text_variant(language, "लाभ", "Benefit", "Benefit")
            lines.append(f"   💰 {benefit_label}: {amount_str} {freq_display}".rstrip())

        # Department
        dept = scheme.department_hindi if language == "hi" else scheme.department
        if len(dept) > 40:
            dept = dept[:37] + "..."
        dept_label = _text_variant(language, "विभाग", "Dept", "Dept")
        lines.append(f"   🏛️ {dept_label}: {dept}")

        # Eligibility match
        if match.eligibility_match:
            field_labels = {
                "age": ("आयु", "Age"),
                "income": ("आय", "Income"),
                "income_segment": ("आय वर्ग", "Income band"),
                "category": ("श्रेणी", "Category"),
                "gender": ("लिंग", "Gender"),
            }
            parts = []
            for field, is_match in match.eligibility_match.items():
                hi_lbl, en_lbl = field_labels.get(field, (field, field))
                lbl = hi_lbl if language == "hi" else en_lbl
                mark = "✓" if is_match else "✗"
                parts.append(f"{lbl} {mark}")

            all_match = all(match.eligibility_match.values())
            prefix = (
                _text_variant(language, "✅ पात्र", "✅ Eligible", "✅ Eligible")
                if all_match
                else _text_variant(language, "⚠️ जाँचें", "⚠️ Check", "⚠️ Check")
            )
            lines.append(f"   {prefix}: {' • '.join(parts)}")

        lines.append("")  # blank line between schemes

    footer = _text_variant(
        language,
        "👆 नीचे बटन दबाएं या नंबर बताएं।",
        "👆 Tap a button below or type the number.",
        "👆 Neeche button dabaiye ya number type kijiye.",
    )
    lines.append(footer)
    return "\n".join(lines)


async def _build_scheme_details_text(
    pool: asyncpg.Pool,
    scheme_id: str,
    profile: UserProfile,
    language: str,
) -> str:
    """Build a scheme overview and justification view."""
    scheme = await scheme_repo.get_scheme_by_id(pool, scheme_id)
    if not scheme:
        return _text_variant(language, "योजना नहीं मिली।", "Scheme not found.", "Scheme nahi mili.")

    icon = "📋"
    preferred_life_events = list(scheme.life_events)
    if profile.life_event and profile.life_event in preferred_life_events:
        preferred_life_events = [profile.life_event] + [
            event for event in preferred_life_events if event != profile.life_event
        ]
    for event in preferred_life_events:
        if event in LIFE_EVENT_ICONS:
            icon = LIFE_EVENT_ICONS[event]
            break

    name = scheme.name_hindi if language == "hi" else scheme.name
    lines = [f"{icon} {name}", ""]

    desc = scheme.description_hindi if language == "hi" else scheme.description
    desc = _truncate_at_sentence(desc, 380)
    lines.append(desc)
    lines.append("")

    if scheme.benefits_amount:
        amount_str = _format_currency_plain(scheme.benefits_amount, language)
        benefit_label = _text_variant(language, "लाभ राशि", "Benefit", "Benefit")
        lines.append(f"💰 {benefit_label}: {amount_str}")

    elig = scheme.eligibility
    elig_parts = []
    if elig.min_age or elig.max_age:
        age_lbl = _text_variant(language, "आयु", "Age", "Age")
        elig_parts.append(f"{age_lbl}: {elig.min_age or 18}-{elig.max_age or '∞'}")
    if elig.max_income:
        income_str = _format_currency_plain(elig.max_income, language)
        income_lbl = _text_variant(language, "अधिकतम आय", "Max income", "Max income")
        elig_parts.append(f"{income_lbl}: {income_str}")
    if elig.caste_categories:
        cat_lbl = _text_variant(language, "श्रेणी", "Category", "Category")
        elig_parts.append(f"{cat_lbl}: {', '.join(elig.caste_categories)}")
    if elig.income_segments:
        band_lbl = _text_variant(language, "आय वर्ग", "Income band", "Income band")
        elig_parts.append(f"{band_lbl}: {', '.join(elig.income_segments)}")
    if elig_parts:
        elig_label = _text_variant(language, "पात्रता", "Eligibility", "Eligibility")
        lines.append(f"✅ {elig_label}: {' | '.join(elig_parts)}")

    match_details = scheme_repo.calculate_eligibility_match(scheme, profile)
    if match_details:
        lines.append("")
        why_label = _text_variant(
            language,
            "🎯 यह योजना क्यों दिखाई गई:",
            "🎯 Why this scheme was shown:",
            "🎯 Ye scheme kyon dikhayi gayi:",
        )
        lines.append(why_label)

        for field, is_match in match_details.items():
            if not is_match:
                continue
            if field == "age" and profile.age is not None:
                lines.append(f"• {_text_variant(language, 'आयु मेल खाती है', 'Age matches', 'Age match karti hai')}: {profile.age}")
            elif field == "category" and profile.category:
                lines.append(f"• {_text_variant(language, 'श्रेणी मेल खाती है', 'Category matches', 'Category match karti hai')}: {profile.category}")
            elif field == "gender" and profile.gender:
                lines.append(f"• {_text_variant(language, 'लिंग मेल खाता है', 'Gender matches', 'Gender match karta hai')}: {profile.gender}")
            elif field == "income" and profile.annual_income is not None:
                income_str = _format_currency_plain(profile.annual_income, language)
                lines.append(f"• {_text_variant(language, 'आय सीमा के भीतर है', 'Income is within range', 'Income range ke andar hai')}: {income_str}")
            elif field == "income_segment" and profile.annual_income is not None:
                income_str = _format_currency_plain(profile.annual_income, language)
                lines.append(f"• {_text_variant(language, 'आय वर्ग उपयुक्त है', 'Income band fits', 'Income band fit hota hai')}: {income_str}")

    lines.append("")
    lines.append(
        _text_variant(
            language,
            "अगला क्या देखें: दस्तावेज, अस्वीकृति चेतावनियाँ, या आवेदन प्रक्रिया?",
            "What would you like next: documents, rejection warnings, or application steps?",
            "Aage kya dekhna hai: documents, rejection warnings, ya application steps?",
        )
    )
    return "\n".join(lines)


async def _build_document_guidance_text(
    pool: asyncpg.Pool,
    session: Session,
    scheme_id: str,
    language: str,
) -> str:
    """Build focused document guidance for the selected scheme."""
    scheme = await scheme_repo.get_scheme_by_id(pool, scheme_id)
    if not scheme:
        return _text_variant(language, "योजना नहीं मिली।", "Scheme not found.", "Scheme nahi mili.")

    documents = await document_resolver.resolve_documents_for_scheme(pool, scheme.documents_required)
    header = _text_variant(
        language,
        f"📄 {scheme.name_hindi} के दस्तावेज:",
        f"📄 Documents for {scheme.name}:",
        f"📄 {scheme.name} ke documents:",
    )
    lines = [header, ""]

    if not documents:
        lines.append(_text_variant(language, "दस्तावेज जानकारी उपलब्ध नहीं है।", "Document guidance is not available yet.", "Document guidance abhi available nahi hai."))
        return "\n".join(lines)

    for idx, chain in enumerate(documents[:5], 1):
        doc = chain.document
        doc_name = doc.name_hindi if language == "hi" else doc.name
        lines.append(f"{idx}. {doc_name}")
        authority = _truncate_at_word(doc.issuing_authority, 80)
        lines.append(f"   🏛️ {_text_variant(language, 'कहाँ से', 'Where from', 'Kahan se')}: {authority}")
        details = []
        if doc.fee:
            fee_val = f"₹{doc.fee}" if doc.fee.isdigit() else doc.fee
            details.append(f"{_text_variant(language, 'शुल्क', 'Fee', 'Fee')}: {fee_val}")
        if doc.processing_time:
            details.append(f"{_text_variant(language, 'समय', 'Time', 'Time')}: {doc.processing_time}")
        if details:
            lines.append(f"   📋 {' | '.join(details)}")
        if doc.online_portal:
            lines.append(f"   🌐 {_text_variant(language, 'ऑनलाइन', 'Online', 'Online')}: {doc.online_portal}")
        lines.append("")

    lines.append(
        _text_variant(
            language,
            "अगर चाहें तो मैं सामान्य अस्वीकृति चेतावनियाँ भी बता सकता हूँ।",
            "If you want, I can also show the common rejection warnings.",
            "Agar chahein to main common rejection warnings bhi bata sakta hoon.",
        )
    )
    draft = "\n".join(lines)
    return await response_generator.translate_grounded_text_if_needed(
        session,
        draft,
        language,
    )


async def _build_rejection_warnings_text(
    pool: asyncpg.Pool,
    scheme_id: str,
    profile: UserProfile,
    language: str,
) -> str:
    """Build focused rejection-prevention guidance for the selected scheme."""
    scheme = await scheme_repo.get_scheme_by_id(pool, scheme_id)
    if not scheme:
        return _text_variant(language, "योजना नहीं मिली।", "Scheme not found.", "Scheme nahi mili.")

    warnings = await rejection_engine.get_rejection_warnings(pool, scheme_id, profile)
    header = _text_variant(
        language,
        f"⚠️ {scheme.name_hindi} की अस्वीकृति चेतावनियाँ:",
        f"⚠️ Rejection warnings for {scheme.name}:",
        f"⚠️ {scheme.name} ki rejection warnings:",
    )
    lines = [header, ""]

    if not warnings:
        lines.append(
            _text_variant(
                language,
                "फिलहाल अस्वीकृति चेतावनियाँ उपलब्ध नहीं हैं।",
                "No rejection warnings are available right now.",
                "Abhi rejection warnings available nahi hain.",
            )
        )
        return "\n".join(lines)

    severity_icons = {"critical": "🔴", "high": "🟠", "warning": "🟡"}
    for rule in sorted(warnings[:5], key=lambda rule: rule.severity_order):
        icon = severity_icons.get(rule.severity, "⚠️")
        if language == "hi":
            tip = rule.description_hindi or rule.description
        else:
            tip = rule.prevention_tip or rule.description
        lines.append(f"{icon} {_truncate_at_sentence(tip, 220)}")
    lines.append("")
    lines.append(
        _text_variant(
            language,
            "अगर चाहें तो मैं आवेदन प्रक्रिया भी बता सकता हूँ।",
            "If you want, I can also show the application process.",
            "Agar chahein to main application process bhi bata sakta hoon.",
        )
    )
    return "\n".join(lines)


async def _build_application_help_text(
    pool: asyncpg.Pool,
    session: Session,
    scheme_id: str,
    language: str,
) -> str:
    """Build focused application guidance for the selected scheme."""
    scheme = await scheme_repo.get_scheme_by_id(pool, scheme_id)
    if not scheme:
        return _text_variant(language, "योजना नहीं मिली।", "Scheme not found.", "Scheme nahi mili.")

    helpline_phone = scheme.helpline.phone if scheme.helpline else None
    draft = response_generator.generate_application_guidance(
        scheme.name_hindi if language == "hi" else scheme.name,
        scheme.application_url,
        scheme.offline_process,
        application_steps=scheme.application_steps,
        processing_time=scheme.processing_time,
        helpline_phone=helpline_phone,
        language=language,
    )
    return await response_generator.translate_grounded_text_if_needed(
        session,
        draft,
        language,
    )


async def _build_scheme_question_answer_text(
    pool: asyncpg.Pool,
    session: Session,
    scheme_id: str,
    profile: UserProfile,
    user_question: str,
    language: str,
    *,
    active_view: str | None = None,
) -> str:
    """Answer a follow-up question about the active scheme."""
    scheme = await scheme_repo.get_scheme_by_id(pool, scheme_id)
    if not scheme:
        return _text_variant(language, "योजना नहीं मिली।", "Scheme not found.", "Scheme nahi mili.")
    return await response_generator.generate_scheme_question_response(
        session,
        scheme,
        profile,
        user_question,
        language,
        active_view=active_view or session.state.value,
    )


async def _build_handoff_text(
    pool: asyncpg.Pool,
    profile: UserProfile,
    language: str,
) -> str:
    """Build handoff text with nearby office info."""
    offices = []
    if profile.latitude and profile.longitude:
        offices = await office_repo.get_nearest_offices(
            pool, profile.latitude, profile.longitude, 3, "CSC"
        )
    elif profile.district:
        offices = await office_repo.get_offices_by_district(
            pool, profile.district, 3
        )

    lines = [
        _text_variant(
            language,
            "🏛️ आपकी और सहायता के लिए नजदीकी सेवा केंद्र:",
            "🏛️ Nearest service centers for further help:",
            "🏛️ Aur madad ke liye nearest service centers:",
        ),
        "",
    ]

    if offices:
        for office in offices[:3]:
            lines.append(f"📍 {office.name}")
            if office.address:
                lines.append(f"   📫 {office.address[:60]}")
            if office.phone:
                lines.append(f"   📞 {office.phone}")
            if office.working_hours:
                hrs_lbl = _text_variant(language, "समय", "Hours", "Hours")
                lines.append(f"   🕐 {hrs_lbl}: {office.working_hours}")
            lines.append("")
    else:
        no_office = _text_variant(
            language,
            "नजदीकी केंद्र की जानकारी उपलब्ध नहीं है।",
            "No nearby center information available.",
            "Nearby center ki information available nahi hai.",
        )
        lines.append(no_office)
        lines.append("")

    cta = _text_variant(
        language,
        "कृपया अपने सभी दस्तावेज लेकर जाएं।",
        "Please carry all your documents.",
        "Please apne saare documents saath lekar jaiye.",
    )
    lines.append(cta)
    return "\n".join(lines)


def _resolve_scheme_from_text(session: Session, text: str) -> str | None:
    """Try to match user text to a previously presented scheme by number or name."""
    presented = session.presented_schemes
    if not presented:
        return None

    stripped_text = text.strip()
    text_lower = stripped_text.lower()

    # Match by exact number / ordinal: "1", "scheme 2", "second option", "दूसरी योजना", etc.
    exact_number = re.fullmatch(r"\s*(\d+)\s*", stripped_text)
    if exact_number:
        index = int(exact_number.group(1)) - 1
        if 0 <= index < len(presented):
            return presented[index]["id"]

    number_map = {
        0: ("first", "1st", "पहला", "पहली"),
        1: ("second", "2nd", "दूसरा", "दूसरी"),
        2: ("third", "3rd", "तीसरा", "तीसरी"),
        3: ("fourth", "4th", "चौथा", "चौथी"),
        4: ("fifth", "5th", "पांचवां", "पांचवीं"),
    }
    numbered_selection = re.search(
        r"\b(?:scheme|option|number|no\.?|select|choose|pick|details? for|about)\s*(\d+)\b",
        text_lower,
    )
    if numbered_selection:
        index = int(numbered_selection.group(1)) - 1
        if 0 <= index < len(presented):
            return presented[index]["id"]

    if _is_selection_phrase(stripped_text):
        for index, keywords in number_map.items():
            if index >= len(presented):
                continue
            for keyword in keywords:
                if re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text_lower):
                    return presented[index]["id"]

    # Match by scheme name using overlap scoring so natural sentences like
    # "why did you suggest education loan scheme?" still resolve correctly.
    user_tokens = _tokenize_scheme_reference(stripped_text)
    if not user_tokens:
        return None

    best_scheme_id = None
    best_score = 0.0
    for scheme_info in presented:
        name_lower = scheme_info.get("name", "").lower()
        name_hindi = scheme_info.get("name_hindi", "").lower()
        scheme_tokens = _tokenize_scheme_reference(f"{name_lower} {name_hindi}")
        if not scheme_tokens:
            continue

        if text_lower in {name_lower.strip(), name_hindi.strip()}:
            return scheme_info["id"]

        overlap = user_tokens & scheme_tokens
        overlap_ratio = len(overlap) / len(scheme_tokens)

        if len(overlap) >= 2 or overlap_ratio >= 0.6:
            if overlap_ratio > best_score:
                best_scheme_id = scheme_info["id"]
                best_score = overlap_ratio
            elif overlap_ratio == best_score:
                best_scheme_id = None

    return best_scheme_id


def _store_presented_schemes(session: Session, schemes: list[SchemeMatch]) -> Session:
    """Store presented scheme info in session metadata for text-based selection."""
    presented = [
        {"id": m.scheme.id, "name": m.scheme.name, "name_hindi": m.scheme.name_hindi}
        for m in schemes[:5]
    ]
    return session_manager.set_presented_schemes(session, presented)


def _default_scheme_from_session(
    session: Session,
    requested_state: ConversationState | None,
) -> str | None:
    """Resolve a scheme from session context when only one option is active."""
    if session.selected_scheme_id:
        return session.selected_scheme_id

    if requested_state in {
        ConversationState.SCHEME_DETAILS,
        ConversationState.DOCUMENT_GUIDANCE,
        ConversationState.REJECTION_WARNINGS,
        ConversationState.APPLICATION_HELP,
    } and len(session.presented_schemes) == 1:
        return session.presented_schemes[0]["id"]

    return None


# ---------------------------------------------------------------------------
# Affirmative / confirmation detection
# ---------------------------------------------------------------------------

_AFFIRMATIVE_RE = re.compile(
    r"(?i)\b("
    r"yes|yeah|yep|yup|sure|ok|okay|alright|"
    r"absolutely|definitely|of course|please|go ahead|"
    r"let'?s do|proceed|I want|I'?d like|"
    r"haan|haa|ha|ji|bilkul|zaroor|chalo|thik|"
    r"हां|हाँ|जी|ठीक|ज़रूर|बिल्कुल|चलो"
    r")\b"
)


def _is_affirmative(text: str) -> bool:
    """Check if the user's message is a short affirmative/confirmation."""
    words = text.strip().split()
    return len(words) <= 5 and bool(_AFFIRMATIVE_RE.search(text))


# ---------------------------------------------------------------------------
# "I don't know" / skip detection
# ---------------------------------------------------------------------------

_SKIP_PATTERNS = [
    # English patterns (with word boundaries)
    r"\bdon'?t know\b",
    r"\bdo not know\b",
    r"\bno idea\b",
    r"\bnot sure\b",
    r"\bunsure\b",
    r"\bskip\b",
    r"\bpass\b",
    r"\bnext\b",
    r"\bmove on\b",
    # Hinglish patterns (with word boundaries)
    r"\bnahi pata\b",
    r"\bpata nahi\b",
    r"\bmaloom nahi\b",
    r"\bnahi maloom\b",
    # Hindi patterns (without word boundaries for Devanagari)
    r"पता नहीं",
    r"नहीं पता",
    r"मालूम नहीं",
    r"छोड़ो",
    r"अगला",
]

_SKIP_RE = re.compile("|".join(_SKIP_PATTERNS), re.IGNORECASE)


def _wants_to_skip(text: str) -> bool:
    """Check if the user wants to skip the current question."""
    return bool(_SKIP_RE.search(text))


# ---------------------------------------------------------------------------
# Conversation service
# ---------------------------------------------------------------------------

class ConversationService:
    """Main conversation orchestrator."""

    def __init__(self, db_pool: asyncpg.Pool):
        self.pool = db_pool
        self.settings = get_settings()
        self.ai = get_ai_orchestrator()
        # Keep the raw client reachable for existing tests and narrow mocks.
        self.llm = self.ai.llm_client

    async def _save_completed_turn(
        self,
        session: Session,
        *,
        user_message: str,
        response_text: str,
    ) -> Session:
        """Persist a completed user-assistant turn and enqueue memory refresh if due."""
        session = await session_manager.add_message(session, "user", user_message)
        session = await session_manager.add_message(session, "assistant", response_text)
        session = session_manager.mark_turn_completed(session)

        refresh_due = should_refresh_working_memory(
            session,
            trigger_turns=self.settings.ai_memory_refresh_turns,
            trigger_tokens=self.settings.ai_memory_refresh_token_threshold,
        )
        if refresh_due:
            session = session_manager.set_pending_memory_job(session, True)

        await session_manager.save_session(session)

        if not refresh_due:
            return session

        enqueued = await enqueue_memory_refresh(
            session.user_id,
            session.completed_turn_count,
        )
        if enqueued:
            return session

        logger.warning(
            "Memory refresh queue unavailable for user=%s turn=%s",
            session.user_id,
            session.completed_turn_count,
        )
        session = session_manager.set_pending_memory_job(session, False)
        await session_manager.save_session(session)
        return session

    async def _build_command_response(
        self,
        session: Session,
        *,
        user_message: str,
        response_text: str,
        language: str,
        inline_keyboard: list[list[dict[str, str]]] | None = None,
    ) -> ChatResponse:
        """Persist a deterministic command turn and return a response."""
        await self._save_completed_turn(
            session,
            user_message=user_message,
            response_text=response_text,
        )
        return ChatResponse(
            text=response_text,
            next_state=session.state.value,
            language=language,
            inline_keyboard=inline_keyboard,
        )

    async def _render_state_snapshot(
        self,
        session: Session,
        language: str,
    ) -> tuple[str, list[list[dict[str, str]]] | None]:
        """Render the user's current context in a chosen language."""
        profile = session.user_profile
        state = session.state

        if state == ConversationState.GREETING:
            return response_generator.generate_greeting_response(language), None

        if state in {
            ConversationState.SITUATION_UNDERSTANDING,
            ConversationState.PROFILE_COLLECTION,
        }:
            next_question = profile_extractor.get_next_question(
                profile,
                language,
                session.skipped_fields,
            )
            if next_question:
                return next_question, None
            if state == ConversationState.SITUATION_UNDERSTANDING or not profile.life_event:
                return response_generator.generate_clarification_response(
                    "life_event",
                    language,
                ), None
            return response_generator.generate_help_response(language), None

        if state == ConversationState.SCHEME_PRESENTATION:
            selection_text = _build_presented_scheme_selection_text(
                session.presented_schemes,
                language,
            )
            if selection_text:
                return selection_text, format_presented_scheme_keyboard(
                    session.presented_schemes,
                    language,
                )
            return response_generator.generate_scheme_selection_response(language), None

        if state == ConversationState.SCHEME_DETAILS and session.selected_scheme_id:
            return await _build_scheme_details_text(
                self.pool,
                session.selected_scheme_id,
                profile,
                language,
            ), None

        if state == ConversationState.DOCUMENT_GUIDANCE and session.selected_scheme_id:
            return await _build_document_guidance_text(
                self.pool,
                session,
                session.selected_scheme_id,
                language,
            ), None

        if state == ConversationState.REJECTION_WARNINGS and session.selected_scheme_id:
            return await _build_rejection_warnings_text(
                self.pool,
                session.selected_scheme_id,
                profile,
                language,
            ), None

        if state == ConversationState.APPLICATION_HELP and session.selected_scheme_id:
            return await _build_application_help_text(
                self.pool,
                session,
                session.selected_scheme_id,
                language,
            ), None

        if state == ConversationState.CSC_HANDOFF:
            return await _build_handoff_text(self.pool, profile, language), None

        return response_generator.generate_help_response(language), None

    async def handle_message(self, request: ChatRequest) -> ChatResponse:
        """Handle incoming user message and return response.

        Main flow:
        1. Load/create session
        2. Sanitize input
        3. LLM analysis (intent + life_event + entities in one call)
        4. Update profile (immutable merge)
        5. FSM transition
        6. Execute state-specific logic with formatted responses
        7. Save session
        """
        # 1. Load or create session
        session = await session_manager.get_or_create_session(request.user_id)

        # 2. Sanitize input
        user_message = sanitize_input(request.message)
        if not user_message:
            return ChatResponse(
                text="मुझे आपका संदेश समझ नहीं आया। कृपया दोबारा लिखें।",
                language=session.language_preference if session.language_preference != "auto" else "hi",
            )

        command = _extract_supported_command(user_message)

        # Handle /start command — always reset and greet
        if command == "start":
            session = session_manager.reset_session(session)
            lang = session.language_preference if session.language_preference != "auto" else "hi"
            response_text = response_generator.generate_greeting_response(lang)
            return await self._build_command_response(
                session,
                user_message=user_message,
                response_text=response_text,
                language=lang,
            )

        # Handle /help command without invoking the LLM.
        if command == "help":
            response_language = _command_response_language(session)
            response_text = response_generator.generate_help_response(
                session.language_preference,
                has_active_scheme=bool(
                    session.selected_scheme_id and session.state in SCHEME_CONTEXT_STATES
                ),
            )
            return await self._build_command_response(
                session,
                user_message=user_message,
                response_text=response_text,
                language=response_language,
                inline_keyboard=format_language_keyboard(session.language_preference),
            )

        # Handle callback data (scheme selection via inline keyboard)
        if command == "language":
            response_language = _command_response_language(session)
            response_text = response_generator.generate_language_selection_response(
                session.language_preference
            )
            return await self._build_command_response(
                session,
                user_message=user_message,
                response_text=response_text,
                language=response_language,
                inline_keyboard=format_language_keyboard(session.language_preference),
            )

        # Handle callback data (scheme selection via inline keyboard)
        if request.message_type == "callback" and request.callback_data:
            return await self._handle_callback(session, request.callback_data)

        # 3. Analyze message via LLM (intent + entities only)
        explicit_language = _detect_explicit_language_request(user_message)
        inferred_turn_language = explicit_language or _infer_text_language(user_message)
        llm_session_language = (
            session.language_preference
            if session.language_locked and session.language_preference != "auto"
            else inferred_turn_language
        )
        conversation_history = session_manager.get_conversation_history(
            session,
            include_assistant=bool(session.currently_asking),
        )

        analysis = await self.ai.analyze_message(
            session=session,
            user_message=user_message,
            conversation_history=conversation_history,
            system_prompt=get_system_prompt(),
            session_language=llm_session_language,
        )

        intent = analysis.get("intent", "unknown")
        detected_life_event = analysis.get("life_event")
        extracted_fields = analysis.get("extracted_fields", {})
        detected_language = _normalize_language(
            analysis.get("language", llm_session_language)
        )
        selected_scheme_id = analysis.get("selected_scheme_id")
        llm_response_text = analysis.get("response_text")
        llm_action = analysis.get("action")

        # Deterministic extraction fallback for common profile fields.
        rule_based_fields = profile_extractor.extract_by_patterns(
            user_message,
            current_field=session.currently_asking,
        )
        extracted_fields = {**extracted_fields, **rule_based_fields}

        resolved_scheme_id = selected_scheme_id or _resolve_scheme_from_text(
            session, user_message
        )
        action = _detect_action_override(
            user_message,
            session.state,
            session.currently_asking,
            resolved_scheme_id,
            session.selected_scheme_id,
        ) or llm_action
        has_scheme_context = bool(
            resolved_scheme_id
            or session.selected_scheme_id
            or session.presented_schemes
        )
        if _should_answer_scheme_question(
            user_message,
            session.state,
            action,
            resolved_scheme_id,
            session.selected_scheme_id,
            has_scheme_context,
        ):
            action = "answer_scheme_question"

        # Contextual extraction: if the bot asked for a specific field and
        # the user replies with a bare number, interpret it in context.
        # This is the deterministic guardrail — catches cases where both the
        # LLM and the regex missed an obvious contextual answer.
        currently_asking = session.currently_asking
        if currently_asking and currently_asking not in extracted_fields:
            bare_match = re.match(r"^\s*(\d+)\s*$", user_message.strip())
            if bare_match:
                num = int(bare_match.group(1))
                if currently_asking == "age" and 1 <= num <= 120:
                    extracted_fields["age"] = num
                elif currently_asking == "annual_income" and num > 0:
                    extracted_fields["annual_income"] = num

        # Deterministic life-event fallback when LLM returns null/unknown.
        if not detected_life_event:
            detected_life_event = life_event_classifier.classify_by_keywords(user_message)

        # Update language preference. Explicit requests are sticky; otherwise we
        # keep using the latest observed language without locking the session.
        language_changed = False
        if explicit_language:
            language_changed = session.language_preference != explicit_language
            session = session_manager.set_language(
                session,
                explicit_language,
                locked=True,
            )
            lang = explicit_language
        else:
            if session.language_locked and session.language_preference != "auto":
                lang = session.language_preference
            else:
                previous_language = session.language_preference
                if (
                    detected_language != previous_language
                    or session.language_preference == "auto"
                ):
                    session = session_manager.set_language(
                        session,
                        detected_language,
                        locked=False,
                    )
                    language_changed = previous_language != detected_language
                lang = (
                    session.language_preference
                    if session.language_preference != "auto"
                    else detected_language
                )

        # If user switches language while still in GREETING and hasn't provided
        # any substantive profile info, re-greet in the new language so they
        # feel welcomed before we start collecting information.
        if (
            language_changed
            and session.state == ConversationState.GREETING
            and not detected_life_event
            and not extracted_fields
        ):
            response_text = response_generator.generate_greeting_response(lang)
            await self._save_completed_turn(
                session,
                user_message=user_message,
                response_text=response_text,
            )
            return ChatResponse(
                text=response_text,
                next_state=ConversationState.GREETING.value,
                language=lang,
            )

        # Explicitly reset full session context on goodbye / restart.
        if action == "start_over":
            session = session_manager.reset_session(session, preserve_language=True)
            response_text = response_generator.generate_greeting_response(lang)
            await self._save_completed_turn(
                session,
                user_message=user_message,
                response_text=response_text,
            )

            return ChatResponse(
                text=response_text,
                next_state=ConversationState.GREETING.value,
                language=lang,
            )

        if intent == "goodbye" or action == "goodbye":
            # Send farewell, not a greeting — the user said goodbye.
            response_text = response_generator.generate_farewell_response(lang)
            session = session_manager.reset_session(session, preserve_language=True)
            await self._save_completed_turn(
                session,
                user_message=user_message,
                response_text=response_text,
            )

            return ChatResponse(
                text=response_text,
                next_state=ConversationState.GREETING.value,
                language=lang,
            )

        # 4. Update profile with extracted fields
        before_profile = session.user_profile
        if extracted_fields:
            new_profile = UserProfile(**{
                k: v for k, v in extracted_fields.items() if v is not None
            })
            session = session_manager.update_profile(session, new_profile)

        # Allow topic / life-event replacement instead of trapping the user in
        # the original topic until /start.
        if _should_update_life_event(
            session,
            detected_life_event,
            extracted_fields,
            action,
            user_message,
        ):
            session = session_manager.update_profile(
                session,
                UserProfile(life_event=detected_life_event)
            )
        profile = session.user_profile
        changed_fields = _matching_field_changes(before_profile, profile)
        profile_changed = bool(changed_fields)
        matching_inputs_changed = bool(changed_fields & MATCH_RELEVANT_FIELDS)

        # Clear the no-match guard when profile actually changes, so
        # the next FSM cycle is allowed to re-trigger matching.
        if profile_changed:
            session = session_manager.set_awaiting_profile_change(session, False)
            session = session_manager.set_skipped_fields(
                session,
                [field for field in session.skipped_fields if field not in changed_fields],
            )
            if "life_event" in changed_fields and before_profile.life_event is not None:
                session = session_manager.clear_selection(session)
                session = session_manager.set_presented_schemes(session, [])
                session = session_manager.set_currently_asking(session, None)
                session = session_manager.set_skipped_fields(session, [])
            elif (
                matching_inputs_changed
                and session.selected_scheme_id
                and not _should_preserve_scheme_context_action(action)
            ):
                session = session_manager.clear_selection(session)
                session = session_manager.set_presented_schemes(session, [])

        # 5. Determine next FSM state
        requested_state = _requested_scheme_view(
            user_message,
            action,
            session.state,
            has_selected_scheme=bool(session.selected_scheme_id),
            resolved_scheme_id=resolved_scheme_id,
            active_scheme_id=session.selected_scheme_id,
        )
        next_state = fsm.determine_next_state(
            current_state=session.state,
            profile=profile,
            intent=intent,
            selected_scheme_id=resolved_scheme_id,
            has_selected_scheme=bool(session.selected_scheme_id),
            action=action,
            requested_state=requested_state,
        )

        # Prevent re-matching loop: if the previous turn got zero results
        # and the user hasn't provided new profile information, don't
        # auto-trigger matching again — stay in collection so the LLM
        # can respond naturally and guide the user.
        if (
            next_state == ConversationState.SCHEME_MATCHING
            and session.awaiting_profile_change
            and not profile_changed
        ):
            next_state = (
                ConversationState.PROFILE_COLLECTION
                if profile.life_event
                else ConversationState.SITUATION_UNDERSTANDING
            )

        if _should_refresh_matches_after_profile_change(
            session=session,
            profile=profile,
            matching_inputs_changed=matching_inputs_changed,
            action=action,
            requested_state=requested_state,
        ):
            next_state = ConversationState.SCHEME_MATCHING

        # Affirmative on a scheme follow-up moves into application guidance.
        if (
            next_state == ConversationState.SCHEME_DETAILS
            and session.state in {
                ConversationState.SCHEME_DETAILS,
                ConversationState.DOCUMENT_GUIDANCE,
                ConversationState.REJECTION_WARNINGS,
            }
            and _is_affirmative(user_message)
        ):
            next_state = ConversationState.APPLICATION_HELP

        # 6. Execute state-specific logic
        schemes: list[SchemeMatch] = []
        documents: list[DocumentChain] = []
        warnings = []
        offices = []
        inline_keyboard = None
        response_text = ""

        match next_state:

            case ConversationState.GREETING:
                # Deliver warm greeting (now reachable for greeting intent)
                if session.state == ConversationState.CSC_HANDOFF:
                    session = session_manager.reset_session(session)
                response_text = response_generator.generate_greeting_response(lang)

            case ConversationState.SITUATION_UNDERSTANDING:
                if profile.life_event:
                    next_state = (
                        ConversationState.SCHEME_MATCHING
                        if profile.is_complete_for_matching
                        else ConversationState.PROFILE_COLLECTION
                    )
                    if next_state == ConversationState.SCHEME_MATCHING:
                        next_state, response_text, schemes, inline_keyboard, session = (
                            await self._run_matching(profile, user_message, session, lang)
                        )
                    else:
                        response_text = (
                            profile_extractor.get_next_question(
                                profile,
                                lang,
                                session.skipped_fields,
                            )
                            or response_generator.generate_clarification_response("age", lang)
                        )
                        next_field = profile_extractor.get_next_missing_field(
                            profile,
                            session.skipped_fields,
                        )
                        session = session_manager.set_currently_asking(session, next_field)
                else:
                    response_text = (
                        llm_response_text
                        or response_generator.generate_clarification_response("life_event", lang)
                    )
                    session = session_manager.set_currently_asking(session, "life_event")

            case ConversationState.PROFILE_COLLECTION:
                # LLM-first: prefer the natural LLM response so the bot
                # sounds human.  But guard against the LLM re-asking for a
                # field that our deterministic extraction already captured
                # this turn (the LLM generated response_text before the
                # rule-based layer ran, so it can be stale).
                awaiting_profile_change_guard = (
                    session.awaiting_profile_change
                    and not profile_changed
                    and profile.is_complete_for_matching
                )
                if not profile.life_event:
                    next_state = ConversationState.SITUATION_UNDERSTANDING
                    response_text = response_generator.generate_clarification_response(
                        "life_event",
                        lang,
                    )
                    session = session_manager.set_currently_asking(session, "life_event")
                else:
                    next_question = profile_extractor.get_next_question(
                        profile,
                        lang,
                        session.skipped_fields,
                    )
                    next_field = profile_extractor.get_next_missing_field(
                        profile,
                        session.skipped_fields,
                    )
                    previously_asking = session.currently_asking

                    # ---------- SKIP HANDLING ----------
                    # If user says "I don't know" or wants to skip, move to the next field
                    # or try to match with partial profile.
                    if action == "ask_field_reason" and previously_asking:
                        response_text = response_generator.generate_field_reason_response(
                            previously_asking,
                            lang,
                        )
                        session = session_manager.set_currently_asking(
                            session,
                            previously_asking,
                        )
                    elif action == "clarify_field" and previously_asking:
                        response_text = response_generator.generate_field_help_response(
                            previously_asking,
                            lang,
                        )
                        session = session_manager.set_currently_asking(
                            session,
                            previously_asking,
                        )
                    elif action == "skip_field" and previously_asking:
                        # Mark this field as skipped so we don't ask again
                        skipped = list(session.skipped_fields)
                        if previously_asking not in skipped:
                            skipped.append(previously_asking)
                        session = session_manager.set_skipped_fields(session, skipped)

                        # Find the next field that hasn't been skipped
                        next_unskipped = profile_extractor.get_next_missing_field(
                            profile,
                            skipped,
                        )

                        if next_unskipped:
                            # Move to next field
                            session = session_manager.set_currently_asking(
                                session,
                                next_unskipped,
                            )
                            response_text = profile_extractor.get_next_question(
                                profile,
                                lang,
                                skipped,
                            ) or ""
                        else:
                            # No more fields to ask — try matching with partial profile
                            if awaiting_profile_change_guard:
                                response_text = response_generator.generate_no_schemes_response(lang)
                            else:
                                next_state, response_text, schemes, inline_keyboard, session = (
                                    await self._run_matching(profile, user_message, session, lang)
                                )
                            session = session_manager.set_currently_asking(session, None)
                    else:
                        # Normal flow: validation and LLM response handling

                        # ---------- INPUT VALIDATION ----------
                        # If we were asking for a specific field and the user's response
                        # wasn't successfully extracted, check if it was an invalid attempt.
                        validation_error = None
                        if previously_asking and previously_asking not in extracted_fields:
                            is_valid, error_type = profile_extractor.validate_field_response(
                                previously_asking, user_message, extracted_fields
                            )
                            if not is_valid and error_type:
                                validation_error = error_type

                        translated_reask = (
                            explicit_language is not None
                            and previously_asking is not None
                            and not extracted_fields
                            and not detected_life_event
                        )
                        use_llm = bool(llm_response_text)
                        if (
                            use_llm
                            and profile_changed
                            and previously_asking
                            and previously_asking in extracted_fields
                            and next_question
                        ):
                            # The rule-based layer caught the field the LLM missed.
                            # The LLM's response_text likely re-asks for it — discard.
                            use_llm = False
                        if (
                            use_llm
                            and previously_asking
                            and previously_asking not in extracted_fields
                            and action not in {
                                "ask_field_reason",
                                "clarify_field",
                                "skip_field",
                                "change_language",
                                "start_over",
                                "request_handoff",
                            }
                            and not validation_error
                        ):
                            # When we are waiting for a specific field, prefer a deterministic
                            # re-ask over letting the LLM freewheel into another topic.
                            use_llm = False

                        # If there was a validation error, provide helpful guidance
                        # instead of just repeating the question or using LLM response.
                        if validation_error:
                            response_text = profile_extractor.get_validation_re_prompt(
                                previously_asking, validation_error, lang
                            )
                            # Don't change currently_asking — we're still asking for the same field
                        elif translated_reask:
                            response_text = profile_extractor.get_next_question(
                                profile,
                                lang,
                                session.skipped_fields,
                            ) or ""
                        elif use_llm:
                            response_text = llm_response_text
                        elif next_question:
                            response_text = next_question
                        else:
                            # All fields filled but FSM didn't trigger MATCHING
                            # (edge case safety) — run matching inline
                            if awaiting_profile_change_guard:
                                response_text = response_generator.generate_no_schemes_response(lang)
                            else:
                                next_state, response_text, schemes, inline_keyboard, session = (
                                    await self._run_matching(profile, user_message, session, lang)
                                )

                        # Track which field we're asking about so the next turn
                        # can use it for contextual extraction + LLM prompting.
                        # Keep tracking the same field if there was a validation error.
                        if validation_error or translated_reask:
                            # Keep currently_asking the same — we're still asking for this field
                            session = session_manager.set_currently_asking(
                                session,
                                previously_asking,
                            )
                        elif next_field:
                            session = session_manager.set_currently_asking(
                                session,
                                next_field,
                            )
                        else:
                            session = session_manager.set_currently_asking(session, None)

            case ConversationState.SCHEME_MATCHING:
                # Run scheme matching and build formatted response
                next_state, response_text, schemes, inline_keyboard, session = (
                    await self._run_matching(profile, user_message, session, lang)
                )
                # Clear field tracking only when we successfully moved into
                # scheme presentation; clarification/no-match paths keep it.
                if next_state == ConversationState.SCHEME_PRESENTATION:
                    session = session_manager.set_currently_asking(session, None)

            case ConversationState.SCHEME_PRESENTATION:
                scheme_id = resolved_scheme_id or _default_scheme_from_session(
                    session,
                    requested_state,
                )
                if action == "answer_scheme_question" and scheme_id:
                    session = session_manager.select_scheme(session, scheme_id)
                    next_state = ConversationState.SCHEME_DETAILS
                    response_text = await _build_scheme_question_answer_text(
                        self.pool,
                        session,
                        scheme_id,
                        profile,
                        user_message,
                        lang,
                        active_view=ConversationState.SCHEME_DETAILS.value,
                    )
                elif requested_state == ConversationState.DOCUMENT_GUIDANCE and scheme_id:
                    session = session_manager.select_scheme(session, scheme_id)
                    next_state = ConversationState.DOCUMENT_GUIDANCE
                    response_text = await _build_document_guidance_text(
                        self.pool,
                        session,
                        scheme_id,
                        lang,
                    )
                elif requested_state == ConversationState.REJECTION_WARNINGS and scheme_id:
                    session = session_manager.select_scheme(session, scheme_id)
                    next_state = ConversationState.REJECTION_WARNINGS
                    response_text = await _build_rejection_warnings_text(
                        self.pool,
                        scheme_id,
                        profile,
                        lang,
                    )
                elif requested_state == ConversationState.APPLICATION_HELP and scheme_id:
                    session = session_manager.select_scheme(session, scheme_id)
                    next_state = ConversationState.APPLICATION_HELP
                    response_text = await _build_application_help_text(
                        self.pool,
                        session,
                        scheme_id,
                        lang,
                    )
                elif scheme_id:
                    session = session_manager.select_scheme(session, scheme_id)
                    next_state = ConversationState.SCHEME_DETAILS
                    response_text = await _build_scheme_details_text(
                        self.pool,
                        scheme_id,
                        profile,
                        lang,
                    )
                else:
                    # Re-present schemes with keyboard
                    schemes = await scheme_matcher.match_schemes(
                        pool=self.pool,
                        profile=profile,
                        query_text=user_message,
                    )
                    if schemes:
                        session = _store_presented_schemes(session, schemes)
                        inline_keyboard = format_inline_keyboard(schemes, lang)
                    select_prompt = response_generator.generate_scheme_selection_response(lang)
                    response_text = select_prompt

            case ConversationState.SCHEME_DETAILS:
                scheme_id = resolved_scheme_id or _default_scheme_from_session(
                    session,
                    next_state,
                )
                if not scheme_id:
                    # No scheme selected — fall back to presenting
                    next_state = ConversationState.SCHEME_PRESENTATION
                    select_prompt = _text_variant(
                        lang,
                        "कृपया पहले एक योजना चुनें।",
                        "Please select a scheme first.",
                        "Please pehle ek scheme select kijiye.",
                    )
                    response_text = select_prompt
                else:
                    if scheme_id != session.selected_scheme_id:
                        session = session_manager.select_scheme(session, scheme_id)
                    if action == "answer_scheme_question":
                        response_text = await _build_scheme_question_answer_text(
                            self.pool,
                            session,
                            scheme_id,
                            profile,
                            user_message,
                            lang,
                        )
                    else:
                        response_text = await _build_scheme_details_text(
                            self.pool, scheme_id, profile, lang
                        )

            case ConversationState.DOCUMENT_GUIDANCE:
                scheme_id = resolved_scheme_id or _default_scheme_from_session(
                    session,
                    next_state,
                )
                if not scheme_id:
                    next_state = ConversationState.SCHEME_PRESENTATION
                    response_text = _text_variant(
                        lang,
                        "कृपया पहले एक योजना चुनें।",
                        "Please select a scheme first.",
                        "Please pehle ek scheme select kijiye.",
                    )
                else:
                    if scheme_id != session.selected_scheme_id:
                        session = session_manager.select_scheme(session, scheme_id)
                    if action == "answer_scheme_question":
                        response_text = await _build_scheme_question_answer_text(
                            self.pool,
                            session,
                            scheme_id,
                            profile,
                            user_message,
                            lang,
                        )
                    else:
                        response_text = await _build_document_guidance_text(
                            self.pool,
                            session,
                            scheme_id,
                            lang,
                        )

            case ConversationState.REJECTION_WARNINGS:
                scheme_id = resolved_scheme_id or _default_scheme_from_session(
                    session,
                    next_state,
                )
                if not scheme_id:
                    next_state = ConversationState.SCHEME_PRESENTATION
                    response_text = _text_variant(
                        lang,
                        "कृपया पहले एक योजना चुनें।",
                        "Please select a scheme first.",
                        "Please pehle ek scheme select kijiye.",
                    )
                else:
                    if scheme_id != session.selected_scheme_id:
                        session = session_manager.select_scheme(session, scheme_id)
                    if action == "answer_scheme_question":
                        response_text = await _build_scheme_question_answer_text(
                            self.pool,
                            session,
                            scheme_id,
                            profile,
                            user_message,
                            lang,
                        )
                    else:
                        response_text = await _build_rejection_warnings_text(
                            self.pool,
                            scheme_id,
                            profile,
                            lang,
                        )

            case ConversationState.APPLICATION_HELP:
                scheme_id = resolved_scheme_id or _default_scheme_from_session(
                    session,
                    next_state,
                )
                if not scheme_id:
                    next_state = ConversationState.SCHEME_PRESENTATION
                    response_text = _text_variant(
                        lang,
                        "कृपया पहले एक योजना चुनें।",
                        "Please select a scheme first.",
                        "Please pehle ek scheme select kijiye.",
                    )
                elif scheme_id != session.selected_scheme_id:
                    session = session_manager.select_scheme(session, scheme_id)
                    next_state = ConversationState.SCHEME_DETAILS
                    response_text = await _build_scheme_details_text(
                        self.pool,
                        scheme_id,
                        profile,
                        lang,
                    )
                else:
                    if action == "answer_scheme_question":
                        response_text = await _build_scheme_question_answer_text(
                            self.pool,
                            session,
                            scheme_id,
                            profile,
                            user_message,
                            lang,
                        )
                    else:
                        response_text = await _build_application_help_text(
                            self.pool,
                            session,
                            scheme_id,
                            lang,
                        )

            case ConversationState.CSC_HANDOFF:
                # Use LLM response if available (more natural), fall back to
                # structured office info when user explicitly requests it.
                if llm_response_text:
                    response_text = llm_response_text
                else:
                    response_text = await _build_handoff_text(self.pool, profile, lang)

            case _:
                response_text = response_generator.generate_greeting_response(lang)

        response_text = await response_generator.ensure_response_language(
            session,
            response_text,
            lang,
        )

        # 7. Update session state and save
        session = session_manager.update_state(session, next_state)
        await self._save_completed_turn(
            session,
            user_message=user_message,
            response_text=response_text,
        )

        return ChatResponse(
            text=response_text,
            schemes=schemes,
            documents=documents,
            rejection_warnings=warnings,
            offices=offices,
            inline_keyboard=inline_keyboard,
            next_state=next_state.value,
            language=lang,
        )

    async def _run_matching(
        self,
        profile: UserProfile,
        user_message: str,
        session: Session,
        lang: str,
    ) -> tuple[ConversationState, str, list[SchemeMatch], list | None, Session]:
        """Run scheme matching and return (next_state, text, schemes, keyboard, session)."""
        logger.info(
            "Running scheme matching for user=%s state=%s profile.life_event=%s age=%s category=%s income=%s",
            session.user_id,
            session.state.value,
            profile.life_event,
            profile.age,
            profile.category,
            profile.annual_income,
        )
        low_context_turn = _is_low_context_matching_turn(session, user_message)
        matching_query_text = (
            _build_matching_focus_text(profile, user_message)
            if low_context_turn
            else user_message
        )
        schemes = await scheme_matcher.match_schemes(
            pool=self.pool,
            profile=profile,
            query_text=matching_query_text,
        )

        if schemes:
            relevance_focus_text = _build_matching_focus_text(profile, user_message)
            candidate_payload = scheme_relevance.build_candidate_payload(schemes)
            judgement = None
            if self.ai.should_run_relevance_judge(schemes):
                try:
                    judgement = await self.ai.judge_scheme_relevance(
                        session=session,
                        user_message=relevance_focus_text,
                        conversation_history=session_manager.get_conversation_history(session),
                        candidate_schemes=candidate_payload,
                        session_language=lang,
                    )
                except Exception as exc:
                    logger.warning(
                        "Scheme relevance judging failed for user=%s: %s",
                        session.user_id,
                        exc,
                    )
            else:
                logger.info(
                    "Skipping AI relevance judge for user=%s top_score=%.3f",
                    session.user_id,
                    schemes[0].deterministic_score,
                )
            relevance = scheme_relevance.apply_relevance_judgement(
                schemes,
                judgement,
                lang,
                profile.life_event,
            )
            if low_context_turn and relevance["should_clarify"]:
                logger.info(
                    "Ignoring low-context relevance clarification for user=%s currently_asking=%s",
                    session.user_id,
                    session.currently_asking,
                )
                relevance["should_clarify"] = False
                relevance["clarification_question"] = None
            schemes = relevance["matches"]
            logger.info(
                "Scheme relevance gate: should_clarify=%s overall_confidence=%s top_ids=%s",
                relevance["should_clarify"],
                relevance["overall_confidence"],
                [match.scheme.id for match in schemes[:3]],
            )

            if relevance["should_clarify"]:
                session = session_manager.set_awaiting_profile_change(session, False)
                session = session_manager.clear_selection(session)
                session = session_manager.set_presented_schemes(session, [])
                session = session_manager.set_currently_asking(session, "life_event")
                return (
                    ConversationState.SITUATION_UNDERSTANDING,
                    relevance["clarification_question"],
                    [],
                    None,
                    session,
                )

            # Clear the no-match flag since we found results
            session = session_manager.set_awaiting_profile_change(session, False)
            session = _store_presented_schemes(session, schemes)
            response_text = _build_scheme_list_text(schemes, profile, lang)
            inline_keyboard = format_inline_keyboard(schemes, lang)
            return ConversationState.SCHEME_PRESENTATION, response_text, schemes, inline_keyboard, session

        # No match — return to collection so the user can explore
        # different areas or update their profile, instead of dumping them
        # into the handoff dead-end.
        no_match_text = response_generator.generate_no_schemes_response(lang)
        session = session_manager.set_awaiting_profile_change(session, True)
        session = session_manager.clear_selection(session)
        session = session_manager.set_presented_schemes(session, [])
        fallback_state = (
            ConversationState.PROFILE_COLLECTION
            if profile.life_event
            else ConversationState.SITUATION_UNDERSTANDING
        )
        return fallback_state, no_match_text, [], None, session

    async def _handle_callback(
        self,
        session: Session,
        callback_data: str,
    ) -> ChatResponse:
        """Handle callback query (scheme selection from inline keyboard)."""
        lang = session.language_preference if session.language_preference != "auto" else "hi"

        if callback_data.startswith("lang:"):
            requested_language = callback_data.replace("lang:", "", 1)
            if requested_language not in {"hi", "en", "hinglish"}:
                return ChatResponse(
                    text="अमान्य भाषा चयन।" if lang == "hi" else "Invalid language selection.",
                    language=lang,
                )

            session = session_manager.set_language(session, requested_language, locked=True)
            state_text, inline_keyboard = await self._render_state_snapshot(
                session,
                requested_language,
            )
            response_text = (
                response_generator.generate_language_changed_response(
                    requested_language,
                    has_active_scheme=bool(
                        session.selected_scheme_id and session.state in SCHEME_CONTEXT_STATES
                    ),
                )
                + "\n\n"
                + state_text
            )
            response_text = await response_generator.ensure_response_language(
                session,
                response_text,
                requested_language,
            )

            session = await session_manager.add_message(session, "assistant", response_text)
            await session_manager.save_session(session)

            return ChatResponse(
                text=response_text,
                next_state=session.state.value,
                language=requested_language,
                inline_keyboard=inline_keyboard,
            )

        if callback_data.startswith("scheme:"):
            scheme_id = callback_data.replace("scheme:", "")

            # Select scheme and transition to scheme overview
            session = session_manager.select_scheme(session, scheme_id)
            session = session_manager.update_state(session, ConversationState.SCHEME_DETAILS)

            # Build scheme overview response
            response_text = await _build_scheme_details_text(
                self.pool, scheme_id, session.user_profile, lang
            )
            response_text = await response_generator.ensure_response_language(
                session,
                response_text,
                lang,
            )

            # Save session
            session = await session_manager.add_message(session, "assistant", response_text)
            await session_manager.save_session(session)

            return ChatResponse(
                text=response_text,
                next_state=ConversationState.SCHEME_DETAILS.value,
                language=lang,
            )

        return ChatResponse(
            text="अमान्य चयन।" if lang == "hi" else "Invalid selection.",
            language=lang,
        )
