"""Language detection and language-dependent phrasing for a turn.

Three response languages are supported end to end: ``hi`` (Devanagari),
``en``, and ``hinglish`` (romanised Hindi). A session may additionally be
``auto``, meaning no language has been observed or chosen yet.

A session language is either *locked* — the user asked for it explicitly and
it sticks — or *unlocked*, in which case it tracks whatever the user last
wrote. The unlocked case is the awkward one: a bare answer like "50000" or
"SC" carries almost no language signal, so
:func:`should_preserve_unlocked_session_language` keeps the previous language
rather than letting a value-only reply flip the conversation into English.
"""

import re

from src.models.session import Session

SUPPORTED_LANGUAGES = frozenset({"hi", "en", "hinglish"})

# Unicode Devanagari block bounds, spelled as codepoints so the range stays
# readable and cannot be mangled by an editor that rewrites literals.
DEVANAGARI_FIRST = chr(0x0900)
DEVANAGARI_LAST = chr(0x097F)

# Above this share of Devanagari among alphabetic characters the text is
# treated as Hindi regardless of what else it contains.
DEVANAGARI_THRESHOLD = 0.3

# Romanised Hindi function words. Their presence in otherwise-Latin text is
# what separates Hinglish from English.
_HINGLISH_MARKERS = (
    "mujhe", "chahiye", "batao", "batayiye", "kyu", "kya", "kaise",
    "sahayata", "madad", "mera", "meri", "mere", "kripya", "hai",
    "hain", "hoon", "saal", "nahi", "nahin", "liye", "bhi", "beti",
    "pati", "patni", "umar", "vidhwa", "bhai", "aap", "karo",
    "karein", "kariye", "baare", "guzar",
)

_LANGUAGE_REQUEST_PATTERNS = {
    "en": (
        r"\benglish\b",
        r"\buse english\b",
        r"\bplease use english\b",
        r"\bi don't understand hindi\b",
    ),
    "hi": (
        r"\bhindi\b",
        r"हिंदी",
        r"\buse hindi\b",
        r"\bhindi language\b",
    ),
    "hinglish": (
        r"\bhinglish\b",
        r"\buse hinglish\b",
        r"\broman hindi\b",
    ),
}

_EMPATHY_MARKERS = (
    "sorry",
    "condolence",
    "dukh",
    "afsos",
    "samvedana",
    "mushkil",
    "खेद",
    "दुख",
    "माफ़",
)

_HUSBAND_TOKENS = ("husband", "pati", "पति")
_WIFE_TOKENS = ("wife", "patni", "पत्नी")

_WORD_RE = re.compile(f"[A-Za-z0-9{DEVANAGARI_FIRST}-{DEVANAGARI_LAST}]+")


def text_variant(language: str, hi: str, en: str, hinglish: str | None = None) -> str:
    """Select a language-specific string, falling back to English."""
    if language == "hi":
        return hi
    if language == "hinglish":
        return hinglish or en
    return en


def normalize_language(language: str | None) -> str:
    """Normalize language codes to supported response variants."""
    if language in SUPPORTED_LANGUAGES:
        return language
    return "hi"


def devanagari_ratio(text: str) -> float:
    """Share of alphabetic characters written in Devanagari, 0.0 if none."""
    devanagari_chars = sum(
        1 for char in text if DEVANAGARI_FIRST <= char <= DEVANAGARI_LAST
    )
    alpha_chars = sum(1 for char in text if char.isalpha())
    if not alpha_chars:
        return 0.0
    return devanagari_chars / alpha_chars


def count_markers(text_lower: str, markers: tuple[str, ...]) -> int:
    """Count whole-word occurrences of romanised Hindi markers."""
    return sum(
        1
        for marker in markers
        if re.search(rf"(?<!\w){re.escape(marker)}(?!\w)", text_lower)
    )


def infer_text_language(text: str) -> str:
    """Infer the user's language from the raw text when no session lock exists."""
    if devanagari_ratio(text) > DEVANAGARI_THRESHOLD:
        return "hi"

    marker_hits = count_markers(text.lower(), _HINGLISH_MARKERS)
    # Short messages give the marker set little to work with, so one hit is
    # enough; longer messages need two to avoid tripping on a stray word.
    threshold = 1 if len(_WORD_RE.findall(text)) <= 4 else 2
    if marker_hits >= threshold:
        return "hinglish"
    return "en"


def preferred_turn_language(inferred_turn_language: str, detected_language: str) -> str:
    """Prefer raw user-language cues for unlocked non-English turns."""
    if inferred_turn_language in {"hi", "hinglish"}:
        return inferred_turn_language
    return detected_language


def looks_like_low_context_field_reply(text: str) -> bool:
    """Return True for short value-style replies that carry weak language signal."""
    stripped = text.strip().lower()
    if not stripped:
        return False

    if re.fullmatch(r"₹?\s*[\d,]+(?:\.\d+)?", stripped):
        return True
    if re.fullmatch(r"(sc|st|obc|ews|general)", stripped, re.IGNORECASE):
        return True

    word_tokens = re.findall(r"[a-z]+", stripped)
    if not re.search(r"\d", stripped):
        return False
    if len(word_tokens) > 12:
        return False

    value_cues = (
        "income",
        "family",
        "annual",
        "year",
        "yearly",
        "around",
        "approx",
        "month",
        "monthly",
        "lakh",
        "lac",
        "rupees",
        "rs",
        "per",
        "hai",
        "age",
        "category",
        "saal",
        "mahina",
        "hazar",
        "hazaar",
        "ka",
        "ki",
        "umar",
        "sal",
    )
    if any(cue in stripped for cue in value_cues):
        return True

    profile_reply_patterns = (
        r"\bi am\b",
        r"\bi'm\b",
        r"\bmain\b",
        r"\bmeri\b",
        r"\bmy\b",
    )
    return any(re.search(pattern, stripped) for pattern in profile_reply_patterns)


def should_preserve_unlocked_session_language(
    session: Session,
    user_message: str,
    inferred_turn_language: str,
) -> bool:
    """Keep the active unlocked Hindi/Hinglish language on low-context field replies."""
    if session.language_locked:
        return False
    if session.language_preference not in {"hi", "hinglish"}:
        return False
    if session.currently_asking is None:
        return False
    if inferred_turn_language in {"hi", "hinglish"}:
        return False
    return looks_like_low_context_field_reply(user_message)


def detect_explicit_language_request(text: str) -> str | None:
    """Detect when the user explicitly asks for a specific language."""
    text_lower = text.lower()
    for language, patterns in _LANGUAGE_REQUEST_PATTERNS.items():
        if any(re.search(pattern, text_lower) for pattern in patterns):
            return language
    return None


def command_response_language(session: Session) -> str:
    """Choose the safest deterministic command response language."""
    return session.language_preference if session.language_preference != "auto" else "en"


def response_conflicts_with_spouse_reference(
    user_message: str,
    response_text: str | None,
) -> bool:
    """Return True when the LLM flips husband/wife relative to user wording."""
    if not response_text:
        return False

    user_lower = user_message.lower()
    response_lower = response_text.lower()

    user_said_husband = any(token in user_lower for token in _HUSBAND_TOKENS)
    user_said_wife = any(token in user_lower for token in _WIFE_TOKENS)
    response_said_husband = any(token in response_lower for token in _HUSBAND_TOKENS)
    response_said_wife = any(token in response_lower for token in _WIFE_TOKENS)

    return bool(
        (user_said_husband and response_said_wife)
        or (user_said_wife and response_said_husband)
    )


def response_has_empathy(text: str) -> bool:
    """Return True when the reply already acknowledges distress or loss."""
    lowered = text.lower()
    return any(marker in lowered for marker in _EMPATHY_MARKERS)


def prepend_death_in_family_empathy(response_text: str, language: str) -> str:
    """Prepend a brief condolence before deterministic widow-flow questions."""
    if not response_text.strip() or response_has_empathy(response_text):
        return response_text

    empathy_prefix = text_variant(
        language,
        "मुझे आपके नुकसान का दुख है।",
        "I am sorry for your loss.",
        "Mujhe aapke nuksaan ka dukh hai.",
    )
    return f"{empathy_prefix}\n\n{response_text}"
