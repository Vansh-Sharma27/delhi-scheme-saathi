"""Resolving which scheme the user is talking about.

After a scheme list is shown, the user can refer back to an entry by
position ("2", "second option"), by name ("education loan scheme"), or by
tapping an inline button. Buttons carry an explicit id; the other two forms
have to be resolved against ``session.presented_schemes``, which stores the
id and both names of the last five schemes shown.
"""

import re

from src.models.scheme import SchemeMatch
from src.models.session import ConversationState, Session
from src.services import session_manager

# States in which a scheme is open, so a follow-up question is about that
# scheme rather than about starting a new search.
SCHEME_CONTEXT_STATES = {
    ConversationState.SCHEME_PRESENTATION,
    ConversationState.SCHEME_DETAILS,
    ConversationState.DOCUMENT_GUIDANCE,
    ConversationState.REJECTION_WARNINGS,
    ConversationState.APPLICATION_HELP,
}

# States where a scheme is already selected, so naming a different scheme
# means switching rather than picking for the first time.
SCHEME_DETAIL_STATES = {
    ConversationState.SCHEME_DETAILS,
    ConversationState.DOCUMENT_GUIDANCE,
    ConversationState.REJECTION_WARNINGS,
    ConversationState.APPLICATION_HELP,
}

# Words that suggest the message is picking something from the list rather
# than making a statement.
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

# Words too generic to identify a scheme by name; dropped before overlap
# scoring so "scheme details please" does not match everything.
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

_ORDINAL_WORDS = {
    0: ("first", "1st", "पहला", "पहली"),
    1: ("second", "2nd", "दूसरा", "दूसरी"),
    2: ("third", "3rd", "तीसरा", "तीसरी"),
    3: ("fourth", "4th", "चौथा", "चौथी"),
    4: ("fifth", "5th", "पांचवां", "पांचवीं"),
}

_NUMBERED_SELECTION_RE = re.compile(
    r"\b(?:scheme|option|number|no\.?|select|choose|pick|details? for|about)\s*(\d+)\b"
)

# A name match needs either two shared significant words or most of the
# scheme's own words; below that the reference is too vague to act on.
_MIN_NAME_TOKEN_OVERLAP = 2
_MIN_NAME_OVERLAP_RATIO = 0.6

MAX_PRESENTED_SCHEMES = 5


def tokenize_scheme_reference(text: str) -> set[str]:
    """Tokenize user text for safe scheme-name matching."""
    tokens = set()
    for token in re.findall(r"[a-z0-9ऀ-ॿ]+", text.lower()):
        if len(token) <= 1:
            continue
        if token in SCHEME_NAME_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def is_selection_phrase(text: str) -> bool:
    """Return True when the message looks like a scheme selection request."""
    stripped = text.strip().lower()
    if re.fullmatch(r"\d+", stripped):
        return True
    if len(stripped.split()) <= 4:
        return True
    return any(cue in stripped for cue in SCHEME_SELECTION_CUES)


def resolved_scheme_matches_active_scheme(
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


def resolve_scheme_from_text(session: Session, text: str) -> str | None:
    """Try to match user text to a previously presented scheme by number or name."""
    presented = session.presented_schemes
    if not presented:
        return None

    stripped_text = text.strip()
    text_lower = stripped_text.lower()

    exact_number = re.fullmatch(r"\s*(\d+)\s*", stripped_text)
    if exact_number:
        index = int(exact_number.group(1)) - 1
        if 0 <= index < len(presented):
            return presented[index]["id"]

    numbered_selection = _NUMBERED_SELECTION_RE.search(text_lower)
    if numbered_selection:
        index = int(numbered_selection.group(1)) - 1
        if 0 <= index < len(presented):
            return presented[index]["id"]

    if is_selection_phrase(stripped_text):
        for index, keywords in _ORDINAL_WORDS.items():
            if index >= len(presented):
                continue
            for keyword in keywords:
                if re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text_lower):
                    return presented[index]["id"]

    return _resolve_by_name(presented, stripped_text, text_lower)


def _resolve_by_name(
    presented: list[dict[str, str]],
    stripped_text: str,
    text_lower: str,
) -> str | None:
    """Match by scheme name using word overlap.

    Overlap scoring rather than substring matching so natural sentences like
    "why did you suggest education loan scheme?" still resolve. A tie between
    two schemes resolves to nothing, because guessing would silently open the
    wrong scheme.
    """
    user_tokens = tokenize_scheme_reference(stripped_text)
    if not user_tokens:
        return None

    best_scheme_id = None
    best_score = 0.0
    for scheme_info in presented:
        name_lower = scheme_info.get("name", "").lower()
        name_hindi = scheme_info.get("name_hindi", "").lower()
        scheme_tokens = tokenize_scheme_reference(f"{name_lower} {name_hindi}")
        if not scheme_tokens:
            continue

        if text_lower in {name_lower.strip(), name_hindi.strip()}:
            return scheme_info["id"]

        overlap = user_tokens & scheme_tokens
        overlap_ratio = len(overlap) / len(scheme_tokens)

        if len(overlap) >= _MIN_NAME_TOKEN_OVERLAP or overlap_ratio >= _MIN_NAME_OVERLAP_RATIO:
            if overlap_ratio > best_score:
                best_scheme_id = scheme_info["id"]
                best_score = overlap_ratio
            elif overlap_ratio == best_score:
                best_scheme_id = None

    return best_scheme_id


def validated_selected_scheme_id(
    session: Session,
    selected_scheme_id: str | None,
) -> str | None:
    """Trust analysis-selected scheme IDs only when they match active scheme context."""
    if not selected_scheme_id:
        return None

    valid_ids = {
        scheme.get("id")
        for scheme in session.presented_schemes
        if scheme.get("id")
    }
    if session.selected_scheme_id:
        valid_ids.add(session.selected_scheme_id)
    if selected_scheme_id in valid_ids:
        return selected_scheme_id
    return None


def store_presented_schemes(session: Session, schemes: list[SchemeMatch]) -> Session:
    """Store presented scheme info in session metadata for text-based selection."""
    presented = [
        {"id": m.scheme.id, "name": m.scheme.name, "name_hindi": m.scheme.name_hindi}
        for m in schemes[:MAX_PRESENTED_SCHEMES]
    ]
    return session_manager.set_presented_schemes(session, presented)


def default_scheme_from_session(
    session: Session,
    requested_state: ConversationState | None,
) -> str | None:
    """Resolve a scheme from session context when only one option is active."""
    if session.selected_scheme_id:
        return session.selected_scheme_id

    if requested_state in SCHEME_DETAIL_STATES and len(session.presented_schemes) == 1:
        return session.presented_schemes[0]["id"]

    return None
