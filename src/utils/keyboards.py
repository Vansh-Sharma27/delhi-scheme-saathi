"""Telegram inline keyboard builders.

Every button carries callback data in one of two forms, both parsed back
out by ``ConversationService._handle_callback``:

- ``scheme:<scheme_id>`` — open a scheme from the presented list
- ``lang:<hi|en|hinglish>`` — switch response language
"""

from src.models.scheme import SchemeMatch

# Telegram wraps long button labels onto several lines and truncates them
# unpredictably on narrow screens, so labels are shortened up front.
MAX_BUTTON_LABEL_LEN = 28
# Matching never presents more than five schemes, and the stored
# presented_schemes list is numbered against the same limit.
MAX_SCHEME_BUTTONS = 5

SUPPORTED_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("hi", "हिंदी"),
    ("en", "English"),
    ("hinglish", "Hinglish"),
)


def _button_label(name: str) -> str:
    """Shorten a scheme name to fit a single Telegram button."""
    if len(name) <= MAX_BUTTON_LABEL_LEN:
        return name
    return name[: MAX_BUTTON_LABEL_LEN - 3] + "..."


def _scheme_rows(
    schemes: list[tuple[str, str]],
) -> list[list[dict[str, str]]]:
    """Build one single-button row per (scheme_id, display_name) pair."""
    return [
        [
            {
                "text": f"{index}. {_button_label(name)}",
                "callback_data": f"scheme:{scheme_id}",
            }
        ]
        for index, (scheme_id, name) in enumerate(schemes[:MAX_SCHEME_BUTTONS], 1)
    ]


def format_inline_keyboard(
    schemes: list[SchemeMatch],
    language: str = "hi",
) -> list[list[dict[str, str]]] | None:
    """Build scheme selection buttons from fresh match results."""
    if not schemes:
        return None

    return _scheme_rows(
        [
            (
                match.scheme.id,
                match.scheme.name_hindi if language == "hi" else match.scheme.name,
            )
            for match in schemes
        ]
    )


def format_presented_scheme_keyboard(
    presented_schemes: list[dict[str, str]],
    language: str = "hi",
) -> list[list[dict[str, str]]] | None:
    """Build scheme selection buttons from the schemes stored on the session.

    Used when replaying the last list without re-running matching, so the
    entries are plain dicts rather than SchemeMatch objects and either name
    may be missing.
    """
    if not presented_schemes:
        return None

    rows: list[tuple[str, str]] = []
    for scheme in presented_schemes:
        preferred = scheme.get("name_hindi") if language == "hi" else scheme.get("name")
        name = preferred or scheme.get("name") or scheme.get("name_hindi") or "Scheme"
        rows.append((scheme["id"], name))
    return _scheme_rows(rows)


def format_language_keyboard(
    active_language: str = "auto",
) -> list[list[dict[str, str]]]:
    """Build the language switcher, ticking the currently active language."""
    return [
        [
            {
                "text": f"{'✓ ' if code == active_language else ''}{label}",
                "callback_data": f"lang:{code}",
            }
        ]
        for code, label in SUPPORTED_LANGUAGES
    ]
