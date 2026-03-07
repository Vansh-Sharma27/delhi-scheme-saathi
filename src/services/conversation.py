"""Main conversation orchestrator.

This is the brain of Delhi Scheme Saathi. It:
1. Loads/creates session
2. Analyzes user input via LLM
3. Updates user profile
4. Executes FSM transitions
5. Runs state-specific logic (matching, document resolution, etc.)
6. Generates structured response using formatters (not LLM)
7. Saves session state
"""

import logging
import re
from typing import Any

import asyncpg

from src.db import scheme_repo, office_repo
from src.integrations.llm_client import get_llm_client
from src.models.api import ChatRequest, ChatResponse
from src.models.document import DocumentChain
from src.models.scheme import SchemeMatch
from src.models.session import ConversationState, Session, UserProfile
from src.prompts.loader import get_system_prompt
from src.services import (
    fsm,
    life_event_classifier,
    profile_extractor,
    scheme_matcher,
    document_resolver,
    rejection_engine,
    response_generator,
    session_manager,
)
from src.utils.validators import sanitize_input
from src.utils.formatters import format_inline_keyboard

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


def _detect_action_override(
    text: str,
    current_state: ConversationState,
    currently_asking: str | None,
    resolved_scheme_id: str | None,
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
    if _wants_to_skip(text):
        return "skip_field"
    if resolved_scheme_id:
        return "switch_scheme" if current_state in {
            ConversationState.DETAILS,
            ConversationState.APPLICATION,
        } else "select_scheme"
    if re.search(r"\b(apply|application|apply kar|apply kare|अवेदन|आवेदन)\b", text_lower):
        return "request_application"
    if re.search(r"\b(csc|human help|service center|service centre|nearest center|operator|contact center)\b", text_lower):
        return "request_handoff"
    if re.search(r"\b(detail|details|document|documents|eligibility|benefit|explain|translate|translation)\b", text_lower):
        return "request_details"
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
        for event in scheme.life_events:
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
    """Build rich plain-text scheme details with documents and warnings.

    Visual hierarchy:
    - Section dividers (───) between major blocks
    - Blank lines for breathing room
    - No harsh truncation on rejection descriptions
    """
    scheme = await scheme_repo.get_scheme_by_id(pool, scheme_id)
    if not scheme:
        return _text_variant(language, "योजना नहीं मिली।", "Scheme not found.", "Scheme nahi mili.")

    DIVIDER = "───────────────────"

    icon = "📋"
    for event in scheme.life_events:
        if event in LIFE_EVENT_ICONS:
            icon = LIFE_EVENT_ICONS[event]
            break

    name = scheme.name_hindi if language == "hi" else scheme.name
    lines = [f"{icon} {name}", ""]

    # Description (word-boundary-aware truncation)
    desc = scheme.description_hindi if language == "hi" else scheme.description
    desc = _truncate_at_word(desc, 400)
    lines.append(desc)
    lines.append("")

    # Benefits + Eligibility block
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
    if elig.categories:
        cat_lbl = _text_variant(language, "श्रेणी", "Category", "Category")
        elig_parts.append(f"{cat_lbl}: {', '.join(elig.categories)}")
    if elig_parts:
        elig_label = _text_variant(language, "पात्रता", "Eligibility", "Eligibility")
        lines.append(f"✅ {elig_label}: {' | '.join(elig_parts)}")

    # ── Documents section ──
    documents = await document_resolver.resolve_documents_for_scheme(
        pool, scheme.documents_required
    )
    if documents:
        lines.append("")
        lines.append(DIVIDER)
        doc_header = _text_variant(
            language,
            "📄 आवश्यक दस्तावेज:",
            "📄 Required Documents:",
            "📄 Required documents:",
        )
        lines.append(doc_header)
        lines.append("")

        for idx, chain in enumerate(documents[:5], 1):
            doc = chain.document
            doc_name = doc.name_hindi if language == "hi" else doc.name
            lines.append(f"  {idx}. {doc_name}")

            # Where to get the document (wrapped for readability)
            where_lbl = _text_variant(language, "कहाँ से", "Where", "Kahan se")
            authority = doc.issuing_authority
            if len(authority) > 60:
                authority = _truncate_at_word(authority, 60)
            lines.append(f"     🏛️ {where_lbl}: {authority}")

            # Fee and time on separate line if present
            details = []
            if doc.fee:
                fee_lbl = _text_variant(language, "शुल्क", "Fee", "Fee")
                fee_val = f"₹{doc.fee}" if doc.fee.isdigit() else doc.fee
                details.append(f"{fee_lbl}: {fee_val}")
            if doc.processing_time:
                time_lbl = _text_variant(language, "समय", "Time", "Time")
                details.append(f"{time_lbl}: {doc.processing_time}")
            if details:
                lines.append(f"     📋 {' | '.join(details)}")

            if doc.online_portal:
                online_lbl = _text_variant(language, "ऑनलाइन", "Online", "Online")
                lines.append(f"     🌐 {online_lbl}: {doc.online_portal}")

            lines.append("")  # blank line between documents

    # ── Rejection warnings section (condensed: tips only) ──
    warnings = await rejection_engine.get_rejection_warnings(pool, scheme_id, profile)
    if warnings:
        lines.append(DIVIDER)
        warn_header = _text_variant(
            language,
            "⚠️ अस्वीकृति से बचें:",
            "⚠️ Tips to Avoid Rejection:",
            "⚠️ Rejection se bachne ke tips:",
        )
        lines.append(warn_header)
        lines.append("")

        severity_icons = {"critical": "🔴", "high": "🟠", "warning": "🟡"}
        for rule in sorted(warnings[:3], key=lambda r: r.severity_order):
            sev_icon = severity_icons.get(rule.severity, "⚠️")
            # Show only the actionable tip (concise), not the full rejection
            # description paragraph — keeps the message scannable.
            if rule.prevention_tip:
                tip = rule.prevention_tip
                if len(tip) > 160:
                    tip = tip[:157] + "..."
                lines.append(f"  {sev_icon} {tip}")
            else:
                desc = rule.description_hindi if language == "hi" else rule.description
                if len(desc) > 160:
                    desc = desc[:157] + "..."
                lines.append(f"  {sev_icon} {desc}")
            lines.append("")  # blank line between warnings

    # ── Application section ──
    if scheme.application_url or scheme.offline_process:
        lines.append(DIVIDER)
        apply_header = _text_variant(
            language,
            "📝 आवेदन कैसे करें:",
            "📝 How to Apply:",
            "📝 Apply kaise karein:",
        )
        lines.append(apply_header)
        lines.append("")
    if scheme.application_url:
        apply_lbl = _text_variant(language, "ऑनलाइन", "Online", "Online")
        lines.append(f"🔗 {apply_lbl}: {scheme.application_url}")
    if scheme.offline_process:
        offline_lbl = _text_variant(language, "ऑफलाइन", "Offline", "Offline")
        lines.append(f"🏛️ {offline_lbl}: {scheme.offline_process}")

    lines.append("")
    cta = _text_variant(
        language,
        "क्या आप इस योजना के लिए आवेदन करना चाहते हैं?",
        "Would you like to apply for this scheme?",
        "Kya aap is scheme ke liye apply karna chahenge?",
    )
    lines.append(cta)
    return "\n".join(lines)


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

    # Match by scheme name, but only when the text is specific enough to avoid
    # selecting a scheme from generic follow-up questions like "scheme details".
    user_tokens = _tokenize_scheme_reference(stripped_text)
    if not user_tokens:
        return None

    matching_scheme_id = None
    for scheme_info in presented:
        name_lower = scheme_info.get("name", "").lower()
        name_hindi = scheme_info.get("name_hindi", "").lower()
        scheme_tokens = _tokenize_scheme_reference(f"{name_lower} {name_hindi}")
        if not scheme_tokens:
            continue

        if text_lower in {name_lower.strip(), name_hindi.strip()}:
            return scheme_info["id"]

        if user_tokens.issubset(scheme_tokens):
            if matching_scheme_id and matching_scheme_id != scheme_info["id"]:
                return None
            matching_scheme_id = scheme_info["id"]

    return matching_scheme_id


def _store_presented_schemes(session: Session, schemes: list[SchemeMatch]) -> Session:
    """Store presented scheme info in session metadata for text-based selection."""
    presented = [
        {"id": m.scheme.id, "name": m.scheme.name, "name_hindi": m.scheme.name_hindi}
        for m in schemes[:5]
    ]
    return session_manager.set_presented_schemes(session, presented)


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
        self.llm = get_llm_client()

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

        # Handle /start command — always reset and greet
        if user_message.strip().lower() in ("/start", "/shuru"):
            session = session_manager.reset_session(session)
            lang = session.language_preference if session.language_preference != "auto" else "hi"
            response_text = response_generator.generate_greeting_response(lang)
            session = await session_manager.add_message(session, "user", user_message)
            session = await session_manager.add_message(session, "assistant", response_text)
            await session_manager.save_session(session)
            return ChatResponse(
                text=response_text,
                next_state=ConversationState.GREETING.value,
                language=lang,
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
            include_assistant=False,
        )

        analysis = await self.llm.analyze_message(
            user_message=user_message,
            conversation_history=conversation_history,
            current_state=session.state.value,
            user_profile={
                **session.user_profile.model_dump(),
                "_currently_asking": session.currently_asking,
            },
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
        rule_based_fields = profile_extractor.extract_by_patterns(user_message)
        extracted_fields = {**extracted_fields, **rule_based_fields}

        resolved_scheme_id = selected_scheme_id or _resolve_scheme_from_text(
            session, user_message
        )
        action = _detect_action_override(
            user_message,
            session.state,
            session.currently_asking,
            resolved_scheme_id,
        ) or llm_action

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
            session = await session_manager.add_message(session, "user", user_message)
            session = await session_manager.add_message(
                session, "assistant", response_text
            )
            await session_manager.save_session(session)
            return ChatResponse(
                text=response_text,
                next_state=ConversationState.GREETING.value,
                language=lang,
            )

        # Explicitly reset full session context on goodbye / restart.
        if action == "start_over":
            session = session_manager.reset_session(session, preserve_language=True)
            response_text = response_generator.generate_greeting_response(lang)
            session = await session_manager.add_message(session, "user", user_message)
            session = await session_manager.add_message(session, "assistant", response_text)
            await session_manager.save_session(session)

            return ChatResponse(
                text=response_text,
                next_state=ConversationState.GREETING.value,
                language=lang,
            )

        if intent == "goodbye" or action == "goodbye":
            # Send farewell, not a greeting — the user said goodbye.
            response_text = response_generator.generate_farewell_response(lang)
            session = session_manager.reset_session(session, preserve_language=True)

            session = await session_manager.add_message(session, "user", user_message)
            session = await session_manager.add_message(session, "assistant", response_text)
            await session_manager.save_session(session)

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
        if detected_life_event and detected_life_event != session.user_profile.life_event:
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
            elif matching_inputs_changed and session.selected_scheme_id:
                session = session_manager.clear_selection(session)
                session = session_manager.set_presented_schemes(session, [])

        # 5. Determine next FSM state
        next_state = fsm.determine_next_state(
            current_state=session.state,
            profile=profile,
            intent=intent,
            selected_scheme_id=resolved_scheme_id,
            has_selected_scheme=bool(session.selected_scheme_id),
            action=action,
        )

        # Prevent re-matching loop: if the previous turn got zero results
        # and the user hasn't provided new profile information, don't
        # auto-trigger MATCHING again — stay in UNDERSTANDING so the LLM
        # can respond naturally and guide the user.
        if (
            next_state == ConversationState.MATCHING
            and session.awaiting_profile_change
            and not profile_changed
        ):
            next_state = ConversationState.UNDERSTANDING

        if (
            matching_inputs_changed
            and profile.is_complete_for_matching
            and session.state in {
                ConversationState.PRESENTING,
                ConversationState.DETAILS,
                ConversationState.APPLICATION,
            }
        ):
            next_state = ConversationState.MATCHING

        # Affirmative in DETAILS → move to APPLICATION (not re-show details).
        # When the bot asks "Would you like to apply?" and the user says
        # "yes"/"sure"/"yeah", the FSM can't distinguish this from
        # a generic selection — so we override here.
        if (
            next_state == ConversationState.DETAILS
            and session.state == ConversationState.DETAILS
            and _is_affirmative(user_message)
        ):
            next_state = ConversationState.APPLICATION

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
                if session.state == ConversationState.HANDOFF:
                    session = session_manager.reset_session(session)
                response_text = response_generator.generate_greeting_response(lang)

            case ConversationState.UNDERSTANDING:
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

            case ConversationState.MATCHING:
                # Run scheme matching and build formatted response
                next_state, response_text, schemes, inline_keyboard, session = (
                    await self._run_matching(profile, user_message, session, lang)
                )
                # Leaving UNDERSTANDING flow — stop tracking field context
                session = session_manager.set_currently_asking(session, None)

            case ConversationState.PRESENTING:
                # Try text-based scheme selection first
                resolved_id = resolved_scheme_id
                if resolved_id:
                    # Transition to DETAILS — build details inline (no fallthrough)
                    session = session_manager.select_scheme(session, resolved_id)
                    next_state = ConversationState.DETAILS
                    response_text = await _build_scheme_details_text(
                        self.pool, resolved_id, profile, lang
                    )
                else:
                    # Re-present schemes with keyboard
                    schemes = await scheme_matcher.match_schemes(
                        pool=self.pool, profile=profile
                    )
                    if schemes:
                        session = _store_presented_schemes(session, schemes)
                        inline_keyboard = format_inline_keyboard(schemes, lang)
                    select_prompt = response_generator.generate_scheme_selection_response(lang)
                    response_text = select_prompt

            case ConversationState.DETAILS:
                scheme_id = resolved_scheme_id or session.selected_scheme_id
                if not scheme_id:
                    # No scheme selected — fall back to presenting
                    next_state = ConversationState.PRESENTING
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
                    response_text = await _build_scheme_details_text(
                        self.pool, scheme_id, profile, lang
                    )

            case ConversationState.APPLICATION:
                if resolved_scheme_id and resolved_scheme_id != session.selected_scheme_id:
                    session = session_manager.select_scheme(session, resolved_scheme_id)
                    next_state = ConversationState.DETAILS
                    response_text = await _build_scheme_details_text(
                        self.pool,
                        resolved_scheme_id,
                        profile,
                        lang,
                    )
                elif session.selected_scheme_id:
                    scheme_id = session.selected_scheme_id
                    scheme = await scheme_repo.get_scheme_by_id(self.pool, scheme_id)
                    if scheme:
                        response_text = response_generator.generate_application_guidance(
                            scheme.name_hindi if lang == "hi" else scheme.name,
                            scheme.application_url,
                            scheme.offline_process,
                            lang,
                        )
                    else:
                        response_text = (
                            "आवेदन जानकारी उपलब्ध नहीं है।"
                            if lang == "hi"
                            else "Application info not available."
                        )
                else:
                    response_text = _text_variant(
                        lang,
                        "कृपया पहले एक योजना चुनें।",
                        "Please select a scheme first.",
                        "Please pehle ek scheme select kijiye.",
                    )

            case ConversationState.HANDOFF:
                # Use LLM response if available (more natural), fall back to
                # structured office info when user explicitly requests it.
                if llm_response_text:
                    response_text = llm_response_text
                else:
                    response_text = await _build_handoff_text(self.pool, profile, lang)

            case _:
                response_text = response_generator.generate_greeting_response(lang)

        # 7. Update session state and save
        session = session_manager.update_state(session, next_state)
        session = await session_manager.add_message(session, "user", user_message)
        session = await session_manager.add_message(session, "assistant", response_text)
        await session_manager.save_session(session)

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
        schemes = await scheme_matcher.match_schemes(
            pool=self.pool,
            profile=profile,
            query_text=user_message,
        )

        if schemes:
            # Clear the no-match flag since we found results
            session = session_manager.set_awaiting_profile_change(session, False)
            session = _store_presented_schemes(session, schemes)
            response_text = _build_scheme_list_text(schemes, profile, lang)
            inline_keyboard = format_inline_keyboard(schemes, lang)
            return ConversationState.PRESENTING, response_text, schemes, inline_keyboard, session

        # No match — return to UNDERSTANDING so the user can explore
        # different areas or update their profile, instead of dumping them
        # into the HANDOFF dead-end.
        no_match_text = response_generator.generate_no_schemes_response(lang)
        session = session_manager.set_awaiting_profile_change(session, True)
        session = session_manager.clear_selection(session)
        session = session_manager.set_presented_schemes(session, [])
        return ConversationState.UNDERSTANDING, no_match_text, [], None, session

    async def _handle_callback(
        self,
        session: Session,
        callback_data: str,
    ) -> ChatResponse:
        """Handle callback query (scheme selection from inline keyboard)."""
        lang = session.language_preference if session.language_preference != "auto" else "hi"

        if callback_data.startswith("scheme:"):
            scheme_id = callback_data.replace("scheme:", "")

            # Select scheme and transition to DETAILS
            session = session_manager.select_scheme(session, scheme_id)
            session = session_manager.update_state(session, ConversationState.DETAILS)

            # Build rich details response
            response_text = await _build_scheme_details_text(
                self.pool, scheme_id, session.user_profile, lang
            )

            # Save session
            session = await session_manager.add_message(session, "assistant", response_text)
            await session_manager.save_session(session)

            return ChatResponse(
                text=response_text,
                next_state=ConversationState.DETAILS.value,
                language=lang,
            )

        return ChatResponse(
            text="अमान्य चयन।" if lang == "hi" else "Invalid selection.",
            language=lang,
        )
