"""Plain-text views sent to the user.

Everything here renders to plain text, not MarkdownV2: the Telegram layer
strips markdown from outgoing messages, so emphasis markers would only show
up as literal asterisks. Emoji do the work that formatting would otherwise do.

Lengths are capped throughout because Telegram truncates hard at 4096
characters and a scheme description straight from the database can run well
past that on its own.
"""

import asyncpg

from src.db import office_repo, scheme_repo
from src.models.scheme import EligibilityCriteria, SchemeMatch
from src.models.session import Session, UserProfile
from src.services import document_resolver, rejection_engine, response_generator
from src.services.conversation.language import text_variant

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

SEVERITY_ICONS = {"critical": "🔴", "high": "🟠", "warning": "🟡"}

DEFAULT_SCHEME_ICON = "📋"

# How many entries each view shows before it stops.
MAX_LISTED_SCHEMES = 5
MAX_LISTED_DOCUMENTS = 5
MAX_LISTED_WARNINGS = 5
MAX_LISTED_OFFICES = 3

# Per-field caps, chosen so a full card stays inside one Telegram message.
MAX_DESCRIPTION_LEN = 380
MAX_DEPARTMENT_LEN = 40
MAX_AUTHORITY_LEN = 80
MAX_WARNING_LEN = 220
MAX_ADDRESS_LEN = 60

_LAKH = 100_000
_CRORE = 10_000_000

_SENTENCE_ENDINGS = (". ", "! ", "? ", ".\n", "!\n", "?\n", "। ", "।\n", "।")

# Truncation keeps a boundary only if it retains at least this share of the
# budget; otherwise cutting there would throw away most of the text.
_MIN_BOUNDARY_RATIO = 0.5


def format_currency_plain(amount: int | float | None, language: str = "hi") -> str:
    """Format an amount in Indian lakh/crore notation."""
    if amount is None:
        return ""
    if amount >= _CRORE:
        crores = amount / _CRORE
        label = "करोड़" if language == "hi" else "Cr"
        return f"₹{crores:.1f} {label}" if crores != int(crores) else f"₹{int(crores)} {label}"
    if amount >= _LAKH:
        lakhs = amount / _LAKH
        label = "लाख" if language == "hi" else "lakh"
        return f"₹{lakhs:.1f} {label}" if lakhs != int(lakhs) else f"₹{int(lakhs)} {label}"
    return f"₹{amount:,.0f}"


def truncate_at_word(text: str, max_len: int, ellipsis: str = "...") -> str:
    """Truncate at a word boundary so the last word is not cut in half."""
    if len(text) <= max_len:
        return text

    truncated = text[: max_len - len(ellipsis)]
    last_space = truncated.rfind(" ")
    if last_space > max_len * _MIN_BOUNDARY_RATIO:
        truncated = truncated[:last_space]
    return truncated.rstrip() + ellipsis


def truncate_at_sentence(text: str, max_len: int, ellipsis: str = "...") -> str:
    """Prefer sentence boundaries when shortening long text for chat output.

    Recognises the Devanagari danda alongside Latin sentence endings, and
    falls back to word truncation when no boundary is late enough to be worth
    using.
    """
    if len(text) <= max_len:
        return text

    sentence_end = max(
        (
            text.rfind(marker, 0, max_len) + 1
            for marker in _SENTENCE_ENDINGS
            if text.rfind(marker, 0, max_len) != -1
        ),
        default=-1,
    )
    if sentence_end >= max_len * _MIN_BOUNDARY_RATIO:
        return text[:sentence_end].rstrip()
    return truncate_at_word(text, max_len, ellipsis)


def _scheme_icon(life_events: list[str], preferred_life_event: str | None) -> str:
    """Pick the icon for a scheme, preferring the user's own life event.

    A scheme often covers several life events; showing the icon for the one
    the user actually asked about makes the list easier to scan.
    """
    ordered = list(life_events)
    if preferred_life_event and preferred_life_event in ordered:
        ordered = [preferred_life_event] + [
            event for event in ordered if event != preferred_life_event
        ]
    for event in ordered:
        if event in LIFE_EVENT_ICONS:
            return LIFE_EVENT_ICONS[event]
    return DEFAULT_SCHEME_ICON


def _scheme_not_found(language: str) -> str:
    """Message shown when a scheme id no longer resolves."""
    return text_variant(language, "योजना नहीं मिली।", "Scheme not found.", "Scheme nahi mili.")


def build_presented_scheme_selection_text(
    presented_schemes: list[dict[str, str]],
    language: str,
) -> str | None:
    """Render stored presented schemes without needing full match payloads."""
    if not presented_schemes:
        return None

    header = text_variant(
        language,
        "🎯 आपने ये योजना विकल्प देखे थे:",
        "🎯 You were viewing these scheme options:",
        "🎯 Aap ye scheme options dekh rahe the:",
    )
    footer = text_variant(
        language,
        "नीचे बटन दबाकर योजना चुनें।",
        "Tap a button below to open a scheme.",
        "Neeche button dabakar scheme kholiye.",
    )
    lines = [header, ""]
    for index, scheme in enumerate(presented_schemes[:MAX_LISTED_SCHEMES], 1):
        name = scheme.get("name_hindi") if language == "hi" else scheme.get("name")
        display_name = name or scheme.get("name") or scheme.get("name_hindi") or "Scheme"
        lines.append(f"{index}. {display_name}")
    lines.extend(["", footer])
    return "\n".join(lines)


def build_multi_beneficiary_scope_response(language: str) -> str:
    """Explain how to handle self-plus-child support questions during collection."""
    return text_variant(
        language,
        (
            "मैं आपकी और आपकी बेटी दोनों की मदद कर सकता हूँ, लेकिन सही योजना मिलाने के लिए "
            "एक समय में एक आवेदक पर ध्यान देना बेहतर रहेगा। अभी बताइए कि पहले योजनाएँ "
            "किसके लिए देखनी हैं, आपके लिए या आपकी बेटी के लिए?"
        ),
        (
            "I can help both you and your daughter, but it is more accurate to check "
            "schemes for one applicant at a time. Please tell me whose schemes you want "
            "to focus on first: yours or your daughter's?"
        ),
        (
            "Main aapki aur aapki beti dono ki madad kar sakta hoon, lekin sahi matching "
            "ke liye ek time par ek applicant par focus karna better rahega. Ab batayiye "
            "pehle schemes kiske liye dekhni hain, aapke liye ya aapki beti ke liye?"
        ),
    )


def build_select_scheme_first_text(language: str) -> str:
    """Prompt shown when a scheme view is requested with no scheme selected."""
    return text_variant(
        language,
        "कृपया पहले एक योजना चुनें।",
        "Please select a scheme first.",
        "Please pehle ek scheme select kijiye.",
    )


def build_scheme_list_text(
    schemes: list[SchemeMatch],
    profile: UserProfile,
    language: str,
) -> str:
    """Build a numbered, plain-text scheme list with eligibility info."""
    if not schemes:
        return text_variant(
            language,
            "कोई योजना नहीं मिली।",
            "No matching schemes found.",
            "Koi matching scheme nahi mili.",
        )

    header = text_variant(
        language,
        "🎯 आपके लिए ये योजनाएं मिली हैं:",
        "🎯 Found these schemes for you:",
        "🎯 Aapke liye ye schemes mili hain:",
    )
    lines = [header, ""]

    for index, match in enumerate(schemes[:MAX_LISTED_SCHEMES], 1):
        scheme = match.scheme
        icon = _scheme_icon(scheme.life_events, profile.life_event)
        name = scheme.name_hindi if language == "hi" else scheme.name
        lines.append(f"{index}. {icon} {name}")

        if scheme.benefits_amount:
            amount_str = format_currency_plain(scheme.benefits_amount, language)
            freq_map = {
                "monthly": text_variant(language, "मासिक", "/month", "per month"),
                "yearly": text_variant(language, "वार्षिक", "/year", "per year"),
                "one-time": text_variant(language, "एकमुश्त", "one-time", "one-time"),
                "installments": text_variant(
                    language, "किश्तों में", "in installments", "installments mein"
                ),
            }
            freq_display = freq_map.get(scheme.benefits_frequency or "", "")
            benefit_label = text_variant(language, "लाभ", "Benefit", "Benefit")
            lines.append(f"   💰 {benefit_label}: {amount_str} {freq_display}".rstrip())

        dept = scheme.department_hindi if language == "hi" else scheme.department
        if len(dept) > MAX_DEPARTMENT_LEN:
            dept = dept[: MAX_DEPARTMENT_LEN - 3] + "..."
        dept_label = text_variant(language, "विभाग", "Dept", "Dept")
        lines.append(f"   🏛️ {dept_label}: {dept}")

        if match.eligibility_match:
            lines.append(f"   {_eligibility_summary(match.eligibility_match, language)}")

        lines.append("")

    lines.append(
        text_variant(
            language,
            "👆 नीचे बटन दबाएं या नंबर बताएं।",
            "👆 Tap a button below or type the number.",
            "👆 Neeche button dabaiye ya number type kijiye.",
        )
    )
    return "\n".join(lines)


def _eligibility_summary(eligibility_match: dict[str, bool], language: str) -> str:
    """Render the per-field eligibility ticks for one scheme."""
    field_labels = {
        "age": ("आयु", "Age"),
        "income": ("आय", "Income"),
        "income_segment": ("आय वर्ग", "Income band"),
        "category": ("श्रेणी", "Category"),
        "gender": ("लिंग", "Gender"),
    }
    parts = []
    for field, is_match in eligibility_match.items():
        hi_label, en_label = field_labels.get(field, (field, field))
        label = hi_label if language == "hi" else en_label
        parts.append(f"{label} {'✓' if is_match else '✗'}")

    prefix = (
        text_variant(language, "✅ पात्र", "✅ Eligible", "✅ Eligible")
        if all(eligibility_match.values())
        else text_variant(language, "⚠️ जाँचें", "⚠️ Check", "⚠️ Check")
    )
    return f"{prefix}: {' • '.join(parts)}"


async def build_scheme_details_text(
    pool: asyncpg.Pool,
    scheme_id: str,
    profile: UserProfile,
    language: str,
) -> str:
    """Build a scheme overview and justification view."""
    scheme = await scheme_repo.get_scheme_by_id(pool, scheme_id)
    if not scheme:
        return _scheme_not_found(language)

    icon = _scheme_icon(scheme.life_events, profile.life_event)
    name = scheme.name_hindi if language == "hi" else scheme.name
    lines = [f"{icon} {name}", ""]

    desc = scheme.description_hindi if language == "hi" else scheme.description
    lines.append(truncate_at_sentence(desc, MAX_DESCRIPTION_LEN))
    lines.append("")

    if scheme.benefits_amount:
        amount_str = format_currency_plain(scheme.benefits_amount, language)
        benefit_label = text_variant(language, "लाभ राशि", "Benefit", "Benefit")
        lines.append(f"💰 {benefit_label}: {amount_str}")

    elig_parts = _eligibility_rule_parts(scheme.eligibility, language)
    if elig_parts:
        elig_label = text_variant(language, "पात्रता", "Eligibility", "Eligibility")
        lines.append(f"✅ {elig_label}: {' | '.join(elig_parts)}")

    match_details = scheme_repo.calculate_eligibility_match(scheme, profile)
    if match_details:
        lines.append("")
        lines.append(
            text_variant(
                language,
                "🎯 यह योजना क्यों दिखाई गई:",
                "🎯 Why this scheme was shown:",
                "🎯 Ye scheme kyon dikhayi gayi:",
            )
        )
        lines.extend(_match_reason_lines(match_details, profile, language))

    lines.append("")
    lines.append(
        text_variant(
            language,
            "अगला क्या देखें: दस्तावेज, अस्वीकृति चेतावनियाँ, या आवेदन प्रक्रिया?",
            "What would you like next: documents, rejection warnings, or application steps?",
            "Aage kya dekhna hai: documents, rejection warnings, ya application steps?",
        )
    )
    return "\n".join(lines)


def _eligibility_rule_parts(elig: EligibilityCriteria, language: str) -> list[str]:
    """Summarise a scheme's own eligibility rules, independent of the user."""
    parts = []
    if elig.min_age or elig.max_age:
        age_label = text_variant(language, "आयु", "Age", "Age")
        parts.append(f"{age_label}: {elig.min_age or 18}-{elig.max_age or '∞'}")
    if elig.max_income:
        income_label = text_variant(language, "अधिकतम आय", "Max income", "Max income")
        parts.append(f"{income_label}: {format_currency_plain(elig.max_income, language)}")
    if elig.caste_categories:
        cat_label = text_variant(language, "श्रेणी", "Category", "Category")
        parts.append(f"{cat_label}: {', '.join(elig.caste_categories)}")
    if elig.income_segments:
        band_label = text_variant(language, "आय वर्ग", "Income band", "Income band")
        parts.append(f"{band_label}: {', '.join(elig.income_segments)}")
    return parts


def _match_reason_lines(
    match_details: dict[str, bool],
    profile: UserProfile,
    language: str,
) -> list[str]:
    """Explain which of the user's own values satisfied the scheme's rules."""
    lines = []
    for field, is_match in match_details.items():
        if not is_match:
            continue
        if field == "age" and profile.age is not None:
            label = text_variant(language, "आयु मेल खाती है", "Age matches", "Age match karti hai")
            lines.append(f"• {label}: {profile.age}")
        elif field == "category" and profile.category:
            label = text_variant(
                language, "श्रेणी मेल खाती है", "Category matches", "Category match karti hai"
            )
            lines.append(f"• {label}: {profile.category}")
        elif field == "gender" and profile.gender:
            label = text_variant(
                language, "लिंग मेल खाता है", "Gender matches", "Gender match karta hai"
            )
            lines.append(f"• {label}: {profile.gender}")
        elif field == "income" and profile.annual_income is not None:
            label = text_variant(
                language, "आय सीमा के भीतर है", "Income is within range", "Income range ke andar hai"
            )
            lines.append(f"• {label}: {format_currency_plain(profile.annual_income, language)}")
        elif field == "income_segment" and profile.annual_income is not None:
            label = text_variant(
                language, "आय वर्ग उपयुक्त है", "Income band fits", "Income band fit hota hai"
            )
            lines.append(f"• {label}: {format_currency_plain(profile.annual_income, language)}")
    return lines


async def build_document_guidance_text(
    pool: asyncpg.Pool,
    session: Session,
    scheme_id: str,
    language: str,
) -> str:
    """Build focused document guidance for the selected scheme."""
    scheme = await scheme_repo.get_scheme_by_id(pool, scheme_id)
    if not scheme:
        return _scheme_not_found(language)

    documents = await document_resolver.resolve_documents_for_scheme(
        pool, scheme.documents_required
    )
    header = text_variant(
        language,
        f"📄 {scheme.name_hindi} के दस्तावेज:",
        f"📄 Documents for {scheme.name}:",
        f"📄 {scheme.name} ke documents:",
    )
    lines = [header, ""]

    if not documents:
        lines.append(
            text_variant(
                language,
                "दस्तावेज जानकारी उपलब्ध नहीं है।",
                "Document guidance is not available yet.",
                "Document guidance abhi available nahi hai.",
            )
        )
        return "\n".join(lines)

    for index, chain in enumerate(documents[:MAX_LISTED_DOCUMENTS], 1):
        doc = chain.document
        doc_name = doc.name_hindi if language == "hi" else doc.name
        lines.append(f"{index}. {doc_name}")
        authority = truncate_at_word(doc.issuing_authority, MAX_AUTHORITY_LEN)
        where_label = text_variant(language, "कहाँ से", "Where from", "Kahan se")
        lines.append(f"   🏛️ {where_label}: {authority}")

        details = []
        if doc.fee:
            fee_value = f"₹{doc.fee}" if doc.fee.isdigit() else doc.fee
            details.append(f"{text_variant(language, 'शुल्क', 'Fee', 'Fee')}: {fee_value}")
        if doc.processing_time:
            details.append(
                f"{text_variant(language, 'समय', 'Time', 'Time')}: {doc.processing_time}"
            )
        if details:
            lines.append(f"   📋 {' | '.join(details)}")
        if doc.online_portal:
            online_label = text_variant(language, "ऑनलाइन", "Online", "Online")
            lines.append(f"   🌐 {online_label}: {doc.online_portal}")
        lines.append("")

    lines.append(
        text_variant(
            language,
            "अगर चाहें तो मैं सामान्य अस्वीकृति चेतावनियाँ भी बता सकता हूँ।",
            "If you want, I can also show the common rejection warnings.",
            "Agar chahein to main common rejection warnings bhi bata sakta hoon.",
        )
    )
    return await response_generator.translate_grounded_text_if_needed(
        session,
        "\n".join(lines),
        language,
    )


async def build_rejection_warnings_text(
    pool: asyncpg.Pool,
    scheme_id: str,
    profile: UserProfile,
    language: str,
) -> str:
    """Build focused rejection-prevention guidance for the selected scheme."""
    scheme = await scheme_repo.get_scheme_by_id(pool, scheme_id)
    if not scheme:
        return _scheme_not_found(language)

    warnings = await rejection_engine.get_rejection_warnings(pool, scheme_id, profile)
    header = text_variant(
        language,
        f"⚠️ {scheme.name_hindi} की अस्वीकृति चेतावनियाँ:",
        f"⚠️ Rejection warnings for {scheme.name}:",
        f"⚠️ {scheme.name} ki rejection warnings:",
    )
    lines = [header, ""]

    if not warnings:
        lines.append(
            text_variant(
                language,
                "फिलहाल अस्वीकृति चेतावनियाँ उपलब्ध नहीं हैं।",
                "No rejection warnings are available right now.",
                "Abhi rejection warnings available nahi hain.",
            )
        )
        return "\n".join(lines)

    for rule in sorted(warnings[:MAX_LISTED_WARNINGS], key=lambda rule: rule.severity_order):
        icon = SEVERITY_ICONS.get(rule.severity, "⚠️")
        if language == "hi":
            tip = rule.description_hindi or rule.description
        else:
            tip = rule.prevention_tip or rule.description
        lines.append(f"{icon} {truncate_at_sentence(tip, MAX_WARNING_LEN)}")

    lines.append("")
    lines.append(
        text_variant(
            language,
            "अगर चाहें तो मैं आवेदन प्रक्रिया भी बता सकता हूँ।",
            "If you want, I can also show the application process.",
            "Agar chahein to main application process bhi bata sakta hoon.",
        )
    )
    return "\n".join(lines)


async def build_application_help_text(
    pool: asyncpg.Pool,
    session: Session,
    scheme_id: str,
    language: str,
) -> str:
    """Build focused application guidance for the selected scheme."""
    scheme = await scheme_repo.get_scheme_by_id(pool, scheme_id)
    if not scheme:
        return _scheme_not_found(language)

    draft = response_generator.generate_application_guidance(
        scheme.name_hindi if language == "hi" else scheme.name,
        scheme.application_url,
        scheme.offline_process,
        application_steps=scheme.application_steps,
        processing_time=scheme.processing_time,
        helpline_phone=scheme.helpline.phone if scheme.helpline else None,
        language=language,
    )
    return await response_generator.translate_grounded_text_if_needed(
        session,
        draft,
        language,
    )


async def build_scheme_question_answer_text(
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
        return _scheme_not_found(language)
    return await response_generator.generate_scheme_question_response(
        session,
        scheme,
        profile,
        user_question,
        language,
        active_view=active_view or session.state.value,
    )


async def build_handoff_text(
    pool: asyncpg.Pool,
    profile: UserProfile,
    language: str,
) -> str:
    """Build handoff text with nearby office info."""
    offices = []
    if profile.latitude and profile.longitude:
        offices = await office_repo.get_nearest_offices(
            pool, profile.latitude, profile.longitude, MAX_LISTED_OFFICES, "CSC"
        )
    elif profile.district:
        offices = await office_repo.get_offices_by_district(
            pool, profile.district, MAX_LISTED_OFFICES
        )

    lines = [
        text_variant(
            language,
            "🏛️ आपकी और सहायता के लिए नजदीकी सेवा केंद्र:",
            "🏛️ Nearest service centers for further help:",
            "🏛️ Aur madad ke liye nearest service centers:",
        ),
        "",
    ]

    if offices:
        for office in offices[:MAX_LISTED_OFFICES]:
            lines.append(f"📍 {office.name}")
            if office.address:
                lines.append(f"   📫 {office.address[:MAX_ADDRESS_LEN]}")
            if office.phone:
                lines.append(f"   📞 {office.phone}")
            if office.working_hours:
                hours_label = text_variant(language, "समय", "Hours", "Hours")
                lines.append(f"   🕐 {hours_label}: {office.working_hours}")
            lines.append("")
    else:
        lines.append(
            text_variant(
                language,
                "नजदीकी केंद्र की जानकारी उपलब्ध नहीं है।",
                "No nearby center information available.",
                "Nearby center ki information available nahi hai.",
            )
        )
        lines.append("")

    lines.append(
        text_variant(
            language,
            "कृपया अपने सभी दस्तावेज लेकर जाएं।",
            "Please carry all your documents.",
            "Please apne saare documents saath lekar jaiye.",
        )
    )
    return "\n".join(lines)
