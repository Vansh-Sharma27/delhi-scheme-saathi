"""Rich formatting utilities for Telegram messages.

Implements demo-quality message cards for:
- Scheme presentation with eligibility explainability
- Document procurement guides with prerequisites
- Rejection warnings with severity indicators
- Office/CSC location cards
"""

import re
from typing import Any

from src.models.document import Document, DocumentChain
from src.models.office import Office
from src.models.rejection_rule import RejectionRule
from src.models.scheme import Scheme, SchemeMatch
from src.models.session import UserProfile


def escape_markdown_v2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    if not text:
        return ""
    special_chars = r"_*[]()~`>#+-=|{}.!"
    for char in special_chars:
        text = text.replace(char, f"\\{char}")
    return text


def format_currency(amount: int | float | None, language: str = "hi") -> str:
    """Format amount as Indian currency with lakh/crore notation."""
    if amount is None:
        return "—"

    if amount >= 10000000:  # 1 crore
        crores = amount / 10000000
        if crores == int(crores):
            return f"₹{int(crores)} करोड़" if language == "hi" else f"₹{int(crores)} Cr"
        return f"₹{crores:.1f} करोड़" if language == "hi" else f"₹{crores:.1f} Cr"

    if amount >= 100000:  # 1 lakh
        lakhs = amount / 100000
        if lakhs == int(lakhs):
            return f"₹{int(lakhs)} लाख" if language == "hi" else f"₹{int(lakhs)} lakh"
        return f"₹{lakhs:.1f} लाख" if language == "hi" else f"₹{lakhs:.1f} lakh"

    if amount >= 1000:
        return f"₹{amount:,.0f}"

    return f"₹{amount:.0f}"


# =============================================================================
# Scheme Formatting
# =============================================================================

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


def format_scheme_card(
    match: SchemeMatch,
    profile: UserProfile | None = None,
    language: str = "hi",
    show_eligibility: bool = True,
) -> str:
    """Format scheme as a rich Telegram message card.

    Example output (Hindi):
    🏠 *प्रधानमंत्री आवास योजना - शहरी 2.0*
    💰 लाभ: ₹2.5 लाख तक
    📋 विभाग: MoHUA
    ✅ आप पात्र हैं: आयु ✓ आय ✓ श्रेणी ✓
    """
    scheme = match.scheme

    # Icon based on life event
    icon = "📋"
    for event in scheme.life_events:
        if event in LIFE_EVENT_ICONS:
            icon = LIFE_EVENT_ICONS[event]
            break

    # Name
    name = scheme.name_hindi if language == "hi" else scheme.name
    lines = [f"{icon} *{escape_markdown_v2(name)}*"]

    # Benefits line
    if scheme.benefits_amount:
        amount_str = format_currency(scheme.benefits_amount, language)
        freq = scheme.benefits_frequency or ""
        freq_map = {
            "monthly": "मासिक" if language == "hi" else "/month",
            "yearly": "वार्षिक" if language == "hi" else "/year",
            "one-time": "एकमुश्त" if language == "hi" else "one-time",
            "as-needed": "आवश्यकतानुसार" if language == "hi" else "as needed",
            "installments": "किश्तों में" if language == "hi" else "in installments",
        }
        freq_display = freq_map.get(freq, freq)
        benefit_label = "लाभ" if language == "hi" else "Benefit"
        lines.append(f"💰 {benefit_label}: {amount_str} {freq_display}")

    # Department
    dept = scheme.department_hindi if language == "hi" else scheme.department
    if len(dept) > 40:
        dept = dept[:37] + "..."
    dept_label = "विभाग" if language == "hi" else "Dept"
    lines.append(f"🏛️ {dept_label}: {escape_markdown_v2(dept)}")

    # Eligibility match with explainability
    if show_eligibility and match.eligibility_match:
        eligibility_line = _format_eligibility_match(match.eligibility_match, profile, language)
        if eligibility_line:
            lines.append(eligibility_line)

    # Similarity indicator (for debugging/demo)
    if match.similarity > 0.3:
        relevance = "उच्च" if language == "hi" else "High"
        lines.append(f"🎯 {relevance} match \\({match.similarity:.0%}\\)")

    return "\n".join(lines)


def _format_eligibility_match(
    eligibility_match: dict[str, bool],
    profile: UserProfile | None,
    language: str,
) -> str:
    """Format eligibility match with profile values."""
    parts = []

    field_labels = {
        "age": ("आयु", "Age"),
        "income": ("आय", "Income"),
        "category": ("श्रेणी", "Category"),
        "gender": ("लिंग", "Gender"),
    }

    for field, is_match in eligibility_match.items():
        label_hi, label_en = field_labels.get(field, (field, field))
        label = label_hi if language == "hi" else label_en
        icon = "✓" if is_match else "✗"

        # Add profile value if available
        value = ""
        if profile:
            if field == "age" and profile.age:
                value = f" {profile.age}"
            elif field == "income" and profile.annual_income:
                value = f" ₹{profile.annual_income // 1000}K"
            elif field == "category" and profile.category:
                value = f" {profile.category}"

        parts.append(f"{label}{value} {icon}")

    if not parts:
        return ""

    all_match = all(eligibility_match.values())
    prefix = "✅ पात्र" if language == "hi" else "✅ Eligible"
    if not all_match:
        prefix = "⚠️ जाँचें" if language == "hi" else "⚠️ Check"

    return f"{prefix}: {' • '.join(parts)}"


def format_scheme_list(
    matches: list[SchemeMatch],
    profile: UserProfile | None = None,
    language: str = "hi",
) -> str:
    """Format multiple schemes as a numbered list."""
    if not matches:
        no_schemes = "कोई योजना नहीं मिली" if language == "hi" else "No schemes found"
        return f"❌ {no_schemes}"

    header = "🎯 *आपके लिए योजनाएं:*" if language == "hi" else "🎯 *Schemes for you:*"
    lines = [header, ""]

    for i, match in enumerate(matches[:5], 1):
        card = format_scheme_card(match, profile, language, show_eligibility=True)
        lines.append(f"*{i}\\.*")
        lines.append(card)
        lines.append("")

    select_msg = "नंबर चुनें या 'विस्तार' बोलें" if language == "hi" else "Select number or say 'details'"
    lines.append(f"👆 {select_msg}")

    return "\n".join(lines)


def format_inline_keyboard(
    schemes: list[SchemeMatch],
    language: str = "hi",
) -> list[list[dict[str, str]]] | None:
    """Format schemes as Telegram inline keyboard buttons."""
    if not schemes:
        return None

    keyboard = []
    for i, match in enumerate(schemes[:5], 1):
        scheme = match.scheme
        name = scheme.name_hindi if language == "hi" else scheme.name

        # Truncate name for button (max 30 chars)
        if len(name) > 28:
            name = name[:25] + "..."

        keyboard.append([
            {
                "text": f"{i}. {name}",
                "callback_data": f"scheme:{scheme.id}"
            }
        ])

    return keyboard


# =============================================================================
# Document Formatting
# =============================================================================

def format_document_card(
    document: Document,
    language: str = "hi",
    show_prerequisites: bool = True,
) -> str:
    """Format document as a rich procurement guide card.

    Example output (Hindi):
    📄 *आय प्रमाण पत्र (Income Certificate)*
    🏛 कहाँ से: SDM Office / Tehsildar
    🌐 ऑनलाइन: edistrict.delhigovt.nic.in
    💲 शुल्क: ₹10 (BPL: मुफ्त)
    ⏱ समय: 7-15 दिन
    ⚠️ ध्यान रखें: नाम आधार से बिल्कुल मैच हो
    """
    name = document.name_hindi if language == "hi" else document.name
    lines = [f"📄 *{escape_markdown_v2(name)}*"]

    # Issuing authority
    if document.issuing_authority:
        where_label = "कहाँ से" if language == "hi" else "Where"
        lines.append(f"🏛 {where_label}: {escape_markdown_v2(document.issuing_authority)}")

    # Online portal
    if document.online_portal:
        online_label = "ऑनलाइन" if language == "hi" else "Online"
        # Extract domain from URL for brevity
        portal = document.online_portal.replace("https://", "").replace("http://", "")
        if len(portal) > 35:
            portal = portal[:32] + "..."
        lines.append(f"🌐 {online_label}: {escape_markdown_v2(portal)}")

    # Fee
    fee_parts = []
    if document.fee:
        fee_parts.append(f"₹{document.fee}" if document.fee.isdigit() else document.fee)
    if document.fee_bpl:
        bpl_label = "BPL" if language == "hi" else "BPL"
        bpl_fee = "मुफ्त" if document.fee_bpl.lower() == "free" else f"₹{document.fee_bpl}"
        fee_parts.append(f"\\({bpl_label}: {bpl_fee}\\)")
    if fee_parts:
        fee_label = "शुल्क" if language == "hi" else "Fee"
        lines.append(f"💲 {fee_label}: {' '.join(fee_parts)}")

    # Processing time
    if document.processing_time:
        time_label = "समय" if language == "hi" else "Time"
        lines.append(f"⏱ {time_label}: {document.processing_time}")

    # Common mistakes as warnings
    if document.common_mistakes:
        warning_label = "ध्यान रखें" if language == "hi" else "Note"
        mistake = document.common_mistakes[0][:60]
        lines.append(f"⚠️ {warning_label}: {escape_markdown_v2(mistake)}")

    return "\n".join(lines)


def format_document_chain(
    chain: DocumentChain,
    language: str = "hi",
) -> str:
    """Format document prerequisite chain as procurement order.

    Example output:
    📌 *पहले ये बनवाएं:*
      1. आधार कार्ड (मुफ्त, UIDAI)
      2. राशन कार्ड (₹5, Food & Supply)
    📄 *फिर बनवाएं:* आय प्रमाण पत्र
    """
    docs = chain.flat_list
    if not docs:
        return ""

    if len(docs) == 1:
        # Single document, no chain
        return format_document_card(docs[0], language)

    # Multiple documents - show chain
    main_doc = docs[-1]  # Last is the target
    prereqs = docs[:-1]

    lines = []

    # Prerequisites
    if prereqs:
        prereq_header = "📌 *पहले ये बनवाएं:*" if language == "hi" else "📌 *Get these first:*"
        lines.append(prereq_header)

        for i, doc in enumerate(prereqs, 1):
            name = doc.name_hindi if language == "hi" else doc.name
            fee_info = f"₹{doc.fee}" if doc.fee and doc.fee.isdigit() else (doc.fee or "मुफ्त")
            lines.append(f"  {i}\\. {escape_markdown_v2(name)} \\({fee_info}\\)")

        lines.append("")

    # Main document
    then_label = "📄 *फिर बनवाएं:*" if language == "hi" else "📄 *Then get:*"
    lines.append(then_label)
    lines.append(format_document_card(main_doc, language, show_prerequisites=False))

    return "\n".join(lines)


def format_document_list(
    documents: list[Document] | list[dict[str, Any]],
    language: str = "hi",
) -> str:
    """Format document list for Telegram message."""
    if not documents:
        return ""

    header = "📄 *आवश्यक दस्तावेज:*" if language == "hi" else "📄 *Required Documents:*"
    lines = [header, ""]

    for i, doc in enumerate(documents[:8], 1):
        if isinstance(doc, Document):
            name = doc.name_hindi if language == "hi" else doc.name
            where = doc.issuing_authority or ""
            fee = doc.fee or ""
        else:
            name = doc.get("name_hindi" if language == "hi" else "name", "Document")
            where = doc.get("issuing_authority", "")
            fee = doc.get("fee", "")

        lines.append(f"{i}\\. *{escape_markdown_v2(name)}*")
        if where:
            where_label = "कहाँ से" if language == "hi" else "Where"
            lines.append(f"   {where_label}: {escape_markdown_v2(where)}")
        if fee:
            fee_label = "शुल्क" if language == "hi" else "Fee"
            fee_display = f"₹{fee}" if fee.isdigit() else fee
            lines.append(f"   {fee_label}: {fee_display}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Rejection Warning Formatting
# =============================================================================

SEVERITY_ICONS = {
    "critical": "🔴",
    "high": "🟠",
    "warning": "🟡",
}


def format_rejection_warning(
    rule: RejectionRule,
    language: str = "hi",
) -> str:
    """Format a single rejection warning."""
    icon = SEVERITY_ICONS.get(rule.severity, "⚠️")
    desc = rule.description_hindi if language == "hi" else rule.description
    tip = rule.prevention_tip

    lines = [f"{icon} *{escape_markdown_v2(desc[:60])}*"]

    if tip:
        tip_label = "बचाव" if language == "hi" else "Prevention"
        lines.append(f"   ✅ {tip_label}: {escape_markdown_v2(tip[:80])}")

    return "\n".join(lines)


def format_rejection_warnings(
    warnings: list[RejectionRule] | list[dict[str, Any]],
    language: str = "hi",
) -> str:
    """Format rejection warnings for Telegram message.

    Example output:
    ⚠️ *अस्वीकृति चेतावनी*
    🔴 गंभीर: NPA Classification — EMI समय पर भरें
    🔴 गंभीर: 5-Year Lock-in — 5 साल तक बेचना मना
    ✅ बचाव: सभी EMI समय पर, 5 साल रहने की योजना बनाएं
    """
    if not warnings:
        return ""

    header = "⚠️ *अस्वीकृति से बचें:*" if language == "hi" else "⚠️ *Avoid Rejection:*"
    lines = [header, ""]

    # Sort by severity
    severity_order = {"critical": 0, "high": 1, "warning": 2}

    sorted_warnings = sorted(
        warnings,
        key=lambda w: severity_order.get(
            w.severity if isinstance(w, RejectionRule) else w.get("severity", "warning"),
            3
        )
    )

    for warning in sorted_warnings[:5]:
        if isinstance(warning, RejectionRule):
            lines.append(format_rejection_warning(warning, language))
        else:
            # Dict format fallback
            icon = SEVERITY_ICONS.get(warning.get("severity", "warning"), "⚠️")
            desc = warning.get("description_hindi" if language == "hi" else "description", "")[:60]
            tip = warning.get("prevention_tip", "")[:80]

            lines.append(f"{icon} {escape_markdown_v2(desc)}")
            if tip:
                tip_label = "बचाव" if language == "hi" else "Prevention"
                lines.append(f"   ✅ {tip_label}: {escape_markdown_v2(tip)}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Office Formatting
# =============================================================================

def format_office_card(
    office: Office,
    language: str = "hi",
) -> str:
    """Format office as a location card."""
    lines = [f"📍 *{escape_markdown_v2(office.name)}*"]

    if office.distance_km is not None:
        dist_label = "दूरी" if language == "hi" else "Distance"
        lines.append(f"🚶 {dist_label}: {office.distance_km:.1f} km")

    if office.address:
        addr = office.address[:60] if len(office.address) > 60 else office.address
        lines.append(f"📫 {escape_markdown_v2(addr)}")

    if office.phone:
        lines.append(f"📞 {office.phone}")

    if office.working_hours:
        hours_label = "समय" if language == "hi" else "Hours"
        lines.append(f"🕐 {hours_label}: {office.working_hours}")

    return "\n".join(lines)


def format_office_list(
    offices: list[Office] | list[dict[str, Any]],
    language: str = "hi",
) -> str:
    """Format office list for Telegram message."""
    if not offices:
        return ""

    header = "🏛️ *नजदीकी केंद्र:*" if language == "hi" else "🏛️ *Nearby Centers:*"
    lines = [header, ""]

    for office in offices[:3]:
        if isinstance(office, Office):
            lines.append(format_office_card(office, language))
        else:
            # Dict format fallback
            name = office.get("name", "Office")
            address = office.get("address", "")[:50]
            phone = office.get("phone", "")
            hours = office.get("working_hours", "")
            distance = office.get("distance_km")

            lines.append(f"📍 *{escape_markdown_v2(name)}*")
            if distance:
                lines.append(f"   🚶 {distance:.1f} km")
            if address:
                lines.append(f"   📫 {escape_markdown_v2(address)}")
            if phone:
                lines.append(f"   📞 {phone}")
            if hours:
                lines.append(f"   🕐 {hours}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Composite Message Formatting
# =============================================================================

def format_scheme_details(
    scheme: Scheme,
    documents: list[Document],
    rejection_rules: list[RejectionRule],
    offices: list[Office],
    profile: UserProfile | None = None,
    language: str = "hi",
) -> str:
    """Format complete scheme details page."""
    sections = []

    # Scheme header
    name = scheme.name_hindi if language == "hi" else scheme.name
    icon = LIFE_EVENT_ICONS.get(scheme.life_events[0] if scheme.life_events else "", "📋")
    sections.append(f"{icon} *{escape_markdown_v2(name)}*")
    sections.append("")

    # Description (truncated)
    desc = scheme.description_hindi if language == "hi" else scheme.description
    if len(desc) > 200:
        desc = desc[:197] + "..."
    sections.append(escape_markdown_v2(desc))
    sections.append("")

    # Benefits
    if scheme.benefits_amount:
        amount = format_currency(scheme.benefits_amount, language)
        benefit_label = "लाभ राशि" if language == "hi" else "Benefit Amount"
        sections.append(f"💰 *{benefit_label}:* {amount}")

    # Eligibility summary
    elig = scheme.eligibility
    elig_parts = []
    if elig.min_age or elig.max_age:
        age_range = f"{elig.min_age or 18}\\-{elig.max_age or 'कोई सीमा नहीं'}"
        elig_parts.append(f"आयु: {age_range}")
    if elig.max_income:
        income = format_currency(elig.max_income, language)
        elig_parts.append(f"अधिकतम आय: {income}")
    if elig.categories:
        elig_parts.append(f"श्रेणी: {', '.join(elig.categories)}")

    if elig_parts:
        elig_label = "पात्रता" if language == "hi" else "Eligibility"
        sections.append(f"✅ *{elig_label}:* {' • '.join(elig_parts)}")
    sections.append("")

    # Documents section
    if documents:
        sections.append(format_document_list(documents, language))

    # Rejection warnings section
    if rejection_rules:
        sections.append(format_rejection_warnings(rejection_rules, language))

    # Nearby offices section
    if offices:
        sections.append(format_office_list(offices, language))

    # Application link
    if scheme.application_url:
        apply_label = "आवेदन करें" if language == "hi" else "Apply"
        sections.append(f"🔗 [{apply_label}]({scheme.application_url})")

    return "\n".join(sections)


def format_greeting(language: str = "hi") -> str:
    """Format greeting message."""
    if language == "hi":
        return (
            "🙏 *नमस्ते\\! मैं दिल्ली स्कीम साथी हूँ\\.*\n\n"
            "मैं आपको सरकारी योजनाओं की जानकारी देने में मदद करता हूँ\\.\n\n"
            "आप मुझे बताएं, आज आपको किस तरह की सहायता चाहिए?\n"
            "\\(जैसे: घर, स्वास्थ्य, शिक्षा, रोजगार, पेंशन\\)"
        )
    return (
        "🙏 *Hello\\! I am Delhi Scheme Saathi\\.*\n\n"
        "I help you find government welfare schemes\\.\n\n"
        "Please tell me, what kind of assistance do you need today?\n"
        "\\(e\\.g\\.: housing, health, education, employment, pension\\)"
    )


def format_error(language: str = "hi") -> str:
    """Format error message."""
    if language == "hi":
        return (
            "❌ माफ करें, कुछ गड़बड़ हो गई\\.\n"
            "कृपया दोबारा कोशिश करें या /start से शुरू करें\\."
        )
    return (
        "❌ Sorry, something went wrong\\.\n"
        "Please try again or use /start to begin\\."
    )
