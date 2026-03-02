"""Formatting utilities for Telegram messages."""

import re
from typing import Any

from src.models.scheme import SchemeMatch


def escape_markdown_v2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    special_chars = r"_*[]()~`>#+-=|{}.!"
    for char in special_chars:
        text = text.replace(char, f"\\{char}")
    return text


def format_currency(amount: int | float | None, language: str = "hi") -> str:
    """Format amount as Indian currency."""
    if amount is None:
        return ""

    if amount >= 100000:
        lakhs = amount / 100000
        if lakhs == int(lakhs):
            return f"₹{int(lakhs)} लाख" if language == "hi" else f"₹{int(lakhs)} lakh"
        return f"₹{lakhs:.1f} लाख" if language == "hi" else f"₹{lakhs:.1f} lakh"

    return f"₹{amount:,.0f}"


def format_scheme_card(match: SchemeMatch, language: str = "hi") -> str:
    """Format scheme as a Telegram message card."""
    scheme = match.scheme

    # Name with icon based on life events
    icons = {
        "HOUSING": "🏠",
        "HEALTH_CRISIS": "🏥",
        "EDUCATION": "📚",
        "DEATH_IN_FAMILY": "🙏",
        "BUSINESS_STARTUP": "💼",
        "WOMEN_EMPOWERMENT": "👩",
        "JOB_LOSS": "💼",
    }
    icon = "📋"
    for event in scheme.life_events:
        if event in icons:
            icon = icons[event]
            break

    name = scheme.name_hindi if language == "hi" else scheme.name
    lines = [f"{icon} *{name}*"]

    # Benefits
    if scheme.benefits_amount:
        amount_str = format_currency(scheme.benefits_amount, language)
        benefit_label = "लाभ" if language == "hi" else "Benefit"
        lines.append(f"💰 {benefit_label}: {amount_str}")

    # Department
    dept = scheme.department_hindi if language == "hi" else scheme.department
    dept_label = "विभाग" if language == "hi" else "Dept"
    lines.append(f"🏛️ {dept_label}: {dept}")

    # Eligibility match
    if match.eligibility_match:
        eligibility_parts = []
        for field, is_match in match.eligibility_match.items():
            icon = "✓" if is_match else "✗"
            field_name = {
                "age": "आयु" if language == "hi" else "Age",
                "income": "आय" if language == "hi" else "Income",
                "category": "श्रेणी" if language == "hi" else "Category",
                "gender": "लिंग" if language == "hi" else "Gender",
            }.get(field, field)
            eligibility_parts.append(f"{field_name} {icon}")

        if eligibility_parts:
            match_label = "पात्रता" if language == "hi" else "Eligibility"
            lines.append(f"✅ {match_label}: {', '.join(eligibility_parts)}")

    return "\n".join(lines)


def format_inline_keyboard(
    schemes: list[SchemeMatch],
    language: str = "hi",
) -> list[list[dict[str, str]]] | None:
    """Format schemes as Telegram inline keyboard buttons."""
    if not schemes:
        return None

    keyboard = []
    for match in schemes[:5]:  # Max 5 schemes
        scheme = match.scheme
        name = scheme.name_hindi if language == "hi" else scheme.name

        # Truncate name for button
        if len(name) > 30:
            name = name[:27] + "..."

        keyboard.append([
            {
                "text": name,
                "callback_data": f"scheme:{scheme.id}"
            }
        ])

    return keyboard


def format_document_list(
    documents: list[dict[str, Any]],
    language: str = "hi",
) -> str:
    """Format document list for Telegram message."""
    if not documents:
        return ""

    header = "📄 *आवश्यक दस्तावेज:*" if language == "hi" else "📄 *Required Documents:*"
    lines = [header, ""]

    for i, doc in enumerate(documents, 1):
        name = doc.get("name", "Document")
        where = doc.get("issuing_authority", "")
        fee = doc.get("fee", "")

        lines.append(f"{i}\\. *{escape_markdown_v2(name)}*")
        if where:
            where_label = "कहाँ से" if language == "hi" else "Where"
            lines.append(f"   {where_label}: {escape_markdown_v2(where)}")
        if fee:
            fee_label = "शुल्क" if language == "hi" else "Fee"
            lines.append(f"   {fee_label}: {fee}")
        lines.append("")

    return "\n".join(lines)


def format_rejection_warnings(
    warnings: list[dict[str, Any]],
    language: str = "hi",
) -> str:
    """Format rejection warnings for Telegram message."""
    if not warnings:
        return ""

    header = "⚠️ *अस्वीकृति से बचें:*" if language == "hi" else "⚠️ *Avoid Rejection:*"
    lines = [header, ""]

    for warning in warnings[:5]:
        icon = warning.get("icon", "⚠️")
        desc = warning.get("description", "")[:80]
        tip = warning.get("prevention_tip", "")[:100]

        lines.append(f"{icon} {escape_markdown_v2(desc)}")
        if tip:
            tip_label = "बचाव" if language == "hi" else "Prevention"
            lines.append(f"   ✅ {tip_label}: {escape_markdown_v2(tip)}")
        lines.append("")

    return "\n".join(lines)


def format_office_list(
    offices: list[dict[str, Any]],
    language: str = "hi",
) -> str:
    """Format office list for Telegram message."""
    if not offices:
        return ""

    header = "🏛️ *नजदीकी केंद्र:*" if language == "hi" else "🏛️ *Nearby Centers:*"
    lines = [header, ""]

    for office in offices[:3]:
        name = office.get("name", "Office")
        address = office.get("address", "")[:50]
        phone = office.get("phone", "")
        hours = office.get("working_hours", "")
        distance = office.get("distance_km")

        lines.append(f"📍 *{escape_markdown_v2(name)}*")
        if distance:
            lines.append(f"   {distance:.1f} km")
        if address:
            lines.append(f"   {escape_markdown_v2(address)}")
        if phone:
            lines.append(f"   📞 {phone}")
        if hours:
            lines.append(f"   🕐 {hours}")
        lines.append("")

    return "\n".join(lines)
