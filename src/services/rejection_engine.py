"""Rejection rule engine for proactive warnings."""

import logging
from typing import Any

import asyncpg

from src.db.rejection_rule_repo import get_rules_by_scheme, get_rules_by_ids
from src.models.rejection_rule import RejectionRule
from src.models.session import UserProfile

logger = logging.getLogger(__name__)


async def get_rejection_warnings(
    pool: asyncpg.Pool,
    scheme_id: str,
    profile: UserProfile | None = None,
) -> list[RejectionRule]:
    """Get rejection warnings for a scheme.

    Returns rules sorted by severity (critical first).
    Optionally filters by profile-relevant rules.
    """
    rules = await get_rules_by_scheme(pool, scheme_id)

    # For now, return all rules
    # TODO: Filter based on profile (e.g., income-related rules for low-income users)

    return rules


async def get_rules_for_scheme_ids(
    pool: asyncpg.Pool,
    rule_ids: list[str],
) -> list[RejectionRule]:
    """Get specific rejection rules by IDs."""
    return await get_rules_by_ids(pool, rule_ids)


def categorize_rules(rules: list[RejectionRule]) -> dict[str, list[RejectionRule]]:
    """Categorize rules by severity."""
    categorized = {
        "critical": [],
        "high": [],
        "warning": [],
    }

    for rule in rules:
        if rule.severity in categorized:
            categorized[rule.severity].append(rule)

    return categorized


def format_rejection_warning(
    rule: RejectionRule,
    language: str = "hi",
) -> dict[str, Any]:
    """Format rejection rule for display."""
    severity_icons = {
        "critical": "🔴",
        "high": "🟠",
        "warning": "🟡",
    }

    severity_labels = {
        "critical": {"hi": "गंभीर", "en": "Critical"},
        "high": {"hi": "महत्वपूर्ण", "en": "High"},
        "warning": {"hi": "चेतावनी", "en": "Warning"},
    }

    return {
        "id": rule.id,
        "icon": severity_icons.get(rule.severity, "⚠️"),
        "severity": rule.severity,
        "severity_label": severity_labels.get(rule.severity, {}).get(language, rule.severity),
        "description": rule.description_hindi if language == "hi" else rule.description,
        "prevention_tip": rule.prevention_tip,
        "rule_type": rule.rule_type,
        "examples": rule.examples,
    }


def generate_rejection_warning_card(
    rules: list[RejectionRule],
    language: str = "hi",
) -> str:
    """Generate formatted rejection warnings card for Telegram."""
    if not rules:
        return ""

    categorized = categorize_rules(rules)

    lines = []

    if language == "hi":
        lines.append("⚠️ *अस्वीकृति से बचें*\n")
    else:
        lines.append("⚠️ *Avoid Rejection*\n")

    # Critical rules first
    for rule in categorized["critical"]:
        desc = rule.description_hindi if language == "hi" else rule.description
        lines.append(f"🔴 *{desc[:80]}*")
        lines.append(f"   ✅ {rule.prevention_tip[:100]}")
        lines.append("")

    # High severity
    for rule in categorized["high"]:
        desc = rule.description_hindi if language == "hi" else rule.description
        lines.append(f"🟠 {desc[:80]}")
        lines.append(f"   ✅ {rule.prevention_tip[:100]}")
        lines.append("")

    # Warnings (just list them)
    if categorized["warning"]:
        if language == "hi":
            lines.append("📋 अन्य सावधानियाँ:")
        else:
            lines.append("📋 Other precautions:")
        for rule in categorized["warning"][:3]:  # Limit to 3
            desc = rule.description_hindi if language == "hi" else rule.description
            lines.append(f"  • {desc[:60]}")

    return "\n".join(lines)


def get_top_warnings(rules: list[RejectionRule], limit: int = 3) -> list[RejectionRule]:
    """Get top N most important warnings."""
    # Sort by severity order (critical=0, high=1, warning=2)
    sorted_rules = sorted(rules, key=lambda r: r.severity_order)
    return sorted_rules[:limit]
