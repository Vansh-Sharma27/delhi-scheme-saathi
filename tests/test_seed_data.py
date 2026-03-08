"""Tests for seed-data helpers."""

import json
from pathlib import Path

import pytest

from scripts.seed_data import DEFAULT_DATABASE_URL, resolve_database_url


def test_resolve_database_url_uses_default_when_missing() -> None:
    """Missing DATABASE_URL should fall back to the local default."""
    assert resolve_database_url(None) == DEFAULT_DATABASE_URL


def test_resolve_database_url_strips_quotes() -> None:
    """Quoted env values should normalize before asyncpg sees them."""
    assert (
        resolve_database_url('"postgresql://user:pass@db.example.com:5432/app"')
        == "postgresql://user:pass@db.example.com:5432/app"
    )


def test_resolve_database_url_rejects_invalid_scheme() -> None:
    """Malformed env values should fail with a clear validation error."""
    with pytest.raises(RuntimeError, match="postgres"):
        resolve_database_url("mysql://user:pass@localhost/db")


def test_all_active_seed_schemes_have_followup_data() -> None:
    """Every active seed scheme should have docs, rejection rules, and application steps."""
    all_schemes_path = Path(__file__).resolve().parents[1] / "data" / "all_schemes.json"
    schemes = json.loads(all_schemes_path.read_text())

    missing_fields: list[str] = []
    for scheme in schemes:
        if not scheme.get("is_active", True):
            continue

        docs = scheme.get("documents_required", []) or scheme.get("documents_required_ids", [])
        rejections = scheme.get("rejection_rules", []) or scheme.get("rejection_rule_ids", [])
        application_steps = scheme.get("application_steps", [])

        if not docs:
            missing_fields.append(f"{scheme['id']} missing documents_required")
        if not rejections:
            missing_fields.append(f"{scheme['id']} missing rejection_rules")
        if not application_steps:
            missing_fields.append(f"{scheme['id']} missing application_steps")

    assert missing_fields == []
