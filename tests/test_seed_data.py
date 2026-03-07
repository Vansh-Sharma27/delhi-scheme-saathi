"""Tests for seed-data helpers."""

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
