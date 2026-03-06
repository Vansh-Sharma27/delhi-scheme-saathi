"""Tests for local Docker seed bootstrap helpers."""

import pytest
from unittest.mock import AsyncMock, patch

from scripts.ensure_seed_data import auto_seed_enabled, ensure_seed_data


def test_auto_seed_enabled_truthy_values() -> None:
    """Recognize supported truthy AUTO_SEED_DATA values."""
    assert auto_seed_enabled("true")
    assert auto_seed_enabled("1")
    assert auto_seed_enabled("On")


def test_auto_seed_enabled_falsey_values() -> None:
    """Reject missing and falsey AUTO_SEED_DATA values."""
    assert not auto_seed_enabled(None)
    assert not auto_seed_enabled("")
    assert not auto_seed_enabled("false")


@pytest.mark.asyncio
async def test_ensure_seed_data_skips_when_disabled() -> None:
    """Do nothing unless AUTO_SEED_DATA is explicitly enabled."""
    with patch.dict("os.environ", {}, clear=False), patch(
        "scripts.ensure_seed_data.wait_for_database_ready",
        new=AsyncMock(),
    ) as wait_ready, patch(
        "scripts.ensure_seed_data.seed_database",
        new=AsyncMock(),
    ) as seed_database_mock:
        seeded = await ensure_seed_data()

    assert seeded is False
    wait_ready.assert_not_called()
    seed_database_mock.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_seed_data_loads_when_database_is_empty() -> None:
    """Seed bundled data when the local database has no schemes."""
    with patch.dict("os.environ", {"AUTO_SEED_DATA": "true"}, clear=False), patch(
        "scripts.ensure_seed_data.wait_for_database_ready",
        new=AsyncMock(),
    ) as wait_ready, patch(
        "scripts.ensure_seed_data.get_scheme_count",
        new=AsyncMock(return_value=0),
    ) as get_count, patch(
        "scripts.ensure_seed_data.seed_database",
        new=AsyncMock(),
    ) as seed_database_mock:
        seeded = await ensure_seed_data()

    assert seeded is True
    wait_ready.assert_awaited_once()
    get_count.assert_awaited_once()
    seed_database_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_seed_data_skips_when_database_is_populated() -> None:
    """Avoid reseeding when local data already exists."""
    with patch.dict("os.environ", {"AUTO_SEED_DATA": "true"}, clear=False), patch(
        "scripts.ensure_seed_data.wait_for_database_ready",
        new=AsyncMock(),
    ), patch(
        "scripts.ensure_seed_data.get_scheme_count",
        new=AsyncMock(return_value=5),
    ), patch(
        "scripts.ensure_seed_data.seed_database",
        new=AsyncMock(),
    ) as seed_database_mock:
        seeded = await ensure_seed_data()

    assert seeded is False
    seed_database_mock.assert_not_called()
