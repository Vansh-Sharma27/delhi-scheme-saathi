"""Tests for application startup behavior."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from src import main as main_module
from src.db.scheme_repo import get_scheme_debug_rows
from src.services.ai_background import InMemoryAIWorkQueue


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    async def fetch(self, query, scheme_ids):  # type: ignore[no-untyped-def]
        return self._rows


class _AcquireContext:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    def __init__(self, rows):
        self._rows = rows

    def acquire(self):
        return _AcquireContext(_FakeConn(self._rows))


@pytest.mark.asyncio
async def test_get_scheme_debug_rows_handles_partial_rows() -> None:
    """Debug-row verification should work on lightweight SELECT payloads."""
    pool = _FakePool(
        [
            {
                "id": "SCH-DELHI-001",
                "name": "PMAY-U 2.0",
                "life_events": ["HOUSING"],
                "eligibility": {
                    "categories": ["EWS", "LIG", "MIG"],
                    "income_by_category": {
                        "EWS": 300000,
                        "LIG": 600000,
                        "MIG": 900000,
                    },
                },
            }
        ]
    )

    rows = await get_scheme_debug_rows(pool, ["SCH-DELHI-001"])

    assert rows == [
        {
            "id": "SCH-DELHI-001",
            "name": "PMAY-U 2.0",
            "life_events": ["HOUSING"],
            "canonical_life_events": ["CHILDBIRTH", "HOUSING", "MARRIAGE"],
            "life_events_match": False,
            "raw_categories": ["EWS", "LIG", "MIG"],
            "caste_categories": [],
            "income_segments": ["EWS", "LIG", "MIG"],
            "income_by_category": {
                "EWS": 300000,
                "LIG": 600000,
                "MIG": 900000,
            },
        }
    ]


@pytest.mark.asyncio
async def test_get_scheme_debug_rows_handles_stringified_eligibility() -> None:
    """Debug-row verification should accept JSON-string eligibility payloads."""
    pool = _FakePool(
        [
            {
                "id": "SCH-DELHI-006",
                "name": "Education Loan Scheme - Delhi",
                "life_events": ["EDUCATION"],
                "eligibility": json.dumps(
                    {
                        "categories": ["SC", "ST", "OBC"],
                        "max_income": 800000,
                    }
                ),
            }
        ]
    )

    rows = await get_scheme_debug_rows(pool, ["SCH-DELHI-006"])

    assert rows == [
        {
            "id": "SCH-DELHI-006",
            "name": "Education Loan Scheme - Delhi",
            "life_events": ["EDUCATION"],
            "canonical_life_events": ["EDUCATION"],
            "life_events_match": True,
            "raw_categories": ["SC", "ST", "OBC"],
            "caste_categories": ["SC", "ST", "OBC"],
            "income_segments": [],
            "income_by_category": {},
        }
    ]


@pytest.mark.asyncio
async def test_lifespan_keeps_db_pool_when_verification_logging_fails() -> None:
    """Startup verification failures should not mark the database as disconnected."""
    fake_pool = AsyncMock()
    fake_pool.close = AsyncMock()
    configure_ai = AsyncMock()
    shutdown_ai = AsyncMock()

    with patch.object(main_module, "init_db_pool", AsyncMock(return_value=fake_pool)), patch(
        "src.db.scheme_repo.get_scheme_debug_rows",
        AsyncMock(side_effect=KeyError("name_hindi")),
    ), patch.object(main_module, "_configure_session_store", lambda: None), patch.object(
        main_module,
        "_configure_ai_background_runtime",
        configure_ai,
    ), patch.object(
        main_module,
        "_shutdown_ai_background_runtime",
        shutdown_ai,
    ):
        main_module.db_pool = None
        async with main_module.lifespan(main_module.app):
            assert main_module.db_pool is fake_pool

        assert main_module.db_pool is None
        fake_pool.close.assert_awaited_once()
        configure_ai.assert_awaited_once()
        shutdown_ai.assert_awaited_once()


@pytest.mark.asyncio
async def test_configure_ai_background_runtime_starts_in_memory_worker() -> None:
    """Local in-memory queue should start the in-process worker."""
    start_worker = AsyncMock()

    with patch(
        "src.services.ai_background.create_default_ai_work_queue",
        return_value=InMemoryAIWorkQueue(),
    ), patch(
        "src.services.ai_background.configure_ai_work_queue",
    ) as configure_queue, patch(
        "src.services.ai_background.start_ai_background_worker",
        start_worker,
    ):
        await main_module._configure_ai_background_runtime()

    configure_queue.assert_called_once()
    start_worker.assert_awaited_once()


@pytest.mark.asyncio
async def test_configure_ai_background_runtime_skips_worker_for_external_queue() -> None:
    """Shared queues like SQS should not start an in-process poller in the web app."""
    start_worker = AsyncMock()
    external_queue = object()

    with patch(
        "src.services.ai_background.create_default_ai_work_queue",
        return_value=external_queue,
    ), patch(
        "src.services.ai_background.configure_ai_work_queue",
    ) as configure_queue, patch(
        "src.services.ai_background.start_ai_background_worker",
        start_worker,
    ):
        await main_module._configure_ai_background_runtime()

    configure_queue.assert_called_once_with(external_queue)
    start_worker.assert_not_awaited()
