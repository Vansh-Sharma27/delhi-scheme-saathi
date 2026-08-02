"""Tests for application startup behavior."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from src import main as main_module
from src.db.scheme_repo import get_scheme_debug_rows
from src.services.ai_background import InMemoryAIWorkQueue


class _FakeRequest:
    """Minimal stand-in exposing only what the chat endpoint reads."""

    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = headers or {}


def _capturing_service(captured: dict[str, str]):
    """Conversation service double that records the session ID it is given."""

    class _Service:
        def __init__(self, pool):  # type: ignore[no-untyped-def]
            pass

        async def handle_message(self, request):  # type: ignore[no-untyped-def]
            captured["user_id"] = request.user_id
            return SimpleNamespace(
                text="ok",
                next_state="GREETING",
                schemes=None,
                documents=None,
                rejection_warnings=None,
            )

    return _Service


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


@pytest.mark.asyncio
async def test_chat_endpoint_namespaces_caller_supplied_user_id() -> None:
    """A Telegram user's numeric ID must not address their real session."""
    captured: dict[str, str] = {}
    telegram_user_id = "780045592"

    with patch.object(main_module, "get_db_pool", lambda: object()), patch.object(
        main_module, "get_settings", lambda: SimpleNamespace(chat_api_key="")
    ), patch(
        "src.services.conversation.ConversationService", _capturing_service(captured)
    ):
        await main_module.chat_endpoint(
            {"user_id": telegram_user_id, "message": "Namaste"}, _FakeRequest()
        )

    assert captured["user_id"] != telegram_user_id
    assert captured["user_id"] == f"{main_module.CHAT_SESSION_PREFIX}{telegram_user_id}"


@pytest.mark.asyncio
async def test_chat_endpoint_rejects_wrong_api_key() -> None:
    """With CHAT_API_KEY set, a bad or missing header must not reach the service."""
    captured: dict[str, str] = {}

    with patch.object(main_module, "get_db_pool", lambda: object()), patch.object(
        main_module, "get_settings", lambda: SimpleNamespace(chat_api_key="expected-key")
    ), patch(
        "src.services.conversation.ConversationService", _capturing_service(captured)
    ):
        for headers in ({}, {"X-API-Key": "wrong-key"}):
            with pytest.raises(HTTPException) as excinfo:
                await main_module.chat_endpoint(
                    {"user_id": "tester", "message": "Namaste"}, _FakeRequest(headers)
                )
            assert excinfo.value.status_code == 403

    assert captured == {}


@pytest.mark.asyncio
async def test_chat_endpoint_accepts_correct_api_key() -> None:
    """The matching header still gets through to the conversation service."""
    captured: dict[str, str] = {}

    with patch.object(main_module, "get_db_pool", lambda: object()), patch.object(
        main_module, "get_settings", lambda: SimpleNamespace(chat_api_key="expected-key")
    ), patch(
        "src.services.conversation.ConversationService", _capturing_service(captured)
    ):
        result = await main_module.chat_endpoint(
            {"user_id": "tester", "message": "Namaste"},
            _FakeRequest({"X-API-Key": "expected-key"}),
        )

    assert result["response"] == "ok"
    assert captured["user_id"] == f"{main_module.CHAT_SESSION_PREFIX}tester"


def test_chat_route_binds_body_and_namespaces_over_http() -> None:
    """Cover the routed path, not just a direct call.

    The endpoint takes both a JSON body and the Request object; a signature
    change can keep direct calls working while breaking FastAPI's body binding.
    """
    from fastapi.testclient import TestClient

    captured: dict[str, str] = {}

    with patch.object(main_module, "get_db_pool", lambda: object()), patch.object(
        main_module, "get_settings", lambda: SimpleNamespace(chat_api_key="")
    ), patch(
        "src.services.conversation.ConversationService", _capturing_service(captured)
    ):
        client = TestClient(main_module.app)
        response = client.post(
            "/api/chat", json={"user_id": "780045592", "message": "Namaste"}
        )

    assert response.status_code == 200
    assert response.json()["response"] == "ok"
    assert captured["user_id"] == "api:780045592"
