"""Tests for Telegram Bot API client helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.integrations.telegram import DEFAULT_BOT_COMMANDS, TelegramClient


class _FakeResponse:
    """Minimal HTTPX-like response for Telegram client tests."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        """Pretend the HTTP response was successful."""

    def json(self):
        """Return the fake Telegram payload."""
        return self._payload


@pytest.mark.asyncio
async def test_set_my_commands_posts_default_commands() -> None:
    """Default Telegram commands should be posted to setMyCommands."""
    with patch(
        "src.integrations.telegram.get_settings",
        return_value=SimpleNamespace(telegram_bot_token="test-token"),
    ):
        client = TelegramClient()

    real_http_client = client._client
    mock_http_client = AsyncMock()
    mock_http_client.post = AsyncMock(return_value=_FakeResponse({"ok": True, "result": True}))
    client._client = mock_http_client
    await real_http_client.aclose()

    result = await client.set_my_commands()

    assert result["ok"] is True
    mock_http_client.post.assert_awaited_once_with(
        "https://api.telegram.org/bottest-token/setMyCommands",
        json={"commands": DEFAULT_BOT_COMMANDS},
    )


@pytest.mark.asyncio
async def test_get_my_commands_uses_telegram_endpoint() -> None:
    """Fetching registered Telegram commands should hit getMyCommands."""
    with patch(
        "src.integrations.telegram.get_settings",
        return_value=SimpleNamespace(telegram_bot_token="test-token"),
    ):
        client = TelegramClient()

    real_http_client = client._client
    mock_http_client = AsyncMock()
    mock_http_client.get = AsyncMock(
        return_value=_FakeResponse(
            {"ok": True, "result": [{"command": "help", "description": "Learn"}]}
        )
    )
    client._client = mock_http_client
    await real_http_client.aclose()

    result = await client.get_my_commands()

    assert result["result"][0]["command"] == "help"
    mock_http_client.get.assert_awaited_once_with(
        "https://api.telegram.org/bottest-token/getMyCommands"
    )
