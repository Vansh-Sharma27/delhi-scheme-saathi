"""Telegram Bot API client."""

import logging
from typing import Any

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_BOT_COMMANDS = [
    {"command": "start", "description": "Start a new scheme search"},
    {"command": "help", "description": "Learn what this bot can do"},
    {"command": "language", "description": "Change the bot language"},
]


class TelegramClient:
    """Async client for Telegram Bot API."""

    def __init__(self) -> None:
        settings = get_settings()
        self._token = settings.telegram_bot_token
        self._base_url = f"https://api.telegram.org/bot{self._token}"
        self._client = httpx.AsyncClient(timeout=30.0)

    @staticmethod
    def _parse_response(response: httpx.Response) -> dict[str, Any]:
        """Raise on Telegram transport/API errors and return JSON payload."""
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok", True):
            raise ValueError(f"Telegram API error: {payload}")
        return payload

    async def send_message(
        self,
        chat_id: int | str,
        text: str,
        parse_mode: str = "MarkdownV2",
        reply_markup: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send text message to chat."""
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
        }

        if parse_mode:
            payload["parse_mode"] = parse_mode

        if reply_markup:
            payload["reply_markup"] = reply_markup

        try:
            response = await self._client.post(
                f"{self._base_url}/sendMessage",
                json=payload,
            )
            return self._parse_response(response)
        except httpx.HTTPStatusError as e:
            logger.error(f"Telegram API error: {e.response.text}")
            # Retry without parse_mode if markdown fails
            if parse_mode:
                payload.pop("parse_mode")
                response = await self._client.post(
                    f"{self._base_url}/sendMessage",
                    json=payload,
                )
                return self._parse_response(response)
            raise

    async def send_text(
        self,
        chat_id: int | str,
        text: str,
        parse_mode: str | None = None,
    ) -> dict[str, Any]:
        """Send plain text message (no markdown)."""
        return await self.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=parse_mode or "",
        )

    async def send_inline_keyboard(
        self,
        chat_id: int | str,
        text: str,
        buttons: list[list[dict[str, str]]],
        parse_mode: str | None = None,
    ) -> dict[str, Any]:
        """Send message with inline keyboard."""
        reply_markup = {
            "inline_keyboard": buttons
        }
        return await self.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=parse_mode or "",
            reply_markup=reply_markup,
        )

    async def answer_callback_query(
        self,
        callback_query_id: str,
        text: str | None = None,
        show_alert: bool = False,
    ) -> dict[str, Any]:
        """Answer callback query from inline keyboard."""
        payload: dict[str, Any] = {
            "callback_query_id": callback_query_id,
        }
        if text:
            payload["text"] = text
        if show_alert:
            payload["show_alert"] = show_alert

        response = await self._client.post(
            f"{self._base_url}/answerCallbackQuery",
            json=payload,
        )
        return self._parse_response(response)

    async def send_voice(
        self,
        chat_id: int | str,
        voice_data: bytes | str,
        caption: str | None = None,
        filename: str = "response.ogg",
        content_type: str = "audio/ogg",
    ) -> dict[str, Any]:
        """Send voice message from either a Telegram URL/file_id or raw bytes."""
        if isinstance(voice_data, bytes):
            files = {
                "voice": (filename, voice_data, content_type),
            }
            data: dict[str, Any] = {"chat_id": chat_id}
            if caption:
                data["caption"] = caption

            response = await self._client.post(
                f"{self._base_url}/sendVoice",
                data=data,
                files=files,
            )
            return self._parse_response(response)

        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "voice": voice_data,
        }
        if caption:
            payload["caption"] = caption

        response = await self._client.post(
            f"{self._base_url}/sendVoice",
            json=payload,
        )
        return self._parse_response(response)

    async def send_audio(
        self,
        chat_id: int | str,
        audio_bytes: bytes,
        filename: str = "response.ogg",
        caption: str | None = None,
        content_type: str = "audio/ogg",
    ) -> dict[str, Any]:
        """Send audio file using Telegram's generic audio endpoint."""
        files = {
            "audio": (filename, audio_bytes, content_type),
        }
        data: dict[str, Any] = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption

        response = await self._client.post(
            f"{self._base_url}/sendAudio",
            data=data,
            files=files,
        )
        return self._parse_response(response)

    async def get_file(self, file_id: str) -> dict[str, Any]:
        """Get file info for downloading."""
        response = await self._client.post(
            f"{self._base_url}/getFile",
            json={"file_id": file_id},
        )
        return self._parse_response(response)

    async def download_file(self, file_path: str) -> bytes:
        """Download file content."""
        url = f"https://api.telegram.org/file/bot{self._token}/{file_path}"
        response = await self._client.get(url)
        response.raise_for_status()
        return response.content

    async def download_voice(self, file_id: str) -> bytes:
        """Download voice message by file_id."""
        file_info = await self.get_file(file_id)
        if not file_info.get("ok"):
            raise ValueError(f"Failed to get file info: {file_info}")

        file_path = file_info["result"]["file_path"]
        return await self.download_file(file_path)

    async def set_webhook(self, url: str) -> dict[str, Any]:
        """Set webhook URL for receiving updates."""
        response = await self._client.post(
            f"{self._base_url}/setWebhook",
            json={"url": url},
        )
        return self._parse_response(response)

    async def set_my_commands(
        self,
        commands: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Register Telegram bot commands shown in the client UI."""
        response = await self._client.post(
            f"{self._base_url}/setMyCommands",
            json={"commands": commands or DEFAULT_BOT_COMMANDS},
        )
        return self._parse_response(response)

    async def get_my_commands(self) -> dict[str, Any]:
        """Fetch the currently registered Telegram bot commands."""
        response = await self._client.get(f"{self._base_url}/getMyCommands")
        return self._parse_response(response)

    async def delete_webhook(self) -> dict[str, Any]:
        """Delete webhook."""
        response = await self._client.post(
            f"{self._base_url}/deleteWebhook",
        )
        return self._parse_response(response)

    async def get_me(self) -> dict[str, Any]:
        """Get bot info."""
        response = await self._client.get(f"{self._base_url}/getMe")
        return self._parse_response(response)

    async def send_chat_action(
        self,
        chat_id: int | str,
        action: str = "typing",
    ) -> dict[str, Any]:
        """Send chat action (typing indicator)."""
        response = await self._client.post(
            f"{self._base_url}/sendChatAction",
            json={"chat_id": chat_id, "action": action},
        )
        return self._parse_response(response)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


# Global client instance
_telegram_client: TelegramClient | None = None


def get_telegram_client() -> TelegramClient:
    """Get or create Telegram client singleton."""
    global _telegram_client
    if _telegram_client is None:
        _telegram_client = TelegramClient()
    return _telegram_client
