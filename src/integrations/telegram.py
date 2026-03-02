"""Telegram Bot API client."""

import logging
from typing import Any

import httpx

from src.config import get_settings

logger = logging.getLogger(__name__)


class TelegramClient:
    """Async client for Telegram Bot API."""

    def __init__(self) -> None:
        settings = get_settings()
        self._token = settings.telegram_bot_token
        self._base_url = f"https://api.telegram.org/bot{self._token}"
        self._client = httpx.AsyncClient(timeout=30.0)

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
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Telegram API error: {e.response.text}")
            # Retry without parse_mode if markdown fails
            if parse_mode:
                payload.pop("parse_mode")
                response = await self._client.post(
                    f"{self._base_url}/sendMessage",
                    json=payload,
                )
                return response.json()
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
        return response.json()

    async def send_voice(
        self,
        chat_id: int | str,
        voice_url: str,
        caption: str | None = None,
    ) -> dict[str, Any]:
        """Send voice message."""
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "voice": voice_url,
        }
        if caption:
            payload["caption"] = caption

        response = await self._client.post(
            f"{self._base_url}/sendVoice",
            json=payload,
        )
        return response.json()

    async def send_audio(
        self,
        chat_id: int | str,
        audio_bytes: bytes,
        filename: str = "response.ogg",
        caption: str | None = None,
    ) -> dict[str, Any]:
        """Send audio file as voice message."""
        files = {
            "voice": (filename, audio_bytes, "audio/ogg"),
        }
        data: dict[str, Any] = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption

        response = await self._client.post(
            f"{self._base_url}/sendVoice",
            data=data,
            files=files,
        )
        return response.json()

    async def get_file(self, file_id: str) -> dict[str, Any]:
        """Get file info for downloading."""
        response = await self._client.post(
            f"{self._base_url}/getFile",
            json={"file_id": file_id},
        )
        return response.json()

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
        return response.json()

    async def delete_webhook(self) -> dict[str, Any]:
        """Delete webhook."""
        response = await self._client.post(
            f"{self._base_url}/deleteWebhook",
        )
        return response.json()

    async def get_me(self) -> dict[str, Any]:
        """Get bot info."""
        response = await self._client.get(f"{self._base_url}/getMe")
        return response.json()

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
        return response.json()

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
