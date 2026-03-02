"""Telegram webhook handler."""

import logging
from typing import Any

import asyncpg

from src.integrations.telegram import get_telegram_client
from src.models.api import ChatRequest, TelegramUpdate
from src.services.conversation import ConversationService
from src.utils.formatters import escape_markdown_v2

logger = logging.getLogger(__name__)


async def handle_telegram_update(
    update_data: dict[str, Any],
    db_pool: asyncpg.Pool,
) -> dict[str, str]:
    """Handle incoming Telegram update.

    Routes to ConversationService and sends response via Telegram API.
    """
    telegram = get_telegram_client()

    # Parse update
    update = TelegramUpdate(**update_data)

    chat_id = update.chat_id
    user_id = update.user_id

    if not chat_id or not user_id:
        logger.warning(f"Invalid update, missing chat_id or user_id: {update_data}")
        return {"status": "ignored", "reason": "missing_ids"}

    # Send typing indicator
    await telegram.send_chat_action(chat_id, "typing")

    # Handle voice messages
    if update.is_voice:
        # For MVP, respond that voice is coming soon
        await telegram.send_text(
            chat_id,
            "🎤 Voice messages will be supported soon! Please type your message for now.\n\n"
            "🎤 जल्द ही आवाज़ संदेश समर्थित होंगे! अभी कृपया टाइप करें।"
        )
        return {"status": "ok", "message": "voice_not_supported"}

    # Get text from message or callback
    text = update.text
    if not text:
        logger.warning(f"No text in update: {update_data}")
        return {"status": "ignored", "reason": "no_text"}

    # Build chat request
    request = ChatRequest(
        user_id=user_id,
        message=text,
        message_type="callback" if update.is_callback else "text",
        callback_data=update.callback_query.get("data") if update.is_callback else None,
    )

    # Handle callback query acknowledgment
    if update.is_callback:
        callback_id = update.callback_query.get("id")
        if callback_id:
            await telegram.answer_callback_query(callback_id)

    # Process through conversation service
    try:
        conversation = ConversationService(db_pool)
        response = await conversation.handle_message(request)
    except Exception as e:
        logger.error(f"Conversation error: {e}", exc_info=True)
        await telegram.send_text(
            chat_id,
            "माफ़ कीजिए, कुछ तकनीकी समस्या है। कृपया थोड़ी देर बाद प्रयास करें।\n\n"
            "Sorry, there was a technical issue. Please try again later."
        )
        return {"status": "error", "message": str(e)}

    # Send response
    try:
        # If we have inline keyboard (scheme selection)
        if response.inline_keyboard:
            await telegram.send_inline_keyboard(
                chat_id=chat_id,
                text=response.text,
                buttons=response.inline_keyboard,
            )
        else:
            # Send plain text (without markdown to avoid escaping issues)
            await telegram.send_text(chat_id, response.text)

    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}", exc_info=True)
        # Try sending without any formatting
        try:
            await telegram.send_text(chat_id, response.text[:4000])
        except Exception:
            pass

    return {"status": "ok"}


def extract_location(update: TelegramUpdate) -> tuple[float, float] | None:
    """Extract location from Telegram update if shared."""
    if update.message and update.message.get("location"):
        loc = update.message["location"]
        return (loc.get("latitude"), loc.get("longitude"))
    return None
