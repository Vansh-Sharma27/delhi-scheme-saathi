"""Telegram webhook handler with voice support.

Handles incoming Telegram updates including:
- Text messages
- Voice messages (via Sarvam AI Saaras v3 STT)
- Callback queries (inline keyboard)
- Location sharing
"""

import logging
import os
from typing import Any

import asyncpg

from src.integrations.sarvam import get_sarvam_client
from src.integrations.telegram import get_telegram_client
from src.models.api import ChatRequest, TelegramUpdate
from src.services.conversation import ConversationService

logger = logging.getLogger(__name__)

# Minimum confidence threshold for STT
STT_CONFIDENCE_THRESHOLD = 0.5


def _get_voice_client():
    """Get the configured voice client (Sarvam AI or Bhashini).

    Prefers Sarvam AI if SARVAM_API_KEY is set, falls back to Bhashini.
    """
    sarvam_key = os.environ.get("SARVAM_API_KEY", "")
    if sarvam_key:
        return get_sarvam_client()

    # Fallback to Bhashini if configured
    bhashini_key = os.environ.get("BHASHINI_API_KEY", "")
    if bhashini_key:
        from src.integrations.bhashini import get_bhashini_client
        return get_bhashini_client()

    # Return Sarvam client (will handle no-key case gracefully)
    return get_sarvam_client()


async def handle_telegram_update(
    update_data: dict[str, Any],
    db_pool: asyncpg.Pool,
) -> dict[str, str]:
    """Handle incoming Telegram update.

    Routes to ConversationService and sends response via Telegram API.
    Supports text, voice, and callback query messages.
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
        text = await _handle_voice_message(update, chat_id)
        if not text:
            return {"status": "ok", "message": "voice_processed"}
    else:
        # Get text from message or callback
        text = update.text

    if not text:
        logger.warning(f"No text in update: {update_data}")
        return {"status": "ignored", "reason": "no_text"}

    # Build chat request
    request = ChatRequest(
        user_id=user_id,
        message=text,
        message_type="voice" if update.is_voice else ("callback" if update.is_callback else "text"),
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
        await _send_response(telegram, chat_id, response, update.is_voice)
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}", exc_info=True)
        # Try sending without any formatting
        try:
            await telegram.send_text(chat_id, response.text[:4000])
        except Exception:
            pass

    return {"status": "ok"}


async def _handle_voice_message(
    update: TelegramUpdate,
    chat_id: str,
) -> str | None:
    """Handle voice message using Sarvam AI or Bhashini STT.

    Downloads voice from Telegram, converts to text via voice service.
    Returns transcribed text or None if failed.
    """
    telegram = get_telegram_client()
    voice_client = _get_voice_client()

    # Check if voice service is configured
    if not voice_client.api_key:
        await telegram.send_text(
            chat_id,
            "🎤 Voice messages will be supported soon! Please type your message for now.\n\n"
            "🎤 जल्द ही आवाज़ संदेश समर्थित होंगे! अभी कृपया टाइप करें।"
        )
        return None

    try:
        # Get voice file info from update
        voice = update.message.get("voice", {})
        file_id = voice.get("file_id")

        if not file_id:
            logger.warning("No file_id in voice message")
            await telegram.send_text(chat_id, "Voice message could not be processed.")
            return None

        # Download voice file from Telegram
        logger.info(f"Downloading voice file: {file_id}")
        audio_bytes = await telegram.download_voice(file_id)

        if not audio_bytes:
            logger.warning("Failed to download voice file")
            await telegram.send_text(
                chat_id,
                "कृपया दोबारा बोलें। आवाज़ स्पष्ट नहीं थी।\n"
                "Please speak again. The audio wasn't clear."
            )
            return None

        # Convert speech to text via voice service
        logger.info(f"Processing voice through STT ({len(audio_bytes)} bytes)")
        result = await voice_client.speech_to_text(
            audio_bytes=audio_bytes,
            source_lang="hi",
            audio_format="ogg",
        )

        if not result.text or result.confidence < STT_CONFIDENCE_THRESHOLD:
            logger.warning(f"Low STT confidence: {result.confidence}")
            await telegram.send_text(
                chat_id,
                "🔊 आवाज़ स्पष्ट नहीं थी। कृपया दोबारा बोलें या टाइप करें।\n"
                "🔊 Audio wasn't clear. Please speak again or type your message."
            )
            return None

        # Show transcription to user for confirmation
        logger.info(f"STT result: '{result.text}' (confidence: {result.confidence})")
        await telegram.send_text(
            chat_id,
            f"🎤 आपने कहा: \"{result.text}\"\n"
            f"🎤 You said: \"{result.text}\""
        )

        return result.text

    except Exception as e:
        logger.error(f"Voice processing error: {e}", exc_info=True)
        await telegram.send_text(
            chat_id,
            "माफ़ कीजिए, आवाज़ प्रोसेस नहीं हो सकी। कृपया टाइप करें।\n"
            "Sorry, couldn't process voice. Please type your message."
        )
        return None


async def _send_response(
    telegram,
    chat_id: str,
    response,
    is_voice: bool = False,
) -> None:
    """Send response to user, optionally with voice.

    For voice requests, sends both text and audio response.
    """
    voice_client = _get_voice_client()

    # If we have inline keyboard (scheme selection)
    if response.inline_keyboard:
        await telegram.send_inline_keyboard(
            chat_id=chat_id,
            text=response.text,
            buttons=response.inline_keyboard,
        )
    else:
        # Send plain text
        await telegram.send_text(chat_id, response.text)

    # For voice requests, also send audio response if voice service is configured
    if is_voice and voice_client.api_key:
        try:
            # Generate TTS audio
            tts_result = await voice_client.text_to_speech(
                text=_clean_for_tts(response.text),
                target_lang="hi",
            )

            if tts_result.audio_bytes:
                await telegram.send_voice(chat_id, tts_result.audio_bytes)
        except Exception as e:
            logger.warning(f"TTS failed, skipping voice response: {e}")


def _clean_for_tts(text: str) -> str:
    """Clean text for TTS by removing emojis and markdown."""
    import re

    # Remove emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F700-\U0001F77F"  # alchemical
        "\U0001F780-\U0001F7FF"  # geometric
        "\U0001F800-\U0001F8FF"  # arrows
        "\U0001F900-\U0001F9FF"  # supplemental
        "\U0001FA00-\U0001FA6F"  # chess
        "\U0001FA70-\U0001FAFF"  # symbols
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub("", text)

    # Remove markdown formatting
    text = re.sub(r"\*+", "", text)  # bold
    text = re.sub(r"_+", "", text)   # italic
    text = re.sub(r"`+", "", text)   # code
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links

    # Remove multiple spaces/newlines
    text = re.sub(r"\s+", " ", text)

    return text.strip()[:500]  # Limit length for TTS


def extract_location(update: TelegramUpdate) -> tuple[float, float] | None:
    """Extract location from Telegram update if shared."""
    if update.message and update.message.get("location"):
        loc = update.message["location"]
        return (loc.get("latitude"), loc.get("longitude"))
    return None
