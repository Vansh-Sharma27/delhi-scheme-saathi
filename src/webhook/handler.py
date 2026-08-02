"""Telegram webhook handler with voice support.

Handles incoming Telegram updates including:
- Text messages
- Voice messages (via Sarvam AI Saaras v3 STT)
- Callback queries (inline keyboard)
- Location sharing
"""

import logging
import re
from contextlib import suppress
from typing import Any

import asyncpg

from src.config import get_settings
from src.integrations.sarvam import get_sarvam_client
from src.integrations.telegram import get_telegram_client
from src.models.api import ChatRequest, TelegramUpdate
from src.services import session_manager
from src.services.conversation import ConversationService, language

logger = logging.getLogger(__name__)

# Minimum confidence threshold for STT
STT_CONFIDENCE_THRESHOLD = 0.5
TTS_MAX_TEXT_LENGTH = 900


def _get_voice_client():
    """Get the configured voice client (Sarvam AI or Bhashini).

    Prefers Sarvam AI when its key is set, falls back to Bhashini. With
    neither configured this still returns the Sarvam client, which reports
    itself as unconfigured so callers can tell the user voice is off.
    """
    settings = get_settings()
    if settings.sarvam_api_key:
        return get_sarvam_client()

    if settings.bhashini_api_key:
        from src.integrations.bhashini import get_bhashini_client
        return get_bhashini_client()

    return get_sarvam_client()


# Deliberately narrower than the marker set used for typed messages. A
# transcript is the speech recogniser's guess, so a single common word is weak
# evidence; requiring two of these unambiguous ones keeps STT scoring from
# swinging to Hinglish on a mis-recognition.
_TRANSCRIPT_HINGLISH_MARKERS = (
    "mujhe",
    "chahiye",
    "batao",
    "batayiye",
    "madad",
    "sahayata",
)
_TRANSCRIPT_MARKER_THRESHOLD = 2


def _infer_transcript_language(text: str) -> str:
    """Infer transcript language using a light heuristic."""
    if language.devanagari_ratio(text) > language.DEVANAGARI_THRESHOLD:
        return "hi"

    marker_hits = language.count_markers(text.lower(), _TRANSCRIPT_HINGLISH_MARKERS)
    if marker_hits >= _TRANSCRIPT_MARKER_THRESHOLD:
        return "hinglish"
    return "en"


def _stt_language_candidates(session) -> list[str]:
    """Return STT probe order based on session language state."""
    if session.language_locked and session.language_preference == "en":
        return ["en", "hi"]
    if session.language_locked and session.language_preference == "hi":
        return ["hi", "en"]
    if session.language_locked and session.language_preference == "hinglish":
        return ["hi", "en"]

    # Unlocked language is only a soft observation from previous turns, so it
    # must not bias voice recognition on the next message.
    return ["en", "hi"]


def _guess_audio_format(update: TelegramUpdate) -> str:
    """Infer audio format from the Telegram update payload."""
    media = {}
    if update.message:
        media = update.message.get("voice") or update.message.get("audio") or {}
    mime_type = media.get("mime_type", "")
    if "mpeg" in mime_type or mime_type.endswith("/mp3"):
        return "mp3"
    if "mp4" in mime_type or "m4a" in mime_type:
        return "m4a"
    if "webm" in mime_type:
        return "webm"
    return "ogg"


def _tts_filename(content_type: str | None) -> str:
    """Choose a filename that matches the synthesized audio type."""
    extension_map = {
        "audio/ogg": "ogg",
        "audio/wav": "wav",
        "audio/mpeg": "mp3",
        "audio/mp4": "m4a",
    }
    extension = extension_map.get(content_type or "", "bin")
    return f"response.{extension}"


def _transcript_echo_text(language: str, transcript: str) -> str:
    """Render the STT transcript back to the user for testing/debugging."""
    if language == "hi":
        return f"आपने कहा: {transcript}"
    if language == "hinglish":
        return f"Aapne kaha: {transcript}"
    return f"You said: {transcript}"


def _echo_language(session, transcript_language: str) -> str:
    """Choose the visible echo prefix language.

    The visible prefix should honor an explicit user language lock even when
    the best STT transcript candidate happened to be English.
    """
    locked_language = getattr(session, "language_preference", "auto")
    if getattr(session, "language_locked", False) and locked_language in {"hi", "en", "hinglish"}:
        return locked_language
    if transcript_language in {"hi", "en", "hinglish"}:
        return transcript_language
    return "en"


async def _transcribe_with_fallbacks(
    voice_client,
    audio_bytes: bytes,
    audio_format: str,
    language_candidates: list[str],
):
    """Run STT against one or more language candidates and pick the best result."""
    def _transcript_quality_score(text: str) -> float:
        """Prefer fuller transcripts over short/noisy recognitions."""
        stripped = text.strip()
        if not stripped:
            return -1.0
        if stripped.startswith("[") and stripped.endswith("]"):
            return -0.5

        tokens = re.findall(r"[A-Za-z0-9\u0900-\u097F]+", stripped)
        meaningful_tokens = [token for token in tokens if len(token) > 1]
        alpha_chars = sum(1 for char in stripped if char.isalpha())

        score = min(len(meaningful_tokens), 6) * 0.05
        if alpha_chars:
            score += min(alpha_chars / max(len(stripped), 1), 1.0) * 0.1
        return score

    best_result = None
    best_score = -1.0

    for candidate in language_candidates:
        result = await voice_client.speech_to_text(
            audio_bytes=audio_bytes,
            source_lang=candidate,
            audio_format=audio_format,
        )
        if not result.text:
            continue

        score = float(result.confidence or 0.0) + _transcript_quality_score(result.text)
        transcript_language = _infer_transcript_language(result.text)
        # Reward a transcript that reads like the language it was decoded as,
        # and penalise one the recogniser itself labels as a different one.
        if (candidate == "en" and transcript_language == "en") or (
            candidate == "hi" and transcript_language in {"hi", "hinglish"}
        ):
            score += 0.2

        detected_language = getattr(result, "language", None)
        if detected_language in {"en", "hi", "hinglish"}:
            score += 0.15 if detected_language == candidate else -0.2

        if score > best_score:
            best_result = result
            best_score = score

    return best_result


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
        logger.warning("Invalid update, missing chat_id or user_id")
        return {"status": "ignored", "reason": "missing_ids"}

    # Send typing indicator
    await telegram.send_chat_action(chat_id, "typing")

    # Load session to get language preference (needed for STT)
    session = await session_manager.get_or_create_session(user_id)
    is_voice_input = update.is_voice or update.is_audio

    # Handle voice messages
    if is_voice_input:
        caption_text = update.text
        transcript_text = await _handle_voice_message(update, chat_id, session)
        text = "\n".join(part for part in (caption_text, transcript_text) if part)
        if not text:
            return {"status": "ok", "message": "voice_processed"}
    else:
        # Get text from message or callback
        text = update.text

    if not text:
        logger.warning("No text in update for chat_id=%s", chat_id)
        return {"status": "ignored", "reason": "no_text"}

    # Build chat request
    request = ChatRequest(
        user_id=user_id,
        message=text,
        message_type="voice" if is_voice_input else ("callback" if update.is_callback else "text"),
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
        logger.error("Conversation error for chat_id=%s: %s", chat_id, e, exc_info=True)
        await telegram.send_text(
            chat_id,
            "माफ़ कीजिए, कुछ तकनीकी समस्या है। कृपया थोड़ी देर बाद प्रयास करें।\n\n"
            "Sorry, there was a technical issue. Please try again later."
        )
        return {"status": "error", "message": "internal_error"}

    # Send response
    try:
        await _send_response(telegram, chat_id, response, is_voice_input)
    except Exception as e:
        logger.error("Failed to send Telegram message: %s", e, exc_info=True)
        # Try sending without any formatting
        with suppress(Exception):
            await telegram.send_text(chat_id, _clean_for_telegram(response.text)[:4000])

    return {"status": "ok"}


async def _handle_voice_message(
    update: TelegramUpdate,
    chat_id: str,
    session=None,
) -> str | None:
    """Handle voice message using Sarvam AI or Bhashini STT.

    Downloads voice from Telegram, converts to text via voice service.
    Uses session language state to probe STT without forcing Hindi on unlocked sessions.
    Returns transcribed text or None if failed.
    """
    telegram = get_telegram_client()
    voice_client = _get_voice_client()
    if session is None:
        session = type(
            "VoiceSession",
            (),
            {"language_preference": "auto", "language_locked": False},
        )()

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
        file_id = update.media_file_id

        if not file_id:
            logger.warning("No file_id in voice message")
            await telegram.send_text(chat_id, "Voice message could not be processed.")
            return None

        # Download voice file from Telegram
        logger.info("Downloading voice file: %s", file_id)
        audio_bytes = await telegram.download_voice(file_id)

        if not audio_bytes:
            logger.warning("Failed to download voice file")
            await telegram.send_text(
                chat_id,
                "कृपया दोबारा बोलें। आवाज़ स्पष्ट नहीं थी।\n"
                "Please speak again. The audio wasn't clear."
            )
            return None

        audio_format = _guess_audio_format(update)
        language_candidates = _stt_language_candidates(session)
        logger.info(
            "Processing voice through STT (%s bytes, candidates=%s)",
            len(audio_bytes),
            language_candidates,
        )
        result = await _transcribe_with_fallbacks(
            voice_client,
            audio_bytes,
            audio_format,
            language_candidates,
        )

        if not result or not result.text or (result.confidence or 0.0) < STT_CONFIDENCE_THRESHOLD:
            logger.warning(
                "Low STT confidence after probing: %s",
                result.confidence if result else None,
            )
            await telegram.send_text(
                chat_id,
                "🔊 आवाज़ स्पष्ट नहीं थी। कृपया दोबारा बोलें या टाइप करें।\n"
                "🔊 Audio wasn't clear. Please speak again or type your message."
            )
            return None

        logger.info("STT result: '%s' (confidence: %s)", result.text, result.confidence)

        transcript_language = result.language
        if transcript_language not in {"en", "hi", "hinglish"}:
            transcript_language = _infer_transcript_language(result.text)
        echo_language = _echo_language(session, transcript_language)
        await telegram.send_text(
            chat_id,
            _transcript_echo_text(echo_language, result.text),
        )

        # Return transcribed text after sending a visible transcript echo.
        return result.text

    except Exception as e:
        logger.error("Voice processing error: %s", e, exc_info=True)
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
    Long messages (>4000 chars) are split at section boundaries
    to stay within Telegram's 4096 char limit.
    """
    voice_client = _get_voice_client()

    clean_text = _clean_for_telegram(response.text)

    # Split long messages at section dividers to maintain readability
    parts = _split_message(clean_text)

    # Send all parts except the last without inline keyboard
    for part in parts[:-1]:
        await telegram.send_text(chat_id, part)

    # Send the last part (with inline keyboard if present)
    last_part = parts[-1] if parts else clean_text
    if response.inline_keyboard:
        await telegram.send_inline_keyboard(
            chat_id=chat_id,
            text=last_part,
            buttons=response.inline_keyboard,
        )
    else:
        await telegram.send_text(chat_id, last_part)

    # For voice requests, also send audio response if voice service is configured
    if is_voice and voice_client.api_key:
        try:
            clean_tts_text = _clean_for_tts(clean_text)
            if response.language == "hinglish":
                logger.info("Skipping TTS for Hinglish response")
                return
            if len(parts) > 1 or len(clean_tts_text) > TTS_MAX_TEXT_LENGTH:
                logger.info("Skipping TTS for long structured response")
                return

            tts_lang = response.language
            tts_result = await voice_client.text_to_speech(
                text=clean_tts_text,
                target_lang=tts_lang,
            )

            if tts_result.audio_bytes:
                filename = _tts_filename(tts_result.content_type)
                if tts_result.content_type == "audio/ogg":
                    await telegram.send_voice(
                        chat_id,
                        tts_result.audio_bytes,
                        filename=filename,
                        content_type=tts_result.content_type,
                    )
                else:
                    await telegram.send_audio(
                        chat_id,
                        tts_result.audio_bytes,
                        filename=filename,
                        content_type=tts_result.content_type,
                    )
        except Exception as e:
            logger.warning("TTS failed, skipping voice response: %s", e)


def _clean_for_telegram(text: str) -> str:
    """Normalize markdown-heavy LLM output for plain Telegram text rendering."""
    if not text:
        return ""

    # Remove markdown headers and emphasis/code markers that appear raw in Telegram.
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"`{1,3}([^`]+)`{1,3}", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)

    # Convert markdown links [text](url) to "text (url)" for readability.
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)

    # Normalize spacing while preserving readable line breaks.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Telegram rejects messages over 4096 characters; the buffer leaves room for
# the trailing newline Telegram clients sometimes add.
_TG_MAX_LEN = 4000


def _split_message(text: str) -> list[str]:
    """Split a long message into parts that fit Telegram's length limit.

    Splits between paragraphs so a card is never cut mid-line. A single
    paragraph longer than the limit is emitted as its own oversized part
    rather than being chopped, because breaking mid-sentence in Hindi reads
    worse than one long message; the caller sends it and lets Telegram
    complain if it truly cannot fit.
    """
    if len(text) <= _TG_MAX_LEN:
        return [text]

    parts: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= _TG_MAX_LEN:
            current = candidate
            continue
        if current:
            parts.append(current.strip())
        current = paragraph

    if current:
        parts.append(current.strip())

    return parts or [text[:_TG_MAX_LEN]]


def _clean_for_tts(text: str) -> str:
    """Clean text for TTS by removing emojis and markdown."""
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

    return text.strip()


def extract_location(update: TelegramUpdate) -> tuple[float, float] | None:
    """Extract location from Telegram update if shared."""
    if update.message and update.message.get("location"):
        loc = update.message["location"]
        return (loc.get("latitude"), loc.get("longitude"))
    return None
