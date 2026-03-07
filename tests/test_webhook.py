"""Tests for Telegram webhook handler."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.models.api import TelegramUpdate
from src.integrations.sarvam import STTResult, TTSResult


class TestTelegramUpdate:
    """Tests for TelegramUpdate model."""

    def test_text_message_parsing(self):
        """Test parsing of text message update."""
        update_data = {
            "update_id": 123456,
            "message": {
                "message_id": 1,
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890, "first_name": "Test"},
                "text": "Hello",
            },
        }
        update = TelegramUpdate(**update_data)
        # chat_id and user_id can be int or str depending on TelegramUpdate implementation
        assert str(update.chat_id) == "12345"
        assert str(update.user_id) == "67890"
        assert update.text == "Hello"
        assert not update.is_voice
        assert not update.is_callback

    def test_voice_message_parsing(self):
        """Test parsing of voice message update."""
        update_data = {
            "update_id": 123456,
            "message": {
                "message_id": 1,
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890, "first_name": "Test"},
                "voice": {
                    "file_id": "voice_file_123",
                    "duration": 5,
                },
            },
        }
        update = TelegramUpdate(**update_data)
        assert update.is_voice
        assert update.message.get("voice", {}).get("file_id") == "voice_file_123"

    def test_callback_query_parsing(self):
        """Test parsing of callback query update."""
        update_data = {
            "update_id": 123456,
            "callback_query": {
                "id": "callback_123",
                "from": {"id": 67890, "first_name": "Test"},
                "message": {
                    "message_id": 1,
                    "chat": {"id": 12345, "type": "private"},
                },
                "data": "scheme:SCH-001",
            },
        }
        update = TelegramUpdate(**update_data)
        assert update.is_callback
        assert update.callback_query.get("data") == "scheme:SCH-001"

    def test_audio_message_parsing(self):
        """Audio uploads should be treated as voice-like input."""
        update_data = {
            "update_id": 123457,
            "message": {
                "message_id": 2,
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890, "first_name": "Test"},
                "audio": {
                    "file_id": "audio_file_123",
                    "duration": 5,
                    "mime_type": "audio/ogg",
                },
                "caption": "please use english",
            },
        }
        update = TelegramUpdate(**update_data)
        assert update.is_audio
        assert update.media_file_id == "audio_file_123"
        assert update.text == "please use english"


class TestVoiceMessageHandling:
    """Tests for voice message processing."""

    @pytest.mark.asyncio
    async def test_voice_without_api_key_sends_fallback(self):
        """Test voice message without voice API key sends fallback."""
        from src.webhook.handler import _handle_voice_message

        update_data = {
            "update_id": 123456,
            "message": {
                "message_id": 1,
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890, "first_name": "Test"},
                "voice": {"file_id": "voice_123", "duration": 3},
            },
        }
        update = TelegramUpdate(**update_data)

        # Mock Telegram client
        mock_telegram = AsyncMock()

        # Mock voice client with no API key
        mock_voice_client = MagicMock()
        mock_voice_client.api_key = ""

        with patch(
            "src.webhook.handler.get_telegram_client", return_value=mock_telegram
        ), patch(
            "src.webhook.handler._get_voice_client", return_value=mock_voice_client
        ):
            result = await _handle_voice_message(update, "12345")

            # Should send "coming soon" message and return None
            assert result is None
            mock_telegram.send_text.assert_called_once()
            call_args = mock_telegram.send_text.call_args[0]
            assert "soon" in call_args[1].lower()

    @pytest.mark.asyncio
    async def test_voice_with_api_key_success(self):
        """Test voice message with voice API key returns transcription."""
        from src.webhook.handler import _handle_voice_message

        update_data = {
            "update_id": 123456,
            "message": {
                "message_id": 1,
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890, "first_name": "Test"},
                "voice": {"file_id": "voice_123", "duration": 3},
            },
        }
        update = TelegramUpdate(**update_data)

        # Mock Telegram client
        mock_telegram = AsyncMock()
        mock_telegram.download_voice = AsyncMock(return_value=b"audio data")

        # Mock voice client with API key
        mock_voice_client = MagicMock()
        mock_voice_client.api_key = "test-key"
        mock_voice_client.speech_to_text = AsyncMock(
            return_value=STTResult(
                text="मुझे पेंशन चाहिए",
                confidence=0.9,
                language="hi",
            )
        )

        with patch(
            "src.webhook.handler.get_telegram_client", return_value=mock_telegram
        ), patch(
            "src.webhook.handler._get_voice_client", return_value=mock_voice_client
        ):
            result = await _handle_voice_message(update, "12345")

            assert result == "मुझे पेंशन चाहिए"
            mock_telegram.download_voice.assert_called_once_with("voice_123")
            assert mock_voice_client.speech_to_text.await_count >= 1

    @pytest.mark.asyncio
    async def test_voice_low_confidence_asks_retry(self):
        """Test low confidence STT asks user to retry."""
        from src.webhook.handler import _handle_voice_message

        update_data = {
            "update_id": 123456,
            "message": {
                "message_id": 1,
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890, "first_name": "Test"},
                "voice": {"file_id": "voice_123", "duration": 3},
            },
        }
        update = TelegramUpdate(**update_data)

        mock_telegram = AsyncMock()
        mock_telegram.download_voice = AsyncMock(return_value=b"audio data")

        mock_voice_client = MagicMock()
        mock_voice_client.api_key = "test-key"
        mock_voice_client.speech_to_text = AsyncMock(
            return_value=STTResult(
                text="unclear",
                confidence=0.3,  # Below threshold
                language="hi",
            )
        )

        with patch(
            "src.webhook.handler.get_telegram_client", return_value=mock_telegram
        ), patch(
            "src.webhook.handler._get_voice_client", return_value=mock_voice_client
        ):
            result = await _handle_voice_message(update, "12345")

            # Should return None and ask to retry
            assert result is None
            mock_telegram.send_text.assert_called_once()
            call_args = mock_telegram.send_text.call_args[0]
            assert "again" in call_args[1].lower() or "दोबारा" in call_args[1]

    @pytest.mark.asyncio
    async def test_voice_prefers_locked_english_before_hindi(self):
        """Locked English sessions should probe English first."""
        from src.webhook.handler import _handle_voice_message

        update = TelegramUpdate(
            update_id=1,
            message={
                "message_id": 1,
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890, "first_name": "Test"},
                "voice": {"file_id": "voice_123", "duration": 3},
            },
        )
        mock_telegram = AsyncMock()
        mock_telegram.download_voice = AsyncMock(return_value=b"audio data")
        mock_voice_client = MagicMock()
        mock_voice_client.api_key = "test-key"
        mock_voice_client.speech_to_text = AsyncMock(
            side_effect=[
                STTResult(text="I need housing help", confidence=0.85, language="en"),
                STTResult(text="", confidence=0.0, language="hi"),
            ]
        )
        session = type(
            "VoiceSession",
            (),
            {"language_preference": "en", "language_locked": True},
        )()

        with patch(
            "src.webhook.handler.get_telegram_client", return_value=mock_telegram
        ), patch(
            "src.webhook.handler._get_voice_client", return_value=mock_voice_client
        ):
            result = await _handle_voice_message(update, "12345", session)

        assert result == "I need housing help"
        first_call = mock_voice_client.speech_to_text.await_args_list[0]
        assert first_call.kwargs["source_lang"] == "en"

    @pytest.mark.asyncio
    async def test_unlocked_hindi_history_does_not_bias_voice_probe_order(self):
        """Unlocked prior language should not force Hindi-first STT."""
        from src.webhook.handler import _handle_voice_message

        update = TelegramUpdate(
            update_id=2,
            message={
                "message_id": 1,
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890, "first_name": "Test"},
                "voice": {"file_id": "voice_123", "duration": 3},
            },
        )
        mock_telegram = AsyncMock()
        mock_telegram.download_voice = AsyncMock(return_value=b"audio data")
        mock_voice_client = MagicMock()
        mock_voice_client.api_key = "test-key"
        mock_voice_client.speech_to_text = AsyncMock(
            side_effect=[
                STTResult(text="I need housing help", confidence=0.82, language="en"),
                STTResult(text="मुझे सहायता चाहिए", confidence=0.7, language="hi"),
            ]
        )
        session = type(
            "VoiceSession",
            (),
            {"language_preference": "hi", "language_locked": False},
        )()

        with patch(
            "src.webhook.handler.get_telegram_client", return_value=mock_telegram
        ), patch(
            "src.webhook.handler._get_voice_client", return_value=mock_voice_client
        ):
            result = await _handle_voice_message(update, "12345", session)

        assert result == "I need housing help"
        first_call = mock_voice_client.speech_to_text.await_args_list[0]
        assert first_call.kwargs["source_lang"] == "en"

    @pytest.mark.asyncio
    async def test_audio_caption_is_combined_with_transcript(self):
        """Audio captions should survive alongside the STT transcript."""
        from src.webhook.handler import handle_telegram_update

        update_data = {
            "update_id": 99,
            "message": {
                "message_id": 1,
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890, "first_name": "Test"},
                "audio": {
                    "file_id": "audio_123",
                    "duration": 5,
                    "mime_type": "audio/ogg",
                },
                "caption": "please use english",
            },
        }

        mock_telegram = AsyncMock()
        mock_voice_client = MagicMock()
        mock_voice_client.api_key = "test-key"
        mock_voice_client.speech_to_text = AsyncMock(
            return_value=STTResult(
                text="I need housing help",
                confidence=0.91,
                language="en",
            )
        )
        mock_service = MagicMock()
        mock_service.handle_message = AsyncMock(
            return_value=MagicMock(
                text="What is your age?",
                inline_keyboard=None,
                language="en",
            )
        )
        session = type(
            "VoiceSession",
            (),
            {"language_preference": "auto", "language_locked": False},
        )()

        with patch(
            "src.webhook.handler.get_telegram_client", return_value=mock_telegram
        ), patch(
            "src.webhook.handler._get_voice_client", return_value=mock_voice_client
        ), patch(
            "src.webhook.handler.ConversationService", return_value=mock_service
        ), patch(
            "src.webhook.handler._send_response", AsyncMock()
        ), patch(
            "src.webhook.handler.session_manager.get_or_create_session",
            AsyncMock(return_value=session),
        ):
            result = await handle_telegram_update(update_data, AsyncMock())

        request = mock_service.handle_message.await_args.args[0]
        assert request.message == "please use english\nI need housing help"
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_transcription_scoring_penalizes_probe_language_mismatch(self):
        """Provider-detected language should outweigh a mismatched probe."""
        from src.webhook.handler import _transcribe_with_fallbacks

        mock_voice_client = MagicMock()
        mock_voice_client.speech_to_text = AsyncMock(
            side_effect=[
                STTResult(
                    text="I need housing help",
                    confidence=0.9,
                    language="en",
                ),
                STTResult(
                    text="I need housing help",
                    confidence=0.82,
                    language="en",
                ),
            ]
        )

        result = await _transcribe_with_fallbacks(
            mock_voice_client,
            b"audio data",
            "ogg",
            ["hi", "en"],
        )

        assert result is not None
        assert result.language == "en"
        second_call = mock_voice_client.speech_to_text.await_args_list[1]
        assert second_call.kwargs["source_lang"] == "en"


class TestSendResponse:
    """Tests for response sending."""

    @pytest.mark.asyncio
    async def test_send_text_response(self):
        """Test sending text-only response."""
        from src.webhook.handler import _send_response

        mock_telegram = AsyncMock()
        mock_voice_client = MagicMock()
        mock_voice_client.api_key = ""

        # Create mock response
        mock_response = MagicMock()
        mock_response.text = "Hello there!"
        mock_response.inline_keyboard = None

        with patch(
            "src.webhook.handler._get_voice_client", return_value=mock_voice_client
        ):
            await _send_response(mock_telegram, "12345", mock_response, is_voice=False)

            mock_telegram.send_text.assert_called_once_with("12345", "Hello there!")
            mock_telegram.send_voice.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_inline_keyboard_response(self):
        """Test sending response with inline keyboard."""
        from src.webhook.handler import _send_response

        mock_telegram = AsyncMock()
        mock_voice_client = MagicMock()
        mock_voice_client.api_key = ""

        mock_response = MagicMock()
        mock_response.text = "Select a scheme:"
        mock_response.inline_keyboard = [
            {"text": "Scheme 1", "callback_data": "SCH-001"},
            {"text": "Scheme 2", "callback_data": "SCH-002"},
        ]

        with patch(
            "src.webhook.handler._get_voice_client", return_value=mock_voice_client
        ):
            await _send_response(mock_telegram, "12345", mock_response, is_voice=False)

            mock_telegram.send_inline_keyboard.assert_called_once()
            mock_telegram.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_voice_response_with_tts(self):
        """Non-OGG TTS should use Telegram's audio endpoint."""
        from src.webhook.handler import _send_response

        mock_telegram = AsyncMock()

        mock_voice_client = MagicMock()
        mock_voice_client.api_key = "test-key"
        mock_voice_client.text_to_speech = AsyncMock(
            return_value=TTSResult(
                audio_bytes=b"audio data",
                content_type="audio/wav",
            )
        )

        mock_response = MagicMock()
        mock_response.text = "नमस्ते!"
        mock_response.inline_keyboard = None
        mock_response.language = "hi"
        mock_response.next_state = "PRESENTING"

        with patch(
            "src.webhook.handler._get_voice_client", return_value=mock_voice_client
        ):
            await _send_response(mock_telegram, "12345", mock_response, is_voice=True)

            # Should send both text and generic audio for WAV bytes
            mock_telegram.send_text.assert_called_once()
            mock_telegram.send_audio.assert_called_once_with(
                "12345",
                b"audio data",
                filename="response.wav",
                content_type="audio/wav",
            )
            mock_telegram.send_voice.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_voice_response_uses_voice_endpoint_for_ogg(self):
        """Native OGG voice bytes should stay on Telegram's voice endpoint."""
        from src.webhook.handler import _send_response

        mock_telegram = AsyncMock()

        mock_voice_client = MagicMock()
        mock_voice_client.api_key = "test-key"
        mock_voice_client.text_to_speech = AsyncMock(
            return_value=TTSResult(
                audio_bytes=b"ogg audio",
                content_type="audio/ogg",
            )
        )

        mock_response = MagicMock()
        mock_response.text = "नमस्ते!"
        mock_response.inline_keyboard = None
        mock_response.language = "hi"
        mock_response.next_state = "PRESENTING"

        with patch(
            "src.webhook.handler._get_voice_client", return_value=mock_voice_client
        ):
            await _send_response(mock_telegram, "12345", mock_response, is_voice=True)

        mock_telegram.send_text.assert_called_once()
        mock_telegram.send_voice.assert_called_once_with(
            "12345",
            b"ogg audio",
            filename="response.ogg",
            content_type="audio/ogg",
        )
        mock_telegram.send_audio.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_voice_response_skips_hinglish_tts(self):
        """Hinglish text should stay text-only for voice replies."""
        from src.webhook.handler import _send_response

        mock_telegram = AsyncMock()
        mock_voice_client = MagicMock()
        mock_voice_client.api_key = "test-key"
        mock_voice_client.text_to_speech = AsyncMock()

        mock_response = MagicMock()
        mock_response.text = "Scheme details chahiye? Batayiye."
        mock_response.inline_keyboard = None
        mock_response.language = "hinglish"
        mock_response.next_state = "DETAILS"

        with patch(
            "src.webhook.handler._get_voice_client", return_value=mock_voice_client
        ):
            await _send_response(mock_telegram, "12345", mock_response, is_voice=True)

        mock_telegram.send_text.assert_called_once()
        mock_voice_client.text_to_speech.assert_not_called()
        mock_telegram.send_voice.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_voice_response_skips_long_structured_tts(self):
        """Long structured replies should not be truncated into partial voice audio."""
        from src.webhook.handler import _send_response

        mock_telegram = AsyncMock()
        mock_voice_client = MagicMock()
        mock_voice_client.api_key = "test-key"
        mock_voice_client.text_to_speech = AsyncMock()

        mock_response = MagicMock()
        mock_response.text = "A" * 1000
        mock_response.inline_keyboard = None
        mock_response.language = "en"
        mock_response.next_state = "DETAILS"

        with patch(
            "src.webhook.handler._get_voice_client", return_value=mock_voice_client
        ):
            await _send_response(mock_telegram, "12345", mock_response, is_voice=True)

        mock_telegram.send_text.assert_called_once()
        mock_voice_client.text_to_speech.assert_not_called()


class TestCleanForTTS:
    """Tests for TTS text cleaning."""

    def test_removes_emojis(self):
        """Test emoji removal from text."""
        from src.webhook.handler import _clean_for_tts

        text = "🎤 नमस्ते 🙏 आपका स्वागत है"
        result = _clean_for_tts(text)
        assert "🎤" not in result
        assert "🙏" not in result
        assert "नमस्ते" in result

    def test_removes_markdown(self):
        """Test markdown removal from text."""
        from src.webhook.handler import _clean_for_tts

        text = "**Bold** and *italic* and `code`"
        result = _clean_for_tts(text)
        assert "**" not in result
        assert "*" not in result
        assert "`" not in result
        assert "Bold" in result

    def test_removes_links(self):
        """Test link removal from text."""
        from src.webhook.handler import _clean_for_tts

        text = "Visit [portal](https://example.com) for more"
        result = _clean_for_tts(text)
        assert "[portal]" not in result
        assert "https://" not in result
        assert "portal" in result

    def test_keeps_long_text_intact(self):
        """Text cleaning should not silently truncate TTS content."""
        from src.webhook.handler import _clean_for_tts

        text = "a" * 1000
        result = _clean_for_tts(text)
        assert len(result) == 1000


class TestCleanForTelegram:
    """Tests for Telegram text cleaning."""

    def test_removes_markdown_artifacts(self):
        """Headers and markdown markers should be normalized for plain text."""
        from src.webhook.handler import _clean_for_telegram

        text = "### Welcome\n\n**Bold** text and `code` with _italics_."
        result = _clean_for_telegram(text)

        assert "###" not in result
        assert "**" not in result
        assert "`" not in result
        assert "_italics_" not in result
        assert "Welcome" in result
        assert "Bold text" in result


class TestExtractLocation:
    """Tests for location extraction."""

    def test_extracts_location(self):
        """Test location extraction from update."""
        from src.webhook.handler import extract_location

        update_data = {
            "update_id": 123456,
            "message": {
                "message_id": 1,
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890, "first_name": "Test"},
                "location": {
                    "latitude": 28.6139,
                    "longitude": 77.2090,
                },
            },
        }
        update = TelegramUpdate(**update_data)
        location = extract_location(update)

        assert location is not None
        assert location[0] == 28.6139
        assert location[1] == 77.2090

    def test_returns_none_without_location(self):
        """Test returns None when no location."""
        from src.webhook.handler import extract_location

        update_data = {
            "update_id": 123456,
            "message": {
                "message_id": 1,
                "chat": {"id": 12345, "type": "private"},
                "from": {"id": 67890, "first_name": "Test"},
                "text": "Hello",
            },
        }
        update = TelegramUpdate(**update_data)
        location = extract_location(update)

        assert location is None
