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
                "data": "select_scheme:SCH-001",
            },
        }
        update = TelegramUpdate(**update_data)
        assert update.is_callback
        assert update.callback_query.get("data") == "select_scheme:SCH-001"


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
            mock_voice_client.speech_to_text.assert_called_once()

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
        """Test sending voice response when TTS available."""
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

        with patch(
            "src.webhook.handler._get_voice_client", return_value=mock_voice_client
        ):
            await _send_response(mock_telegram, "12345", mock_response, is_voice=True)

            # Should send both text and voice
            mock_telegram.send_text.assert_called_once()
            mock_telegram.send_voice.assert_called_once_with("12345", b"audio data")


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

    def test_limits_length(self):
        """Test text is limited to 500 chars."""
        from src.webhook.handler import _clean_for_tts

        text = "a" * 1000
        result = _clean_for_tts(text)
        assert len(result) <= 500


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
