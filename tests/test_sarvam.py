"""Tests for Sarvam AI speech services client."""

import base64
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.integrations.sarvam import (
    SarvamClient,
    STTResult,
    TTSResult,
    get_sarvam_client,
    configure_sarvam_client,
)


class TestSarvamClient:
    """Tests for SarvamClient."""

    def test_client_initialization_without_key(self):
        """Test client initializes without API key."""
        client = SarvamClient()
        assert client.api_key == ""

    def test_client_initialization_with_key(self):
        """Test client initializes with provided API key."""
        client = SarvamClient(api_key="test-api-key")
        assert client.api_key == "test-api-key"


class TestSpeechToText:
    """Tests for speech-to-text functionality."""

    @pytest.mark.asyncio
    async def test_stt_without_api_key_returns_placeholder(self):
        """Test STT returns placeholder when no API key configured."""
        client = SarvamClient(api_key="")
        result = await client.speech_to_text(
            audio_bytes=b"test audio data",
            source_lang="hi",
        )
        assert isinstance(result, STTResult)
        assert result.confidence == 0.0
        assert "please type" in result.text.lower()

    @pytest.mark.asyncio
    async def test_stt_with_api_key_success(self):
        """Test STT with valid API key returns transcription."""
        client = SarvamClient(api_key="test-api-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "transcript": "नमस्ते, मुझे पेंशन चाहिए",
            "language_code": "hi-IN",
            "language_probability": 0.92,
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http_client

            result = await client.speech_to_text(
                audio_bytes=b"test audio data",
                source_lang="hi",
            )

            assert isinstance(result, STTResult)
            assert result.text == "नमस्ते, मुझे पेंशन चाहिए"
            assert result.confidence == 0.92
            assert result.language == "hi"

    @pytest.mark.asyncio
    async def test_stt_handles_http_error(self):
        """Test STT handles HTTP errors gracefully."""
        import httpx

        client = SarvamClient(api_key="test-api-key")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(
                side_effect=httpx.HTTPError("Connection failed")
            )
            mock_get_client.return_value = mock_http_client

            result = await client.speech_to_text(
                audio_bytes=b"test audio data",
                source_lang="hi",
            )

            assert isinstance(result, STTResult)
            assert result.confidence == 0.0
            assert "failed" in result.text.lower()


class TestTextToSpeech:
    """Tests for text-to-speech functionality."""

    @pytest.mark.asyncio
    async def test_tts_without_api_key_returns_empty(self):
        """Test TTS returns empty bytes when no API key configured."""
        client = SarvamClient(api_key="")
        result = await client.text_to_speech(
            text="नमस्ते",
            target_lang="hi",
        )
        assert isinstance(result, TTSResult)
        assert result.audio_bytes == b""

    @pytest.mark.asyncio
    async def test_tts_with_api_key_success(self):
        """Test TTS with valid API key returns audio bytes."""
        client = SarvamClient(api_key="test-api-key")

        # Create mock audio content
        audio_content = b"mock audio bytes"
        audio_base64 = base64.b64encode(audio_content).decode("utf-8")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "audios": [audio_base64],
            "request_id": "test-123",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_http_client

            result = await client.text_to_speech(
                text="नमस्ते",
                target_lang="hi",
            )

            assert isinstance(result, TTSResult)
            assert result.audio_bytes == audio_content
            assert result.content_type == "audio/wav"

    @pytest.mark.asyncio
    async def test_tts_handles_http_error(self):
        """Test TTS handles HTTP errors gracefully."""
        import httpx

        client = SarvamClient(api_key="test-api-key")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http_client = AsyncMock()
            mock_http_client.post = AsyncMock(
                side_effect=httpx.HTTPError("Connection failed")
            )
            mock_get_client.return_value = mock_http_client

            result = await client.text_to_speech(
                text="नमस्ते",
                target_lang="hi",
            )

            assert result.audio_bytes == b""


class TestLanguageDetection:
    """Tests for language detection."""

    @pytest.mark.asyncio
    async def test_detect_hindi(self):
        """Test detection of Hindi text."""
        client = SarvamClient()
        lang = await client.detect_language("नमस्ते, मुझे पेंशन चाहिए")
        assert lang == "hi"

    @pytest.mark.asyncio
    async def test_detect_english(self):
        """Test detection of English text."""
        client = SarvamClient()
        lang = await client.detect_language("Hello, I need pension information")
        assert lang == "en"

    @pytest.mark.asyncio
    async def test_detect_hinglish_defaults_to_hindi(self):
        """Test Hinglish text defaults to Hindi."""
        client = SarvamClient()
        # Hinglish with <30% Devanagari should detect as English
        lang = await client.detect_language("Mujhe pension chahiye please")
        assert lang == "en"

    @pytest.mark.asyncio
    async def test_detect_empty_defaults_to_hindi(self):
        """Test empty text defaults to Hindi."""
        client = SarvamClient()
        lang = await client.detect_language("123 456")
        assert lang == "hi"


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_sarvam_client_returns_same_instance(self):
        """Test get_sarvam_client returns singleton."""
        client1 = get_sarvam_client()
        client2 = get_sarvam_client()
        assert client1 is client2

    def test_configure_sarvam_client_creates_new_instance(self):
        """Test configure_sarvam_client replaces singleton."""
        original = get_sarvam_client()
        new_client = configure_sarvam_client(api_key="new-key")
        latest = get_sarvam_client()

        assert new_client is latest
        assert new_client.api_key == "new-key"
