"""Sarvam AI client for speech-to-text and text-to-speech.

Sarvam AI provides high-quality Indian language AI services.
This client handles:
- Speech-to-Text (ASR) for Hindi and other Indian languages
- Text-to-Speech (TTS) with natural voices

API Documentation: https://docs.sarvam.ai/
"""

import base64
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class STTResult:
    """Speech-to-text result."""
    text: str
    confidence: float
    language: str = "hi"


@dataclass
class TTSResult:
    """Text-to-speech result."""
    audio_bytes: bytes
    content_type: str = "audio/wav"
    duration_seconds: float = 0.0


class SarvamClient:
    """Client for Sarvam AI speech services.

    Uses Sarvam AI API for ASR (speech-to-text) and TTS (text-to-speech).
    """

    # Sarvam AI API endpoints
    BASE_URL = "https://api.sarvam.ai"
    STT_URL = f"{BASE_URL}/speech-to-text"
    TTS_URL = f"{BASE_URL}/text-to-speech"

    # Model IDs
    STT_MODEL = "saaras:v3"    # Saaras v3 - latest STT model (23 languages)
    TTS_MODEL = "bulbul:v3"    # Default TTS model (30+ voices)

    # Default speakers by language (for bulbul:v3) - must be lowercase
    DEFAULT_SPEAKERS = {
        "hi": "ritu",     # Hindi female voice (bulbul:v3)
        "en": "amelia",   # English female voice (bulbul:v3)
    }
    DEFAULT_MALE_SPEAKER = "shubh"  # Default male voice (bulbul:v3)

    # Language code mappings (internal -> Sarvam BCP-47)
    LANGUAGE_CODES = {
        "hi": "hi-IN",
        "en": "en-IN",
        "bn": "bn-IN",
        "ta": "ta-IN",
        "te": "te-IN",
        "mr": "mr-IN",
        "gu": "gu-IN",
        "kn": "kn-IN",
        "ml": "ml-IN",
        "pa": "pa-IN",
        "od": "od-IN",
    }

    def __init__(self, api_key: str | None = None):
        """Initialize Sarvam AI client.

        Args:
            api_key: Sarvam AI API subscription key (or SARVAM_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY", "")
        self._http_client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "api-subscription-key": self.api_key,
                },
            )
        return self._http_client

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def speech_to_text(
        self,
        audio_bytes: bytes,
        source_lang: str = "hi",
        audio_format: str = "ogg",
        mode: str = "transcribe",
    ) -> STTResult:
        """Convert speech to text using Sarvam AI ASR (Saaras v3).

        Args:
            audio_bytes: Audio data (OGG, WAV, MP3, etc.)
            source_lang: Source language code (hi, en, etc.)
            audio_format: Audio format (ogg, wav, mp3)
            mode: Transcription mode (transcribe, translate, verbatim, translit, codemix)

        Returns:
            STTResult with transcribed text and confidence
        """
        if not self.api_key:
            logger.warning("Sarvam API key not configured, returning placeholder")
            return STTResult(
                text="[Voice message - please type your query]",
                confidence=0.0,
                language=source_lang,
            )

        # Map language code to BCP-47
        language_code = self.LANGUAGE_CODES.get(source_lang, "hi-IN")

        # Prepare multipart form data
        # Determine appropriate MIME type
        mime_types = {
            "ogg": "audio/ogg",
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "m4a": "audio/mp4",
            "webm": "audio/webm",
        }
        content_type = mime_types.get(audio_format, "audio/ogg")

        files = {
            "file": (f"audio.{audio_format}", audio_bytes, content_type),
        }
        data = {
            "model": self.STT_MODEL,
            "mode": mode,  # Saaras v3 mode: transcribe, translate, etc.
            "language_code": language_code,
        }

        try:
            client = await self._get_client()
            response = await client.post(
                self.STT_URL,
                files=files,
                data=data,
            )
            response.raise_for_status()

            result = response.json()
            logger.debug(f"Sarvam STT response: {result}")

            transcript = result.get("transcript", "")
            confidence = result.get("transcript_confidence")
            if confidence is None:
                confidence = result.get("confidence")
            if confidence is None:
                confidence = 1.0 if transcript else 0.0

            detected_language = (
                result.get("language_code")
                or result.get("detected_language")
                or source_lang
            )
            if isinstance(detected_language, str):
                detected_language = detected_language.split("-")[0].lower()
            else:
                detected_language = source_lang

            return STTResult(
                text=transcript,
                confidence=float(confidence) if isinstance(confidence, (int, float)) else 0.8,
                language=detected_language,
            )

        except httpx.HTTPError as e:
            logger.error(f"Sarvam STT request failed: {e}")
            return STTResult(
                text="[Voice recognition failed - please type your query]",
                confidence=0.0,
                language=source_lang,
            )

    async def text_to_speech(
        self,
        text: str,
        target_lang: str = "hi",
        voice: str = "female",
    ) -> TTSResult:
        """Convert text to speech using Sarvam AI TTS.

        Args:
            text: Text to convert to speech (max 2500 chars for bulbul:v3)
            target_lang: Target language code (hi, en, etc.)
            voice: Voice type (female, male) or specific speaker name

        Returns:
            TTSResult with audio bytes
        """
        if not self.api_key:
            logger.warning("Sarvam API key not configured")
            return TTSResult(audio_bytes=b"", content_type="audio/wav")

        # Map language code to BCP-47
        language_code = self.LANGUAGE_CODES.get(target_lang, "hi-IN")

        # Select speaker based on voice parameter
        if voice in ["female", "male"]:
            # Use default speaker for language
            speaker = self.DEFAULT_SPEAKERS.get(target_lang, "ritu")
            if voice == "male":
                speaker = self.DEFAULT_MALE_SPEAKER
        else:
            # Use specific speaker name (lowercase)
            speaker = voice.lower()

        # Truncate text if too long (bulbul:v3 limit is 2500)
        text = text[:2500]

        payload = {
            "text": text,
            "target_language_code": language_code,
            "model": self.TTS_MODEL,
            "speaker": speaker,
            "speech_sample_rate": 22050,  # Good balance of quality and size
        }

        try:
            client = await self._get_client()
            response = await client.post(
                self.TTS_URL,
                json=payload,
            )
            response.raise_for_status()

            result = response.json()
            logger.debug("Sarvam TTS response received")

            # Extract audio from response
            audios = result.get("audios", [])
            if audios and len(audios) > 0:
                audio_base64 = audios[0]
                if audio_base64:
                    audio_bytes = base64.b64decode(audio_base64)
                    return TTSResult(
                        audio_bytes=audio_bytes,
                        content_type="audio/wav",
                    )

            logger.warning("No audio in Sarvam TTS response")
            return TTSResult(audio_bytes=b"", content_type="audio/wav")

        except httpx.HTTPError as e:
            logger.error(f"Sarvam TTS request failed: {e}")
            return TTSResult(audio_bytes=b"", content_type="audio/wav")

    async def detect_language(self, text: str) -> str:
        """Detect language of text.

        Simple heuristic-based detection for Hindi vs English.
        """
        # Count Devanagari characters
        devanagari_count = sum(1 for c in text if "\u0900" <= c <= "\u097F")
        total_alpha = sum(1 for c in text if c.isalpha())

        if total_alpha == 0:
            return "hi"  # Default to Hindi

        devanagari_ratio = devanagari_count / total_alpha

        if devanagari_ratio > 0.3:
            return "hi"
        return "en"


# Singleton instance
_sarvam_client: SarvamClient | None = None


def get_sarvam_client() -> SarvamClient:
    """Get singleton Sarvam client instance."""
    global _sarvam_client
    if _sarvam_client is None:
        _sarvam_client = SarvamClient()
    return _sarvam_client


def configure_sarvam_client(api_key: str | None = None) -> SarvamClient:
    """Configure and return Sarvam client."""
    global _sarvam_client
    _sarvam_client = SarvamClient(api_key=api_key)
    return _sarvam_client
