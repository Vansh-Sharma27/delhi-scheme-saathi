"""Bhashini API client for speech-to-text and text-to-speech.

Bhashini is India's sovereign AI platform for language processing.
This client handles:
- Speech-to-Text (ASR) for Hindi voice messages
- Text-to-Speech (TTS) for Hindi audio responses
- Language detection and translation (optional)

API Documentation: https://bhashini.gov.in/ulca/documentation
"""

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


class BhashiniClient:
    """Client for Bhashini speech services.

    Uses the Dhruva API for ASR (speech-to-text) and TTS (text-to-speech).
    """

    # Dhruva API endpoints
    BASE_URL = "https://dhruva-api.bhashini.gov.in"
    INFERENCE_URL = f"{BASE_URL}/services/inference"
    PIPELINE_URL = f"{BASE_URL}/services/pipeline"

    # Model IDs for different languages and tasks
    ASR_MODELS = {
        "hi": "ai4bharat/conformer-hi-gpu--t4",
        "en": "ai4bharat/conformer-en-gpu--t4",
    }

    TTS_MODELS = {
        "hi": "ai4bharat/indic-tts-coqui-indo_female-gpu--t4",
        "en": "ai4bharat/indic-tts-coqui-eng_female-gpu--t4",
    }

    def __init__(
        self,
        api_key: str | None = None,
        user_id: str | None = None,
        ulca_api_key: str | None = None,
    ):
        """Initialize Bhashini client.

        Args:
            api_key: Bhashini API key (or BHASHINI_API_KEY env var)
            user_id: Bhashini user ID (or BHASHINI_USER_ID env var)
            ulca_api_key: ULCA API key for pipeline access (or BHASHINI_ULCA_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("BHASHINI_API_KEY", "")
        self.user_id = user_id or os.environ.get("BHASHINI_USER_ID", "")
        self.ulca_api_key = ulca_api_key or os.environ.get("BHASHINI_ULCA_API_KEY", "")

        self._http_client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "Authorization": self.api_key,
                    "Content-Type": "application/json",
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
    ) -> STTResult:
        """Convert speech to text using Bhashini ASR.

        Args:
            audio_bytes: Audio data (OGG, WAV, or MP3)
            source_lang: Source language code (hi, en)
            audio_format: Audio format (ogg, wav, mp3)

        Returns:
            STTResult with transcribed text and confidence
        """
        import base64

        if not self.api_key:
            logger.warning("Bhashini API key not configured, returning placeholder")
            return STTResult(
                text="[Voice message - please type your query]",
                confidence=0.0,
                language=source_lang,
            )

        # Encode audio as base64
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Get appropriate model
        model_id = self.ASR_MODELS.get(source_lang, self.ASR_MODELS["hi"])

        payload = {
            "pipelineTasks": [
                {
                    "taskType": "asr",
                    "config": {
                        "language": {
                            "sourceLanguage": source_lang,
                        },
                        "serviceId": model_id,
                        "audioFormat": audio_format,
                        "samplingRate": 16000,
                    },
                }
            ],
            "inputData": {
                "audio": [
                    {
                        "audioContent": audio_base64,
                    }
                ]
            },
        }

        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.PIPELINE_URL}",
                json=payload,
            )
            response.raise_for_status()

            data = response.json()
            logger.debug(f"Bhashini ASR response: {data}")

            # Extract transcription from response
            if "pipelineResponse" in data:
                asr_output = data["pipelineResponse"][0].get("output", [])
                if asr_output and len(asr_output) > 0:
                    text = asr_output[0].get("source", "")
                    # Bhashini doesn't always return confidence, default to 0.8
                    confidence = asr_output[0].get("confidence", 0.8)
                    return STTResult(
                        text=text,
                        confidence=confidence,
                        language=source_lang,
                    )

            logger.warning("No transcription in Bhashini response")
            return STTResult(
                text="",
                confidence=0.0,
                language=source_lang,
            )

        except httpx.HTTPError as e:
            logger.error(f"Bhashini ASR request failed: {e}")
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
        """Convert text to speech using Bhashini TTS.

        Args:
            text: Text to convert to speech
            target_lang: Target language code (hi, en)
            voice: Voice type (female, male)

        Returns:
            TTSResult with audio bytes
        """
        import base64

        if not self.api_key:
            logger.warning("Bhashini API key not configured")
            return TTSResult(audio_bytes=b"", content_type="audio/wav")

        # Get appropriate model
        model_id = self.TTS_MODELS.get(target_lang, self.TTS_MODELS["hi"])

        payload = {
            "pipelineTasks": [
                {
                    "taskType": "tts",
                    "config": {
                        "language": {
                            "sourceLanguage": target_lang,
                        },
                        "serviceId": model_id,
                        "gender": voice,
                        "samplingRate": 22050,
                    },
                }
            ],
            "inputData": {
                "input": [
                    {
                        "source": text,
                    }
                ]
            },
        }

        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.PIPELINE_URL}",
                json=payload,
            )
            response.raise_for_status()

            data = response.json()
            logger.debug(f"Bhashini TTS response received")

            # Extract audio from response
            if "pipelineResponse" in data:
                tts_output = data["pipelineResponse"][0].get("audio", [])
                if tts_output and len(tts_output) > 0:
                    audio_base64 = tts_output[0].get("audioContent", "")
                    if audio_base64:
                        audio_bytes = base64.b64decode(audio_base64)
                        return TTSResult(
                            audio_bytes=audio_bytes,
                            content_type="audio/wav",
                        )

            logger.warning("No audio in Bhashini TTS response")
            return TTSResult(audio_bytes=b"", content_type="audio/wav")

        except httpx.HTTPError as e:
            logger.error(f"Bhashini TTS request failed: {e}")
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
_bhashini_client: BhashiniClient | None = None


def get_bhashini_client() -> BhashiniClient:
    """Get singleton Bhashini client instance."""
    global _bhashini_client
    if _bhashini_client is None:
        _bhashini_client = BhashiniClient()
    return _bhashini_client


def configure_bhashini_client(
    api_key: str | None = None,
    user_id: str | None = None,
    ulca_api_key: str | None = None,
) -> BhashiniClient:
    """Configure and return Bhashini client."""
    global _bhashini_client
    _bhashini_client = BhashiniClient(
        api_key=api_key,
        user_id=user_id,
        ulca_api_key=ulca_api_key,
    )
    return _bhashini_client
