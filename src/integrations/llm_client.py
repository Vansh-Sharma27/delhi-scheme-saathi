"""LLM client with Bedrock Nova 2 Lite primary + Grok fallback.

Strategy:
1. Try Bedrock Nova 2 Lite first (AWS native, good multilingual)
2. Fall back to Grok on Bedrock failure (rate limit, API error, timeout)
3. Return error response if both fail

This ensures reliable LLM operations for conversation understanding
and response generation across different deployment environments.
"""

import logging
from typing import Any, Protocol

from src.config import get_settings

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def analyze_message(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]],
        current_state: str,
        user_profile: dict[str, Any],
        system_prompt: str,
        session_language: str = "hi",
    ) -> dict[str, Any]: ...

    async def generate_response(
        self,
        context: dict[str, Any],
        system_prompt: str,
        user_language: str,
    ) -> str: ...

    async def summarize_conversation(
        self,
        messages: list[dict[str, str]],
        current_summary: str | None,
    ) -> str: ...


class FallbackLLMClient:
    """LLM client with Bedrock primary + Grok fallback.

    Provides resilient LLM operations for Delhi Scheme Saathi:
    - Primary: AWS Bedrock Nova 2 Lite (when use_bedrock=True and AWS configured)
    - Fallback: xAI Grok via OpenAI-compatible API (always available)
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._use_bedrock = settings.use_bedrock
        self._use_grok = bool(settings.xai_api_key)
        self._bedrock_client = None
        self._grok_client = None

    def _get_grok_client(self):
        """Lazy init Grok client."""
        if self._grok_client is None:
            from src.integrations.grok_client import GrokLLMClient
            self._grok_client = GrokLLMClient()
        return self._grok_client

    def _get_bedrock_client(self):
        """Lazy init Bedrock client."""
        if self._bedrock_client is None:
            from src.integrations.bedrock_client import BedrockLLMClient
            self._bedrock_client = BedrockLLMClient()
        return self._bedrock_client

    async def analyze_message(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]],
        current_state: str,
        user_profile: dict[str, Any],
        system_prompt: str,
        session_language: str = "hi",
    ) -> dict[str, Any]:
        """Analyze message with automatic LLM fallback.

        Returns:
            dict with intent, life_event, extracted_fields, language, etc.
            On complete failure, returns error response with safe defaults.
        """
        # Try Bedrock first if enabled
        if self._use_bedrock:
            try:
                bedrock = self._get_bedrock_client()
                result = await bedrock.analyze_message(
                    user_message, conversation_history,
                    current_state, user_profile, system_prompt,
                    session_language,
                )
                logger.debug("Used Bedrock for message analysis")
                return result
            except Exception as e:
                logger.warning(f"Bedrock LLM failed, trying Grok: {e}")

        # Fallback to Grok
        if self._use_grok:
            try:
                grok = self._get_grok_client()
                result = await grok.analyze_message(
                    user_message, conversation_history,
                    current_state, user_profile, system_prompt,
                    session_language,
                )
                logger.debug("Used Grok for message analysis")
                return result
            except Exception as e:
                logger.error(f"Grok LLM also failed: {e}")

        # Both failed - return error response with safe defaults
        logger.error("All LLM providers failed for message analysis")
        return {
            "intent": "unknown",
            "life_event": None,
            "extracted_fields": {},
            "language": session_language,
            "selected_scheme_id": None,
            "action": None,
            "needs_clarification": True,
            "clarification_question": None,
            "response_text": None,
            "error": "LLM service unavailable",
        }

    async def generate_response(
        self,
        context: dict[str, Any],
        system_prompt: str,
        user_language: str = "hi",
    ) -> str:
        """Generate natural language response with fallback.

        Returns:
            Generated response text, or error message if all providers fail.
        """
        # Try Bedrock first if enabled
        if self._use_bedrock:
            try:
                bedrock = self._get_bedrock_client()
                result = await bedrock.generate_response(
                    context, system_prompt, user_language
                )
                logger.debug("Used Bedrock for response generation")
                return result
            except Exception as e:
                logger.warning(f"Bedrock response generation failed, trying Grok: {e}")

        # Fallback to Grok
        if self._use_grok:
            try:
                grok = self._get_grok_client()
                result = await grok.generate_response(
                    context, system_prompt, user_language
                )
                logger.debug("Used Grok for response generation")
                return result
            except Exception as e:
                logger.error(f"Grok response generation also failed: {e}")

        # Both failed - return error message
        logger.error("All LLM providers failed for response generation")
        if user_language == "hi":
            return "माफ़ कीजिए, कुछ तकनीकी समस्या है। कृपया थोड़ी देर बाद प्रयास करें।"
        return "Sorry, there is a technical issue. Please try again later."

    async def summarize_conversation(
        self,
        messages: list[dict[str, str]],
        current_summary: str | None = None,
    ) -> str:
        """Summarize conversation with fallback.

        Returns:
            Updated summary, or current summary if all providers fail.
        """
        # Try Bedrock first if enabled
        if self._use_bedrock:
            try:
                bedrock = self._get_bedrock_client()
                result = await bedrock.summarize_conversation(messages, current_summary)
                return result
            except Exception as e:
                logger.warning(f"Bedrock summarization failed, trying Grok: {e}")

        # Fallback to Grok
        if self._use_grok:
            try:
                grok = self._get_grok_client()
                result = await grok.summarize_conversation(messages, current_summary)
                return result
            except Exception as e:
                logger.error(f"Grok summarization also failed: {e}")

        # Both failed - return current summary
        return current_summary or ""


# Backward-compatible alias
class LLMClient(FallbackLLMClient):
    """Alias for FallbackLLMClient (backward compatibility)."""
    pass


# Singleton instance
_llm_client: FallbackLLMClient | None = None


def get_llm_client() -> FallbackLLMClient:
    """Get or create LLM client singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = FallbackLLMClient()
    return _llm_client
