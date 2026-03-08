"""LLM client with Bedrock Nova 2 Lite primary + Grok fallback."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Generic, Literal, Protocol, TypeVar

from src.config import get_settings

logger = logging.getLogger(__name__)

TaskPriority = Literal["inline", "background"]
T = TypeVar("T")


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
        working_memory: dict[str, Any] | None = None,
        priority: TaskPriority = "inline",
    ) -> dict[str, Any]: ...

    async def generate_response(
        self,
        context: dict[str, Any],
        system_prompt: str,
        user_language: str,
        priority: TaskPriority = "inline",
    ) -> str: ...

    async def summarize_conversation(
        self,
        messages: list[dict[str, str]],
        current_summary: str | None,
        priority: TaskPriority = "background",
    ) -> str: ...

    async def judge_scheme_relevance(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]],
        current_state: str,
        user_profile: dict[str, Any],
        candidate_schemes: list[dict[str, Any]],
        session_language: str = "hi",
        working_memory: dict[str, Any] | None = None,
        priority: TaskPriority = "inline",
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ProviderExecutionResult(Generic[T]):
    """Result of a provider execution with metadata for observability."""

    output: T
    provider: str | None
    fallback_used: bool
    latency_ms: float
    error: str | None = None


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

    async def _execute_with_fallback(
        self,
        *,
        task_name: str,
        bedrock_call: Callable[[], Awaitable[T]],
        grok_call: Callable[[], Awaitable[T]],
        safe_output: Callable[[], T],
    ) -> ProviderExecutionResult[T]:
        """Execute a task with Bedrock primary and Grok fallback."""
        started = perf_counter()
        bedrock_attempted = False

        if self._use_bedrock:
            bedrock_attempted = True
            try:
                result = await bedrock_call()
                return ProviderExecutionResult(
                    output=result,
                    provider="bedrock",
                    fallback_used=False,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            except Exception as exc:
                logger.warning("Bedrock %s failed, trying Grok: %s", task_name, exc)

        if self._use_grok:
            try:
                result = await grok_call()
                return ProviderExecutionResult(
                    output=result,
                    provider="grok",
                    fallback_used=bedrock_attempted,
                    latency_ms=(perf_counter() - started) * 1000,
                )
            except Exception as exc:
                logger.error("Grok %s also failed: %s", task_name, exc)

        error = "LLM service unavailable"
        logger.error("All LLM providers failed for %s", task_name)
        return ProviderExecutionResult(
            output=safe_output(),
            provider=None,
            fallback_used=bedrock_attempted,
            latency_ms=(perf_counter() - started) * 1000,
            error=error,
        )

    @staticmethod
    def _safe_analysis_payload(session_language: str) -> dict[str, Any]:
        """Return a user-safe analysis fallback payload."""
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

    @staticmethod
    def _safe_generation_text(user_language: str) -> str:
        """Return a user-safe generation fallback string."""
        if user_language == "hi":
            return "माफ़ कीजिए, कुछ तकनीकी समस्या है। कृपया थोड़ी देर बाद प्रयास करें।"
        return "Sorry, there is a technical issue. Please try again later."

    @staticmethod
    def _safe_relevance_payload(
        candidate_schemes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return deterministic-safe relevance fallback payload."""
        return {
            "should_clarify": False,
            "clarification_question": None,
            "overall_confidence": 0.5,
            "candidate_scores": [
                {
                    "scheme_id": candidate.get("scheme_id"),
                    "relevance_score": candidate.get("deterministic_score", 0.5),
                    "topic_match": None,
                    "reason": None,
                }
                for candidate in candidate_schemes
            ],
            "error": "LLM service unavailable",
        }

    async def analyze_message_with_meta(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]],
        current_state: str,
        user_profile: dict[str, Any],
        system_prompt: str,
        session_language: str = "hi",
        working_memory: dict[str, Any] | None = None,
        priority: TaskPriority = "inline",
    ) -> ProviderExecutionResult[dict[str, Any]]:
        """Analyze message with provider metadata."""
        return await self._execute_with_fallback(
            task_name="message analysis",
            bedrock_call=lambda: self._get_bedrock_client().analyze_message(
                user_message=user_message,
                conversation_history=conversation_history,
                current_state=current_state,
                user_profile=user_profile,
                system_prompt=system_prompt,
                session_language=session_language,
                working_memory=working_memory,
                priority=priority,
            ),
            grok_call=lambda: self._get_grok_client().analyze_message(
                user_message=user_message,
                conversation_history=conversation_history,
                current_state=current_state,
                user_profile=user_profile,
                system_prompt=system_prompt,
                session_language=session_language,
                working_memory=working_memory,
                priority=priority,
            ),
            safe_output=lambda: self._safe_analysis_payload(session_language),
        )

    async def analyze_message(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]],
        current_state: str,
        user_profile: dict[str, Any],
        system_prompt: str,
        session_language: str = "hi",
        working_memory: dict[str, Any] | None = None,
        priority: TaskPriority = "inline",
    ) -> dict[str, Any]:
        """Analyze message with automatic fallback."""
        result = await self.analyze_message_with_meta(
            user_message=user_message,
            conversation_history=conversation_history,
            current_state=current_state,
            user_profile=user_profile,
            system_prompt=system_prompt,
            session_language=session_language,
            working_memory=working_memory,
            priority=priority,
        )
        return result.output

    async def generate_response_with_meta(
        self,
        context: dict[str, Any],
        system_prompt: str,
        user_language: str = "hi",
        priority: TaskPriority = "inline",
    ) -> ProviderExecutionResult[str]:
        """Generate response with provider metadata."""
        return await self._execute_with_fallback(
            task_name="response generation",
            bedrock_call=lambda: self._get_bedrock_client().generate_response(
                context=context,
                system_prompt=system_prompt,
                user_language=user_language,
                priority=priority,
            ),
            grok_call=lambda: self._get_grok_client().generate_response(
                context=context,
                system_prompt=system_prompt,
                user_language=user_language,
                priority=priority,
            ),
            safe_output=lambda: self._safe_generation_text(user_language),
        )

    async def generate_response(
        self,
        context: dict[str, Any],
        system_prompt: str,
        user_language: str = "hi",
        priority: TaskPriority = "inline",
    ) -> str:
        """Generate natural language response with fallback."""
        result = await self.generate_response_with_meta(
            context=context,
            system_prompt=system_prompt,
            user_language=user_language,
            priority=priority,
        )
        return result.output

    async def summarize_conversation_with_meta(
        self,
        messages: list[dict[str, str]],
        current_summary: str | None = None,
        priority: TaskPriority = "background",
    ) -> ProviderExecutionResult[str]:
        """Summarize conversation with provider metadata."""
        return await self._execute_with_fallback(
            task_name="conversation summarization",
            bedrock_call=lambda: self._get_bedrock_client().summarize_conversation(
                messages=messages,
                current_summary=current_summary,
                priority=priority,
            ),
            grok_call=lambda: self._get_grok_client().summarize_conversation(
                messages=messages,
                current_summary=current_summary,
                priority=priority,
            ),
            safe_output=lambda: current_summary or "",
        )

    async def summarize_conversation(
        self,
        messages: list[dict[str, str]],
        current_summary: str | None = None,
        priority: TaskPriority = "background",
    ) -> str:
        """Summarize conversation with fallback."""
        result = await self.summarize_conversation_with_meta(
            messages=messages,
            current_summary=current_summary,
            priority=priority,
        )
        return result.output

    async def judge_scheme_relevance_with_meta(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]],
        current_state: str,
        user_profile: dict[str, Any],
        candidate_schemes: list[dict[str, Any]],
        session_language: str = "hi",
        working_memory: dict[str, Any] | None = None,
        priority: TaskPriority = "inline",
    ) -> ProviderExecutionResult[dict[str, Any]]:
        """Judge scheme relevance with provider metadata."""
        return await self._execute_with_fallback(
            task_name="scheme relevance judging",
            bedrock_call=lambda: self._get_bedrock_client().judge_scheme_relevance(
                user_message=user_message,
                conversation_history=conversation_history,
                current_state=current_state,
                user_profile=user_profile,
                candidate_schemes=candidate_schemes,
                session_language=session_language,
                working_memory=working_memory,
                priority=priority,
            ),
            grok_call=lambda: self._get_grok_client().judge_scheme_relevance(
                user_message=user_message,
                conversation_history=conversation_history,
                current_state=current_state,
                user_profile=user_profile,
                candidate_schemes=candidate_schemes,
                session_language=session_language,
                working_memory=working_memory,
                priority=priority,
            ),
            safe_output=lambda: self._safe_relevance_payload(candidate_schemes),
        )

    async def judge_scheme_relevance(
        self,
        user_message: str,
        conversation_history: list[dict[str, str]],
        current_state: str,
        user_profile: dict[str, Any],
        candidate_schemes: list[dict[str, Any]],
        session_language: str = "hi",
        working_memory: dict[str, Any] | None = None,
        priority: TaskPriority = "inline",
    ) -> dict[str, Any]:
        """Judge deterministic candidates with automatic fallback."""
        result = await self.judge_scheme_relevance_with_meta(
            user_message=user_message,
            conversation_history=conversation_history,
            current_state=current_state,
            user_profile=user_profile,
            candidate_schemes=candidate_schemes,
            session_language=session_language,
            working_memory=working_memory,
            priority=priority,
        )
        return result.output


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
