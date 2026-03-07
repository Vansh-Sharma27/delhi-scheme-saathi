"""Central orchestration layer for all live LLM usage."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from time import perf_counter
from typing import Any, TypeVar

from src.config import get_settings
from src.integrations.llm_client import (
    FallbackLLMClient,
    ProviderExecutionResult,
    get_llm_client,
)
from src.models.scheme import SchemeMatch
from src.models.session import ConversationMemory, Session
from src.services.conversation_memory import build_working_memory, working_memory_payload

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AITaskType(StrEnum):
    """Task classes for policy and telemetry."""

    ANALYZE_MESSAGE = "analyze_message"
    JUDGE_SCHEME_RELEVANCE = "judge_scheme_relevance"
    GENERATE_RESPONSE = "generate_response"
    REFRESH_WORKING_MEMORY = "refresh_working_memory"


@dataclass(frozen=True, slots=True)
class AIExecutionPolicy:
    """Execution controls for a specific AI task."""

    timeout_seconds: float
    priority: str


@dataclass(frozen=True, slots=True)
class LLMUsageEvent:
    """Structured telemetry for one orchestrated LLM task."""

    task_type: str
    session_id: str | None
    provider: str | None
    fallback_used: bool
    latency_ms: float
    prompt_chars: int
    queue_lag_ms: float | None = None
    error: str | None = None


class AIOrchestrator:
    """Apply task policy, timeouts, and telemetry to shared LLM usage."""

    _POLICIES = {
        AITaskType.ANALYZE_MESSAGE: AIExecutionPolicy(timeout_seconds=8.0, priority="inline"),
        AITaskType.JUDGE_SCHEME_RELEVANCE: AIExecutionPolicy(timeout_seconds=3.0, priority="inline"),
        AITaskType.GENERATE_RESPONSE: AIExecutionPolicy(timeout_seconds=8.0, priority="inline"),
        AITaskType.REFRESH_WORKING_MEMORY: AIExecutionPolicy(
            timeout_seconds=20.0,
            priority="background",
        ),
    }

    def __init__(
        self,
        llm_client: FallbackLLMClient | None = None,
    ) -> None:
        self.settings = get_settings()
        self.llm_client = llm_client or get_llm_client()

    @staticmethod
    def _estimate_prompt_chars(*parts: object) -> int:
        """Approximate prompt size without relying on provider token APIs."""
        total = 0
        for part in parts:
            if part in (None, "", [], {}, ()):
                continue
            total += len(json.dumps(part, ensure_ascii=False, default=str))
        return total

    def _log_usage(
        self,
        *,
        task_type: AITaskType,
        session_id: str | None,
        prompt_chars: int,
        result: ProviderExecutionResult[Any],
        queue_lag_ms: float | None = None,
    ) -> None:
        """Emit a structured usage log for observability."""
        event = LLMUsageEvent(
            task_type=task_type.value,
            session_id=session_id,
            provider=result.provider,
            fallback_used=result.fallback_used,
            latency_ms=round(result.latency_ms, 2),
            prompt_chars=prompt_chars,
            queue_lag_ms=queue_lag_ms,
            error=result.error,
        )
        logger.info("llm_usage %s", json.dumps(asdict(event), ensure_ascii=False))

    def _get_instance_override(self, method_name: str) -> Callable[..., Any] | None:
        """Return an instance-level monkeypatch for backward-compatible tests."""
        return getattr(self.llm_client, "__dict__", {}).get(method_name)

    async def _call_public_override(
        self,
        method_name: str,
        **kwargs: Any,
    ) -> ProviderExecutionResult[Any]:
        """Execute an instance-level monkeypatched public LLM method."""
        override = self._get_instance_override(method_name)
        if override is None:
            raise RuntimeError(f"No override found for {method_name}")

        started = perf_counter()
        output = await override(**kwargs)
        return ProviderExecutionResult(
            output=output,
            provider=None,
            fallback_used=False,
            latency_ms=(perf_counter() - started) * 1000,
        )

    async def _run_task(
        self,
        *,
        task_type: AITaskType,
        session_id: str | None,
        prompt_chars: int,
        call: Callable[[str], asyncio.Future[ProviderExecutionResult[T]]]
        | Callable[[str], Any],
        safe_output: Callable[[], T],
        queue_lag_ms: float | None = None,
    ) -> T:
        """Run one orchestrated LLM task under policy controls."""
        policy = self._POLICIES[task_type]
        try:
            async with asyncio.timeout(policy.timeout_seconds):
                result = await call(policy.priority)
        except TimeoutError:
            result = ProviderExecutionResult(
                output=safe_output(),
                provider=None,
                fallback_used=False,
                latency_ms=policy.timeout_seconds * 1000,
                error="timeout",
            )

        self._log_usage(
            task_type=task_type,
            session_id=session_id,
            prompt_chars=prompt_chars,
            result=result,
            queue_lag_ms=queue_lag_ms,
        )
        return result.output

    def should_run_relevance_judge(self, matches: list[SchemeMatch]) -> bool:
        """Only invoke LLM judging when deterministic ranking is ambiguous."""
        if not matches:
            return False

        top_score = matches[0].deterministic_score
        second_score = matches[1].deterministic_score if len(matches) > 1 else 0.0
        score_gap = top_score - second_score

        return (
            top_score <= self.settings.ai_relevance_min_deterministic_score
            or score_gap < self.settings.ai_relevance_score_gap_threshold
        )

    async def analyze_message(
        self,
        *,
        session: Session,
        user_message: str,
        conversation_history: list[dict[str, str]],
        system_prompt: str,
        session_language: str,
    ) -> dict[str, Any]:
        """Analyze the user's message with continuity-aware context."""
        user_profile = {
            **session.user_profile.model_dump(),
            "_currently_asking": session.currently_asking,
        }
        memory = working_memory_payload(session)
        prompt_chars = self._estimate_prompt_chars(
            user_message,
            conversation_history[-10:],
            user_profile,
            memory,
        )

        return await self._run_task(
            task_type=AITaskType.ANALYZE_MESSAGE,
            session_id=session.user_id,
            prompt_chars=prompt_chars,
            call=(
                lambda priority: self._call_public_override(
                    "analyze_message",
                    user_message=user_message,
                    conversation_history=conversation_history,
                    current_state=session.state.value,
                    user_profile=user_profile,
                    system_prompt=system_prompt,
                    session_language=session_language,
                    working_memory=memory,
                    priority=priority,
                )
                if self._get_instance_override("analyze_message") is not None
                else self.llm_client.analyze_message_with_meta(
                    user_message=user_message,
                    conversation_history=conversation_history,
                    current_state=session.state.value,
                    user_profile=user_profile,
                    system_prompt=system_prompt,
                    session_language=session_language,
                    working_memory=memory,
                    priority=priority,
                )
            ),
            safe_output=lambda: FallbackLLMClient._safe_analysis_payload(session_language),
        )

    async def judge_scheme_relevance(
        self,
        *,
        session: Session,
        user_message: str,
        conversation_history: list[dict[str, str]],
        candidate_schemes: list[dict[str, Any]],
        session_language: str,
    ) -> dict[str, Any]:
        """LLM gate for ambiguous deterministic scheme matches."""
        memory = working_memory_payload(session)
        prompt_chars = self._estimate_prompt_chars(
            user_message,
            conversation_history[-8:],
            session.user_profile.model_dump(),
            candidate_schemes,
            memory,
        )

        return await self._run_task(
            task_type=AITaskType.JUDGE_SCHEME_RELEVANCE,
            session_id=session.user_id,
            prompt_chars=prompt_chars,
            call=(
                lambda priority: self._call_public_override(
                    "judge_scheme_relevance",
                    user_message=user_message,
                    conversation_history=conversation_history,
                    current_state=session.state.value,
                    user_profile=session.user_profile.model_dump(),
                    candidate_schemes=candidate_schemes,
                    session_language=session_language,
                    working_memory=memory,
                    priority=priority,
                )
                if self._get_instance_override("judge_scheme_relevance") is not None
                else self.llm_client.judge_scheme_relevance_with_meta(
                    user_message=user_message,
                    conversation_history=conversation_history,
                    current_state=session.state.value,
                    user_profile=session.user_profile.model_dump(),
                    candidate_schemes=candidate_schemes,
                    session_language=session_language,
                    working_memory=memory,
                    priority=priority,
                )
            ),
            safe_output=lambda: FallbackLLMClient._safe_relevance_payload(candidate_schemes),
        )

    async def generate_response(
        self,
        *,
        session: Session,
        context: dict[str, Any],
        system_prompt: str,
        user_language: str,
    ) -> str:
        """Generate a free-form response with working memory attached."""
        enriched_context = dict(context)
        memory = working_memory_payload(session)
        if memory is not None:
            enriched_context["working_memory"] = memory

        prompt_chars = self._estimate_prompt_chars(
            enriched_context,
            system_prompt,
            user_language,
        )

        return await self._run_task(
            task_type=AITaskType.GENERATE_RESPONSE,
            session_id=session.user_id,
            prompt_chars=prompt_chars,
            call=(
                lambda priority: self._call_public_override(
                    "generate_response",
                    context=enriched_context,
                    system_prompt=system_prompt,
                    user_language=user_language,
                    priority=priority,
                )
                if self._get_instance_override("generate_response") is not None
                else self.llm_client.generate_response_with_meta(
                    context=enriched_context,
                    system_prompt=system_prompt,
                    user_language=user_language,
                    priority=priority,
                )
            ),
            safe_output=lambda: FallbackLLMClient._safe_generation_text(user_language),
        )

    async def refresh_working_memory(
        self,
        session: Session,
        *,
        queue_lag_ms: float | None = None,
    ) -> ConversationMemory:
        """Refresh compact working memory from recent turns and summary state."""
        if not session.messages:
            return build_working_memory(session, session.working_memory.summary)

        messages = [{"role": message.role, "content": message.content} for message in session.messages]
        prompt_chars = self._estimate_prompt_chars(messages, session.working_memory.summary)

        summary = await self._run_task(
            task_type=AITaskType.REFRESH_WORKING_MEMORY,
            session_id=session.user_id,
            prompt_chars=prompt_chars,
            queue_lag_ms=queue_lag_ms,
            call=(
                lambda priority: self._call_public_override(
                    "summarize_conversation",
                    messages=messages,
                    current_summary=session.working_memory.summary,
                    priority=priority,
                )
                if self._get_instance_override("summarize_conversation") is not None
                else self.llm_client.summarize_conversation_with_meta(
                    messages=messages,
                    current_summary=session.working_memory.summary,
                    priority=priority,
                )
            ),
            safe_output=lambda: session.working_memory.summary or "",
        )

        refreshed = build_working_memory(session, summary or session.working_memory.summary)
        if refreshed == session.working_memory:
            return session.working_memory
        return refreshed


_ai_orchestrator: AIOrchestrator | None = None


def configure_ai_orchestrator(orchestrator: AIOrchestrator | None) -> None:
    """Override the shared AI orchestrator instance."""
    global _ai_orchestrator
    _ai_orchestrator = orchestrator


def get_ai_orchestrator() -> AIOrchestrator:
    """Return the shared AI orchestrator singleton."""
    global _ai_orchestrator
    if _ai_orchestrator is None:
        _ai_orchestrator = AIOrchestrator()
    return _ai_orchestrator
