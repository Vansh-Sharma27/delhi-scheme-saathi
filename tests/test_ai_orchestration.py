"""Tests for the AI orchestration and working-memory layer."""

from unittest.mock import AsyncMock

import pytest

from src.integrations.llm_client import ProviderExecutionResult
from src.models.scheme import EligibilityCriteria, Scheme, SchemeMatch
from src.models.session import ConversationMemory, Message, Session, UserProfile
from src.services.ai_orchestrator import AIOrchestrator


def _make_match(scheme_id: str, deterministic_score: float) -> SchemeMatch:
    """Create a minimal scheme match for gating tests."""
    return SchemeMatch(
        scheme=Scheme(
            id=scheme_id,
            name=f"Scheme {scheme_id}",
            name_hindi=f"योजना {scheme_id}",
            department="Dept",
            department_hindi="विभाग",
            level="state",
            eligibility=EligibilityCriteria(),
            life_events=["HOUSING"],
            description="Test scheme",
            description_hindi="परीक्षण योजना",
            documents_required=[],
        ),
        deterministic_score=deterministic_score,
    )


@pytest.mark.asyncio
async def test_generate_response_includes_working_memory_context() -> None:
    """Free-form response generation should inject working memory into the prompt context."""
    fake_llm = AsyncMock()
    fake_llm.generate_response_with_meta = AsyncMock(
        return_value=ProviderExecutionResult(
            output="Generated response",
            provider="bedrock",
            fallback_used=False,
            latency_ms=12.0,
        )
    )
    orchestrator = AIOrchestrator(llm_client=fake_llm)
    session = Session(
        user_id="user-memory-context",
        working_memory=ConversationMemory(
            summary="User asked about housing help.",
            profile_facts=["Need area: HOUSING"],
        ),
    )

    response = await orchestrator.generate_response(
        session=session,
        context={"response_mode": "scheme_question_answer"},
        system_prompt="test",
        user_language="en",
    )

    assert response == "Generated response"
    context = fake_llm.generate_response_with_meta.await_args.kwargs["context"]
    assert context["working_memory"]["summary"] == "User asked about housing help."
    assert "Need area: HOUSING" in context["working_memory"]["profile_facts"]


@pytest.mark.asyncio
async def test_refresh_working_memory_builds_summary_and_scheme_context() -> None:
    """Background memory refresh should combine LLM summary with deterministic facts."""
    fake_llm = AsyncMock()
    fake_llm.summarize_conversation_with_meta = AsyncMock(
        return_value=ProviderExecutionResult(
            output="User wants housing support and needs the next eligibility step.",
            provider="bedrock",
            fallback_used=False,
            latency_ms=18.0,
        )
    )
    orchestrator = AIOrchestrator(llm_client=fake_llm)
    session = Session(
        user_id="user-refresh-memory",
        user_profile=UserProfile(life_event="HOUSING", annual_income=300000),
        selected_scheme_id="SCH-1",
        discussed_schemes=["SCH-1", "SCH-2"],
        messages=[
            Message(role="user", content="I need housing help."),
            Message(role="assistant", content="Please share your annual family income."),
        ],
    )

    memory = await orchestrator.refresh_working_memory(session)

    assert memory.summary == "User wants housing support and needs the next eligibility step."
    assert "Need area: HOUSING" in memory.profile_facts
    assert "Annual income: ₹3 lakh" in memory.profile_facts
    assert memory.active_scheme_ids[0] == "SCH-1"


def test_should_run_relevance_judge_only_for_ambiguous_matches() -> None:
    """AI relevance judging should be reserved for ambiguous deterministic rankings."""
    orchestrator = AIOrchestrator(llm_client=AsyncMock())

    assert orchestrator.should_run_relevance_judge(
        [_make_match("SCH-CLEAR", 0.97), _make_match("SCH-LOW", 0.60)]
    ) is False
    assert orchestrator.should_run_relevance_judge(
        [_make_match("SCH-A", 0.84), _make_match("SCH-B", 0.80)]
    ) is True
