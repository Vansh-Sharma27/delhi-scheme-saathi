"""Tests for LLM and embedding fallback behavior."""

import json

import pytest

from src.config import get_settings
from src.integrations import bedrock_client, embedding_client, grok_client, llm_client
from src.models.session import UserProfile
from src.prompts.loader import get_analysis_system_prompt, get_system_prompt
from src.services import scheme_matcher


@pytest.fixture(autouse=True)
def _reset_singletons() -> None:
    """Reset cached/singleton state between tests."""
    get_settings.cache_clear()
    llm_client._llm_client = None
    embedding_client._embedding_client = None


@pytest.mark.asyncio
async def test_llm_falls_back_from_bedrock_to_grok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bedrock failure should use Grok as fallback."""
    monkeypatch.setenv("USE_BEDROCK", "true")
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    get_settings.cache_clear()

    calls: list[str] = []

    class FakeBedrock:
        async def analyze_message(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            calls.append("bedrock")
            raise RuntimeError("bedrock unavailable")

    class FakeGrok:
        async def analyze_message(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            calls.append("grok")
            return {
                "intent": "question",
                "life_event": None,
                "extracted_fields": {},
                "language": "hi",
                "selected_scheme_id": None,
                "needs_clarification": False,
                "clarification_question": None,
            }

    monkeypatch.setattr(bedrock_client, "BedrockLLMClient", FakeBedrock)
    monkeypatch.setattr(grok_client, "GrokLLMClient", FakeGrok)

    client = llm_client.FallbackLLMClient()
    result = await client.analyze_message(
        user_message="मुझे घर चाहिए",
        conversation_history=[],
        current_state="SITUATION_UNDERSTANDING",
        user_profile={},
        system_prompt="test",
    )

    assert calls == ["bedrock", "grok"]
    assert result["intent"] == "question"


@pytest.mark.asyncio
async def test_llm_returns_safe_defaults_when_all_providers_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Bedrock and Grok both fail, wrapper should return safe fallback payload."""
    monkeypatch.setenv("USE_BEDROCK", "true")
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    get_settings.cache_clear()

    class AlwaysFailBedrock:
        async def analyze_message(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("bedrock unavailable")

    class AlwaysFailGrok:
        async def analyze_message(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("grok unavailable")

    monkeypatch.setattr(bedrock_client, "BedrockLLMClient", AlwaysFailBedrock)
    monkeypatch.setattr(grok_client, "GrokLLMClient", AlwaysFailGrok)

    client = llm_client.FallbackLLMClient()
    result = await client.analyze_message(
        user_message="hello",
        conversation_history=[],
        current_state="GREETING",
        user_profile={},
        system_prompt="test",
    )

    assert result["intent"] == "unknown"
    assert result["needs_clarification"] is True
    assert result["error"] == "LLM service unavailable"


@pytest.mark.asyncio
async def test_llm_generate_response_safe_fallback_when_all_providers_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both providers fail for response generation, wrapper returns user-safe message."""
    monkeypatch.setenv("USE_BEDROCK", "true")
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    get_settings.cache_clear()

    class AlwaysFailBedrock:
        async def generate_response(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("bedrock unavailable")

    class AlwaysFailGrok:
        async def generate_response(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("grok unavailable")

    monkeypatch.setattr(bedrock_client, "BedrockLLMClient", AlwaysFailBedrock)
    monkeypatch.setattr(grok_client, "GrokLLMClient", AlwaysFailGrok)

    client = llm_client.FallbackLLMClient()
    result = await client.generate_response(
        context={},
        system_prompt="test",
        user_language="hi",
    )

    assert "तकनीकी समस्या" in result


@pytest.mark.asyncio
async def test_llm_summarize_returns_current_summary_when_all_providers_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both providers fail for summarization, wrapper keeps current summary."""
    monkeypatch.setenv("USE_BEDROCK", "true")
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    get_settings.cache_clear()

    class AlwaysFailBedrock:
        async def summarize_conversation(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("bedrock unavailable")

    class AlwaysFailGrok:
        async def summarize_conversation(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("grok unavailable")

    monkeypatch.setattr(bedrock_client, "BedrockLLMClient", AlwaysFailBedrock)
    monkeypatch.setattr(grok_client, "GrokLLMClient", AlwaysFailGrok)

    client = llm_client.FallbackLLMClient()
    result = await client.summarize_conversation(
        messages=[{"role": "user", "content": "hello"}],
        current_summary="existing summary",
    )

    assert result == "existing summary"


@pytest.mark.asyncio
async def test_llm_relevance_judge_returns_safe_defaults_when_all_providers_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relevance judging should degrade safely when both providers fail."""
    monkeypatch.setenv("USE_BEDROCK", "true")
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    get_settings.cache_clear()

    class AlwaysFailBedrock:
        async def judge_scheme_relevance(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("bedrock unavailable")

    class AlwaysFailGrok:
        async def judge_scheme_relevance(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("grok unavailable")

    monkeypatch.setattr(bedrock_client, "BedrockLLMClient", AlwaysFailBedrock)
    monkeypatch.setattr(grok_client, "GrokLLMClient", AlwaysFailGrok)

    client = llm_client.FallbackLLMClient()
    result = await client.judge_scheme_relevance(
        user_message="I need housing assistance",
        conversation_history=[],
        current_state="SCHEME_MATCHING",
        user_profile={"life_event": "HOUSING"},
        candidate_schemes=[{"scheme_id": "SCH-1", "deterministic_score": 0.8}],
        session_language="en",
    )

    assert result["should_clarify"] is False
    assert result["candidate_scores"][0]["scheme_id"] == "SCH-1"
    assert result["candidate_scores"][0]["relevance_score"] == 0.8


@pytest.mark.asyncio
async def test_embedding_falls_back_from_jina_to_voyage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Jina failure should use Voyage embedding as fallback."""
    monkeypatch.setenv("JINA_API_KEY", "jina-test")
    monkeypatch.setenv("VOYAGE_API_KEY", "voyage-test")
    get_settings.cache_clear()

    calls: list[str] = []
    client = embedding_client.FallbackEmbeddingClient()

    async def fail_jina(text: str) -> list[float]:
        calls.append("jina")
        raise RuntimeError("jina unavailable")

    async def ok_voyage(text: str) -> list[float]:
        calls.append("voyage")
        return [0.1] * embedding_client.EMBEDDING_DIM

    monkeypatch.setattr(client, "_jina_embedding", fail_jina)
    monkeypatch.setattr(client, "_voyage_embedding", ok_voyage)

    vector = await client.get_embedding("housing support")
    assert calls == ["jina", "voyage"]
    assert vector is not None
    assert len(vector) == embedding_client.EMBEDDING_DIM


@pytest.mark.asyncio
async def test_embedding_none_when_all_providers_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedding wrapper should return None instead of zero vector when both providers fail."""
    monkeypatch.setenv("JINA_API_KEY", "jina-test")
    monkeypatch.setenv("VOYAGE_API_KEY", "voyage-test")
    get_settings.cache_clear()

    client = embedding_client.FallbackEmbeddingClient()

    async def fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(client, "_jina_embedding", fail)
    monkeypatch.setattr(client, "_voyage_embedding", fail)

    assert await client.get_embedding("test") is None


@pytest.mark.asyncio
async def test_matcher_skips_vector_ranking_on_failed_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheme matcher should pass None embedding to DB layer when providers fail."""

    class FailedEmbeddingClient:
        async def get_embedding(self, text: str) -> None:
            return None

    captured: dict[str, object] = {}

    async def fake_hybrid_search(*, query_embedding, **kwargs):  # type: ignore[no-untyped-def]
        captured["query_embedding"] = query_embedding
        return []

    monkeypatch.setattr(scheme_matcher, "get_embedding_client", lambda: FailedEmbeddingClient())
    monkeypatch.setattr(scheme_matcher, "hybrid_search", fake_hybrid_search)

    profile = UserProfile(life_event="HOUSING")
    matches = await scheme_matcher.match_schemes(
        pool=object(),  # type: ignore[arg-type]
        profile=profile,
        query_text="मुझे घर चाहिए",
    )

    assert matches == []
    assert captured["query_embedding"] is None


@pytest.mark.asyncio
async def test_grok_generate_response_handles_datetime_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grok prompt building should serialize datetime values in context safely."""
    import datetime as dt

    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    get_settings.cache_clear()

    class FakeCompletions:
        async def create(self, **kwargs):  # type: ignore[no-untyped-def]
            class Msg:
                content = "ok"

            class Choice:
                message = Msg()

            class Resp:
                choices = [Choice()]

            return Resp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAIClient:
        chat = FakeChat()

    client = grok_client.GrokLLMClient()
    monkeypatch.setattr(client, "_client", FakeOpenAIClient())

    text = await client.generate_response(
        context={"created_at": dt.datetime(2026, 3, 3, 0, 0, 0)},
        system_prompt="test",
        user_language="en",
    )
    assert text == "ok"


@pytest.mark.asyncio
async def test_bedrock_generate_response_handles_datetime_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bedrock prompt building should serialize datetime values in context safely."""
    import asyncio
    import datetime as dt

    monkeypatch.setenv("USE_BEDROCK", "true")
    get_settings.cache_clear()

    client = bedrock_client.BedrockLLMClient()

    class FakeBedrockRuntime:
        def converse(self, **kwargs):  # type: ignore[no-untyped-def]
            return {
                "output": {
                    "message": {
                        "content": [{"text": "ok"}]
                    }
                }
            }

    monkeypatch.setattr(client, "_client", FakeBedrockRuntime())

    # Avoid running actual threadpool execution in test.
    loop = asyncio.get_event_loop()
    monkeypatch.setattr(loop, "run_in_executor", lambda *args, **kwargs: asyncio.sleep(0, result=args[1]()))

    text = await client.generate_response(
        context={"created_at": dt.datetime(2026, 3, 3, 0, 0, 0)},
        system_prompt="test",
        user_language="en",
    )
    assert text == "ok"


@pytest.mark.asyncio
async def test_grok_analysis_prompt_allows_direct_entailed_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grok analysis prompt should allow direct semantic entailment without stereotypes."""
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    get_settings.cache_clear()

    captured: dict[str, object] = {}

    class FakeCompletions:
        async def create(self, **kwargs):  # type: ignore[no-untyped-def]
            captured["messages"] = kwargs["messages"]
            captured["temperature"] = kwargs["temperature"]

            class Msg:
                content = json.dumps(
                    {
                        "intent": "question",
                        "action": "answer_field",
                        "life_event": "DEATH_IN_FAMILY",
                        "extracted_fields": {"gender": "female", "marital_status": "widowed"},
                        "language": "en",
                        "selected_scheme_id": None,
                        "needs_clarification": False,
                        "clarification_question": None,
                        "response_text": "What is the applicant/beneficiary age?",
                    }
                )

            class Choice:
                message = Msg()

            class Resp:
                choices = [Choice()]

            return Resp()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAIClient:
        chat = FakeChat()

    client = grok_client.GrokLLMClient()
    monkeypatch.setattr(client, "_client", FakeOpenAIClient())

    await client.analyze_message(
        user_message="My husband died in an accident.",
        conversation_history=[],
        current_state="GREETING",
        user_profile={},
        system_prompt="test",
        session_language="en",
    )

    prompt = captured["messages"][-1]["content"]
    assert captured["temperature"] == 0
    assert "directly entailed by the user's own first-person wording" in prompt
    assert "Do NOT use weak stereotypes" in prompt
    assert "first-person spousal relationship terms" in prompt
    assert "gendered and therefore logically determines the user's schema gender" in prompt
    assert "gendered spouse terminology" in prompt
    assert "spouse-loss wording" in prompt
    assert "do not ask for that same field again" in prompt


@pytest.mark.asyncio
async def test_bedrock_analysis_prompt_allows_direct_entailed_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bedrock analysis prompt should allow direct semantic entailment without stereotypes."""
    import asyncio

    monkeypatch.setenv("USE_BEDROCK", "true")
    get_settings.cache_clear()

    captured: dict[str, object] = {}
    client = bedrock_client.BedrockLLMClient()

    class FakeBedrockRuntime:
        def converse(self, **kwargs):  # type: ignore[no-untyped-def]
            captured["messages"] = kwargs["messages"]
            captured["inference_config"] = kwargs["inferenceConfig"]
            return {
                "output": {
                    "message": {
                        "content": [
                            {
                                "text": json.dumps(
                                    {
                                        "intent": "question",
                                        "action": "answer_field",
                                        "life_event": "DEATH_IN_FAMILY",
                                        "extracted_fields": {
                                            "gender": "female",
                                            "marital_status": "widowed",
                                        },
                                        "language": "en",
                                        "selected_scheme_id": None,
                                        "needs_clarification": False,
                                        "clarification_question": None,
                                        "response_text": "What is the applicant/beneficiary age?",
                                    }
                                )
                            }
                        ]
                    }
                }
            }

    monkeypatch.setattr(client, "_client", FakeBedrockRuntime())

    loop = asyncio.get_event_loop()
    monkeypatch.setattr(
        loop,
        "run_in_executor",
        lambda *args, **kwargs: asyncio.sleep(0, result=args[1]()),
    )

    await client.analyze_message(
        user_message="Mere pati ki maut ho gayi hai.",
        conversation_history=[],
        current_state="GREETING",
        user_profile={},
        system_prompt="test",
        session_language="en",
    )

    prompt = captured["messages"][-1]["content"][0]["text"]
    assert captured["inference_config"]["temperature"] == 0
    assert "directly entailed by the user's own first-person wording" in prompt
    assert "Do NOT use weak stereotypes" in prompt
    assert "first-person spousal relationship terms" in prompt
    assert "gendered and therefore logically determines the user's schema gender" in prompt
    assert "gendered spouse terminology" in prompt
    assert "spouse-loss wording" in prompt
    assert "do not ask for that same field again" in prompt


def test_system_prompt_alias_uses_analysis_prompt() -> None:
    """Legacy loader name should resolve to the analysis-only prompt."""
    prompt = get_analysis_system_prompt()

    assert get_system_prompt() == prompt
    assert "You are a structured analysis engine." in prompt
    assert "Prioritize accurate JSON extraction over" in prompt
    assert "Do not suppress direct entailments as guesses." in prompt
    assert "spouse" in prompt
    assert "marital_status and gender" in prompt
