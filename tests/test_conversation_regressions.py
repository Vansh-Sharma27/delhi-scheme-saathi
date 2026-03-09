"""Regression tests for multi-turn conversation bugs seen in Telegram exports."""
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.db.session_store import InMemorySessionStore, configure_session_store, get_session_store
from src.models.api import ChatRequest
from src.models.document import Document, DocumentChain
from src.models.rejection_rule import RejectionRule
from src.models.scheme import EligibilityCriteria, Scheme, SchemeMatch
from src.models.session import ConversationState, Message, Session, UserProfile
from src.services import response_generator
from src.services.conversation import (
    ConversationService,
    _infer_text_language,
    _resolve_scheme_from_text,
    _truncate_at_sentence,
)
from src.services.life_event_classifier import classify_by_keywords

ACTIVE_SCHEME_SEEDS = [
    {
        "id": scheme["id"],
        "name": scheme["name"],
        "name_hindi": scheme.get("name_hindi", scheme["name"]),
    }
    for scheme in json.loads(
        (Path(__file__).resolve().parents[1] / "data" / "all_schemes.json").read_text()
    )
    if scheme.get("is_active", True)
]


def _make_scheme(
    scheme_id: str,
    *,
    language: str = "en",
    life_event: str = "EDUCATION",
    name: str | None = None,
    name_hindi: str | None = None,
    income_by_category: dict[str, int] | None = None,
    genders: list[str] | None = None,
    categories: list[str] | None = None,
) -> Scheme:
    """Create a simple scheme object for plain-text rendering tests."""
    return Scheme(
        id=scheme_id,
        name=name or f"Scheme {scheme_id}",
        name_hindi=name_hindi or f"योजना {scheme_id}",
        department="Test Department",
        department_hindi="परीक्षण विभाग",
        level="state",
        benefits_amount=250000,
        benefits_frequency="one-time",
        eligibility=EligibilityCriteria(
            min_age=18,
            max_age=None,
            max_income=500000,
            genders=genders or ["all"],
            categories=categories or ["OBC", "General"],
            income_by_category=income_by_category or {},
        ),
        life_events=[life_event],
        description="A test scheme description for verification.",
        description_hindi="सत्यापन के लिए परीक्षण योजना विवरण।",
        documents_required=[],
        application_url="https://example.com/apply",
        offline_process="Visit the district office",
    )


@pytest.fixture(autouse=True)
def reset_session_store() -> None:
    """Use a fresh in-memory session store for each regression test."""
    configure_session_store(InMemorySessionStore())


@pytest.mark.asyncio
async def test_explicit_language_lock_persists_across_turns_and_start() -> None:
    """Explicit language choice should stay sticky across later turns and /start."""
    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        side_effect=[
            {
                "intent": "question",
                "action": "change_language",
                "life_event": None,
                "extracted_fields": {},
                "language": "en",
                "selected_scheme_id": None,
                "response_text": "Please tell me what kind of assistance you need today.",
            },
            {
                "intent": "question",
                "action": "answer_field",
                "life_event": "HOUSING",
                "extracted_fields": {},
                "language": "hi",
                "selected_scheme_id": None,
                "response_text": None,
            },
        ]
    )

    first = await service.handle_message(
        ChatRequest(user_id="user-lock", message="Please use english only")
    )
    second = await service.handle_message(
        ChatRequest(user_id="user-lock", message="मुझे housing help चाहिए")
    )
    third = await service.handle_message(
        ChatRequest(user_id="user-lock", message="/start")
    )

    session = await get_session_store().get("user-lock")
    assert first.language == "en"
    assert second.language == "en"
    assert "age" in second.text.lower()
    assert third.language == "en"
    assert session is not None
    assert session.language_preference == "en"
    assert session.language_locked is True


@pytest.mark.asyncio
async def test_explicit_language_request_does_not_fall_into_farewell_when_llm_mislabels_goodbye() -> None:
    """Language-only requests should not end the conversation even if the LLM says goodbye."""
    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "goodbye",
            "action": "goodbye",
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": "Goodbye.",
        }
    )

    result = await service.handle_message(
        ChatRequest(user_id="user-language-not-farewell", message="Please use English only.")
    )

    session = await get_session_store().get("user-language-not-farewell")
    assert "Take care" not in result.text
    assert "assistance do you need" in result.text
    assert result.language == "en"
    assert session is not None
    assert session.language_preference == "en"
    assert session.language_locked is True


@pytest.mark.asyncio
async def test_help_command_returns_bilingual_guide_for_new_user_without_llm() -> None:
    """New users should get a deterministic bilingual help guide from /help."""
    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock()

    result = await service.handle_message(
        ChatRequest(user_id="user-help-new", message="/help")
    )

    session = await get_session_store().get("user-help-new")
    assert result.next_state == ConversationState.GREETING.value
    assert result.language == "en"
    assert "ENGLISH" in result.text
    assert "हिंदी" in result.text
    assert "/start" in result.text
    assert "/language" in result.text
    assert "start over" in result.text
    assert "bye" in result.text
    assert result.inline_keyboard is not None
    assert session is not None
    assert session.state == ConversationState.GREETING
    assert session.completed_turn_count == 1
    service.llm.analyze_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_language_command_returns_picker_without_llm() -> None:
    """The /language command should show inline language choices deterministically."""
    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock()

    result = await service.handle_message(
        ChatRequest(user_id="user-language-picker", message="/language")
    )

    session = await get_session_store().get("user-language-picker")
    assert result.next_state == ConversationState.GREETING.value
    assert result.language == "en"
    assert "Choose your preferred language" in result.text
    assert result.inline_keyboard is not None
    assert [row[0]["callback_data"] for row in result.inline_keyboard] == [
        "lang:hi",
        "lang:en",
        "lang:hinglish",
    ]
    assert session is not None
    assert session.state == ConversationState.GREETING
    service.llm.analyze_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_unlocked_previous_language_does_not_bias_english_turn() -> None:
    """Observed language should drive unlocked sessions on later turns."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-unlocked-language",
            state=ConversationState.PROFILE_COLLECTION,
            user_profile=UserProfile(life_event="HOUSING"),
            language_preference="hi",
            language_locked=False,
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "answer_field",
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": "What is the applicant/beneficiary age?",
        }
    )

    result = await service.handle_message(
        ChatRequest(
            user_id="user-unlocked-language",
            message="I need help with rent support",
        )
    )

    session = await store.get("user-unlocked-language")
    assert service.llm.analyze_message.await_args.kwargs["session_language"] == "en"
    assert result.language == "en"
    assert session is not None
    assert session.language_preference == "en"
    assert session.language_locked is False


@pytest.mark.asyncio
async def test_unlocked_hindi_bare_age_reply_preserves_language_context() -> None:
    """Bare-value field replies should keep the active Hindi language in unlocked sessions."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-unlocked-hi-bare-age",
            state=ConversationState.PROFILE_COLLECTION,
            user_profile=UserProfile(life_event="HOUSING"),
            messages=[
                Message(role="assistant", content="आवेदक (या लाभार्थी) की उम्र कितनी है?"),
            ],
            currently_asking="age",
            language_preference="hi",
            language_locked=False,
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "answer_field",
            "life_event": None,
            "extracted_fields": {"age": 35},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": "What is the applicant's caste category?",
        }
    )

    result = await service.handle_message(
        ChatRequest(user_id="user-unlocked-hi-bare-age", message="35")
    )

    session = await store.get("user-unlocked-hi-bare-age")
    assert service.llm.analyze_message.await_args.kwargs["session_language"] == "hi"
    assert result.language == "hi"
    assert "What is the applicant's" not in result.text
    assert any("\u0900" <= char <= "\u097F" for char in result.text)
    assert session is not None
    assert session.language_preference == "hi"
    assert session.language_locked is False


@pytest.mark.asyncio
async def test_locked_hinglish_rewrites_hindi_llm_reply_into_hinglish() -> None:
    """Locked Hinglish sessions should not leak Hindi-script LLM replies."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-hinglish-rewrite",
            state=ConversationState.SITUATION_UNDERSTANDING,
            language_preference="hinglish",
            language_locked=True,
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "answer_field",
            "life_event": None,
            "extracted_fields": {},
            "language": "hi",
            "selected_scheme_id": None,
            "response_text": "मैं आपकी मदद कर सकता हूं। अपनी उम्र बताइए।",
        }
    )
    mock_ai = AsyncMock()
    mock_ai.generate_response = AsyncMock(
        return_value="Main aapki madad kar sakta hoon. Apni age batayiye."
    )

    with patch(
        "src.services.response_generator.get_ai_orchestrator",
        return_value=mock_ai,
    ):
        result = await service.handle_message(
            ChatRequest(user_id="user-hinglish-rewrite", message="Mujhe help chahiye")
        )

    assert result.language == "hinglish"
    assert "Main aapki madad" in result.text
    assert "मैं" not in result.text
    assert mock_ai.generate_response.await_count == 1


@pytest.mark.asyncio
async def test_why_question_explains_and_reasks_same_field() -> None:
    """Why-style objections should explain the field instead of falling into a loop."""
    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        side_effect=[
            {
                "intent": "question",
                "action": "answer_field",
                "life_event": "HOUSING",
                "extracted_fields": {},
                "language": "en",
                "selected_scheme_id": None,
                "response_text": "I can help with housing. What is the applicant/beneficiary age?",
            },
            {
                "intent": "question",
                "action": "ask_field_reason",
                "life_event": None,
                "extracted_fields": {},
                "language": "en",
                "selected_scheme_id": None,
                "response_text": None,
            },
        ]
    )

    await service.handle_message(
        ChatRequest(user_id="user-why", message="I need housing help")
    )
    result = await service.handle_message(
        ChatRequest(user_id="user-why", message="What's the matter of age here?")
    )

    session = await get_session_store().get("user-why")
    assert "eligibility" in result.text.lower()
    assert "age" in result.text.lower()
    assert session is not None
    assert session.currently_asking == "age"


@pytest.mark.asyncio
async def test_skip_field_does_not_get_reasked_next_turn() -> None:
    """Skipped fields should remain skipped until the topic changes."""
    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        side_effect=[
            {
                "intent": "question",
                "action": "answer_field",
                "life_event": "HOUSING",
                "extracted_fields": {},
                "language": "en",
                "selected_scheme_id": None,
                "response_text": "What is the applicant/beneficiary age?",
            },
            {
                "intent": "question",
                "action": "skip_field",
                "life_event": None,
                "extracted_fields": {},
                "language": "en",
                "selected_scheme_id": None,
                "response_text": None,
            },
            {
                "intent": "question",
                "action": "answer_field",
                "life_event": None,
                "extracted_fields": {"category": "OBC"},
                "language": "en",
                "selected_scheme_id": None,
                "response_text": None,
            },
        ]
    )

    await service.handle_message(
        ChatRequest(user_id="user-skip", message="I need housing help")
    )
    skipped = await service.handle_message(
        ChatRequest(user_id="user-skip", message="I don't know")
    )
    after_category = await service.handle_message(
        ChatRequest(user_id="user-skip", message="OBC")
    )

    session = await get_session_store().get("user-skip")
    assert "income" in skipped.text.lower()
    assert "income" in after_category.text.lower()
    assert session is not None
    assert "age" in session.skipped_fields
    assert session.currently_asking == "annual_income"


@pytest.mark.asyncio
async def test_detail_language_change_stays_in_details() -> None:
    """Asking for details in another language should not jump to application flow."""
    store = get_session_store()
    seed = Session(
        user_id="user-details",
        state=ConversationState.SCHEME_DETAILS,
        user_profile=UserProfile(
            life_event="EDUCATION",
            age=19,
            category="OBC",
            annual_income=400000,
        ),
        selected_scheme_id="SCH-1",
        language_preference="en",
        language_locked=True,
    )
    await store.save(seed)

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "request_details",
            "life_event": None,
            "extracted_fields": {},
            "language": "hi",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )

    scheme = _make_scheme("SCH-1")
    mock_ai = AsyncMock()
    mock_ai.generate_response = AsyncMock(
        return_value=(
            "📋 योजना SCH-1\n\n"
            "सत्यापन के लिए परीक्षण योजना विवरण।\n\n"
            "💰 लाभ राशि: ₹2.5 लाख\n"
            "✅ पात्रता: आयु: 18-∞ | अधिकतम आय: ₹5 लाख\n\n"
            "अगला क्या देखें: दस्तावेज, अस्वीकृति चेतावनियाँ, या आवेदन प्रक्रिया?"
        )
    )
    with patch("src.services.conversation.scheme_repo.get_scheme_by_id", AsyncMock(return_value=scheme)), patch(
        "src.services.conversation.document_resolver.resolve_documents_for_scheme",
        AsyncMock(return_value=[]),
    ), patch(
        "src.services.conversation.rejection_engine.get_rejection_warnings",
        AsyncMock(return_value=[]),
    ), patch(
        "src.services.response_generator.get_ai_orchestrator",
        return_value=mock_ai,
    ):
        result = await service.handle_message(
            ChatRequest(
                user_id="user-details",
                message="Can you provide the scheme details in Hindi language?",
            )
        )

    session = await store.get("user-details")
    assert result.next_state == ConversationState.SCHEME_DETAILS.value
    assert result.language == "hi"
    assert "आवेदन" not in result.text.splitlines()[0]
    assert session is not None
    assert session.state == ConversationState.SCHEME_DETAILS
    assert session.language_preference == "hi"
    assert session.language_locked is True


@pytest.mark.asyncio
async def test_start_over_action_resets_state_but_keeps_locked_language() -> None:
    """Natural-language restart requests should fully reset the flow."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-restart",
            state=ConversationState.SCHEME_DETAILS,
            user_profile=UserProfile(
                life_event="EDUCATION",
                age=19,
                category="OBC",
                annual_income=400000,
            ),
            selected_scheme_id="SCH-1",
            presented_schemes=[
                {"id": "SCH-1", "name": "Scheme SCH-1", "name_hindi": "योजना SCH-1"},
            ],
            language_preference="en",
            language_locked=True,
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "start_over",
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )

    result = await service.handle_message(
        ChatRequest(user_id="user-restart", message="start over")
    )

    session = await store.get("user-restart")
    assert result.next_state == ConversationState.GREETING.value
    assert result.language == "en"
    assert session is not None
    assert session.user_profile.life_event is None
    assert session.selected_scheme_id is None
    assert session.presented_schemes == []
    assert session.language_preference == "en"
    assert session.language_locked is True


@pytest.mark.asyncio
async def test_help_command_preserves_active_scheme_context_and_language() -> None:
    """The /help guide should not clear the selected scheme or the locked language."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-help-context",
            state=ConversationState.SCHEME_DETAILS,
            user_profile=UserProfile(
                life_event="EDUCATION",
                age=19,
                category="OBC",
                annual_income=400000,
            ),
            selected_scheme_id="SCH-1",
            presented_schemes=[
                {"id": "SCH-1", "name": "Scheme SCH-1", "name_hindi": "योजना SCH-1"},
            ],
            language_preference="en",
            language_locked=True,
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock()

    result = await service.handle_message(
        ChatRequest(user_id="user-help-context", message="/help@DelhiSchemeSaathiBot")
    )

    session = await store.get("user-help-context")
    assert result.next_state == ConversationState.SCHEME_DETAILS.value
    assert result.language == "en"
    assert "Am I eligible?" in result.text
    assert "/language" in result.text
    assert "/start" in result.text
    assert result.inline_keyboard is not None
    assert result.inline_keyboard[1][0]["callback_data"] == "lang:en"
    assert session is not None
    assert session.state == ConversationState.SCHEME_DETAILS
    assert session.selected_scheme_id == "SCH-1"
    assert session.language_preference == "en"
    assert session.language_locked is True
    service.llm.analyze_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_language_callback_re_renders_active_scheme_in_selected_language() -> None:
    """Language callback should preserve the active scheme and rebuild the current view."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-language-callback",
            state=ConversationState.SCHEME_DETAILS,
            user_profile=UserProfile(
                life_event="EDUCATION",
                age=19,
                category="OBC",
                annual_income=400000,
            ),
            selected_scheme_id="SCH-1",
            language_preference="en",
            language_locked=True,
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock()
    scheme = _make_scheme("SCH-1", name_hindi="शिक्षा योजना")

    with patch(
        "src.services.conversation.scheme_repo.get_scheme_by_id",
        AsyncMock(return_value=scheme),
    ):
        result = await service.handle_message(
            ChatRequest(
                user_id="user-language-callback",
                message="lang:hi",
                message_type="callback",
                callback_data="lang:hi",
            )
        )

    session = await store.get("user-language-callback")
    assert result.next_state == ConversationState.SCHEME_DETAILS.value
    assert result.language == "hi"
    assert "भाषा बदल दी गई है" in result.text
    assert "शिक्षा योजना" in result.text
    assert session is not None
    assert session.state == ConversationState.SCHEME_DETAILS
    assert session.selected_scheme_id == "SCH-1"
    assert session.language_preference == "hi"
    assert session.language_locked is True
    service.llm.analyze_message.assert_not_awaited()


def test_scheme_resolution_avoids_numeric_false_positives() -> None:
    """Bare numbers inside profile answers should not select schemes."""
    session = Session(
        user_id="user-selection-guard",
        presented_schemes=[
            {"id": "SCH-1", "name": "Housing Relief Scheme", "name_hindi": "हाउसिंग राहत योजना"},
            {"id": "SCH-2", "name": "Rent Support Scheme", "name_hindi": "किराया सहायता योजना"},
        ],
    )

    assert _resolve_scheme_from_text(session, "my annual income is 210000") is None
    assert _resolve_scheme_from_text(session, "Can you share scheme details?") is None
    assert _resolve_scheme_from_text(session, "scheme 2") == "SCH-2"


def test_scheme_resolution_handles_natural_reference_to_secondary_candidate() -> None:
    """Natural-language follow-ups should resolve a specifically mentioned candidate."""
    session = Session(
        user_id="user-selection-reference",
        selected_scheme_id="SCH-1",
        presented_schemes=[
            {"id": "SCH-1", "name": "Pradhan Mantri Awas Yojana - Urban 2.0", "name_hindi": "प्रधानमंत्री आवास योजना"},
            {"id": "SCH-2", "name": "Education Loan Scheme - Delhi", "name_hindi": "शिक्षा ऋण योजना"},
        ],
    )

    assert _resolve_scheme_from_text(
        session,
        "May I know why you suggested education loan scheme when I was asking for housing schemes?",
    ) == "SCH-2"


@pytest.mark.asyncio
async def test_no_match_guard_does_not_rerun_matching_without_profile_change() -> None:
    """No-match recovery should not loop into matching again on filler replies."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-no-match-guard",
            state=ConversationState.PROFILE_COLLECTION,
            user_profile=UserProfile(
                life_event="HOUSING",
                age=30,
                category="OBC",
                annual_income=200000,
                gender="male",
            ),
            language_preference="en",
            awaiting_profile_change=True,
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "none",
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )
    match_schemes = AsyncMock(return_value=[])

    with patch("src.services.conversation.scheme_matcher.match_schemes", match_schemes):
        result = await service.handle_message(
            ChatRequest(user_id="user-no-match-guard", message="ok")
        )

    assert match_schemes.await_count == 0
    assert result.next_state == ConversationState.PROFILE_COLLECTION.value
    assert "matching your profile" in result.text.lower()


@pytest.mark.asyncio
async def test_details_profile_change_clears_selection_and_rematches() -> None:
    """Eligibility changes inside details should invalidate the old scheme."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-details-rematch",
            state=ConversationState.SCHEME_DETAILS,
            user_profile=UserProfile(
                life_event="EDUCATION",
                age=19,
                category="OBC",
                annual_income=400000,
            ),
            selected_scheme_id="SCH-1",
            language_preference="en",
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "answer_field",
            "life_event": None,
            "extracted_fields": {"annual_income": 100000},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )
    service.llm.judge_scheme_relevance = AsyncMock(
        return_value={
            "should_clarify": False,
            "clarification_question": None,
            "overall_confidence": 0.9,
            "candidate_scores": [
                {
                    "scheme_id": "SCH-2",
                    "relevance_score": 0.9,
                    "topic_match": True,
                    "reason": "Matches the updated eligibility profile.",
                }
            ],
        }
    )
    match_schemes = AsyncMock(
        return_value=[
            SchemeMatch(
                scheme=_make_scheme("SCH-2"),
                eligibility_match={"age": True, "income": True, "category": True},
            )
        ]
    )

    with patch("src.services.conversation.scheme_matcher.match_schemes", match_schemes), patch(
        "src.services.conversation.format_inline_keyboard",
        return_value=[[{"text": "Scheme SCH-2", "callback_data": "scheme:SCH-2"}]],
    ):
        result = await service.handle_message(
            ChatRequest(user_id="user-details-rematch", message="Income is 100000")
        )

    session = await store.get("user-details-rematch")
    assert match_schemes.await_count == 1
    assert result.next_state == ConversationState.SCHEME_PRESENTATION.value
    assert "SCH-2" in result.text
    assert session is not None
    assert session.selected_scheme_id is None
    assert session.presented_schemes[0]["id"] == "SCH-2"


def test_plain_english_scheme_message_stays_english() -> None:
    """The word 'scheme' alone should not force Hinglish auto-detection."""
    assert _infer_text_language("Can you explain this scheme in English?") == "en"


def test_generic_loan_keyword_does_not_force_education_life_event() -> None:
    """Loan alone is too broad to imply education."""
    assert classify_by_keywords("I need help with a loan") is None


@pytest.mark.asyncio
async def test_field_answer_turn_does_not_blindly_replace_known_life_event() -> None:
    """Income collection turns must not overwrite an already-known topic."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-voice-income",
            state=ConversationState.PROFILE_COLLECTION,
            user_profile=UserProfile(
                life_event="HOUSING",
                age=25,
                category="OBC",
            ),
            messages=[
                Message(role="assistant", content="What is the applicant's approximate annual family income?"),
            ],
            currently_asking="annual_income",
            language_preference="en",
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "answer_field",
            "life_event": "EDUCATION",
            "extracted_fields": {"annual_income": 500000},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )
    match_schemes = AsyncMock(return_value=[])

    with patch("src.services.conversation.scheme_matcher.match_schemes", match_schemes):
        result = await service.handle_message(
            ChatRequest(
                user_id="user-voice-income",
                message="it's roughly around 5 lakhs",
            )
        )

    session = await store.get("user-voice-income")
    called_profile = match_schemes.await_args.kwargs["profile"]
    conversation_history = service.llm.analyze_message.await_args.kwargs["conversation_history"]

    assert called_profile.life_event == "HOUSING"
    assert result.next_state == ConversationState.PROFILE_COLLECTION.value
    assert session is not None
    assert session.user_profile.life_event == "HOUSING"
    assert conversation_history[-1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_unparsed_income_answer_reasks_field_instead_of_following_llm_drift() -> None:
    """While collecting income, an unparsed answer should not let the LLM switch topics."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-income-reask",
            state=ConversationState.PROFILE_COLLECTION,
            user_profile=UserProfile(
                life_event="DEATH_IN_FAMILY",
                age=23,
                category="OBC",
            ),
            messages=[
                Message(role="assistant", content="What is the applicant's approximate annual family income?"),
            ],
            currently_asking="annual_income",
            language_preference="en",
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": None,
            "life_event": "EDUCATION",
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": "Are you looking for education loan assistance for yourself or someone else?",
        }
    )

    result = await service.handle_message(
        ChatRequest(user_id="user-income-reask", message="around ₹xx")
    )

    session = await store.get("user-income-reask")
    assert "income" in result.text.lower()
    assert "education loan" not in result.text.lower()
    assert session is not None
    assert session.currently_asking == "annual_income"


@pytest.mark.asyncio
async def test_field_help_question_stays_on_same_pending_field() -> None:
    """Field clarification questions should be answered without drifting off-topic."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-income-help",
            state=ConversationState.PROFILE_COLLECTION,
            user_profile=UserProfile(
                life_event="DEATH_IN_FAMILY",
                age=23,
                category="OBC",
            ),
            messages=[
                Message(role="assistant", content="What is the applicant's approximate annual family income?"),
            ],
            currently_asking="annual_income",
            language_preference="en",
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": None,
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": "Could you clarify that question?",
        }
    )

    result = await service.handle_message(
        ChatRequest(user_id="user-income-help", message="How should I estimate family income?")
    )

    session = await store.get("user-income-help")
    assert "monthly" in result.text.lower() or "annual" in result.text.lower()
    assert session is not None
    assert session.currently_asking == "annual_income"


@pytest.mark.asyncio
async def test_multi_beneficiary_followup_while_collecting_answers_scope_question() -> None:
    """Collection turns should answer applicant-scope questions instead of re-asking income."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-multi-beneficiary-followup",
            state=ConversationState.PROFILE_COLLECTION,
            user_profile=UserProfile(
                life_event="DEATH_IN_FAMILY",
                age=35,
                category="SC",
                gender="female",
                marital_status="widowed",
            ),
            messages=[
                Message(role="assistant", content="What is the applicant's approximate annual family income?"),
            ],
            currently_asking="annual_income",
            language_preference="en",
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": None,
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": "I can help both you and your daughter, but we should focus on one applicant first.",
        }
    )

    result = await service.handle_message(
        ChatRequest(
            user_id="user-multi-beneficiary-followup",
            message="Can you help both me and my daughter, or do I need to choose one person first?",
        )
    )

    session = await store.get("user-multi-beneficiary-followup")
    assert "one applicant" in result.text.lower()
    assert "income" not in result.text.lower()
    assert session is not None
    assert session.currently_asking == "life_event"


@pytest.mark.asyncio
async def test_widow_flow_skips_irrelevant_category_and_asks_income_next() -> None:
    """Widow-support collection should not insist on caste category before income."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-widow-income-next",
            state=ConversationState.PROFILE_COLLECTION,
            user_profile=UserProfile(
                life_event="DEATH_IN_FAMILY",
                gender="female",
                marital_status="widowed",
            ),
            messages=[
                Message(role="assistant", content="What is the applicant/beneficiary age?"),
            ],
            currently_asking="age",
            language_preference="en",
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "answer_field",
            "life_event": None,
            "extracted_fields": {"age": 23},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )

    result = await service.handle_message(
        ChatRequest(user_id="user-widow-income-next", message="23")
    )

    session = await store.get("user-widow-income-next")
    assert "income" in result.text.lower()
    assert "category" not in result.text.lower()
    assert session is not None
    assert session.currently_asking == "annual_income"


@pytest.mark.asyncio
async def test_new_widow_flow_adds_empathy_before_first_collection_question() -> None:
    """Death-in-family intake should prepend empathy before asking the next field."""
    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": None,
            "life_event": "DEATH_IN_FAMILY",
            "extracted_fields": {
                "gender": "female",
                "marital_status": "widowed",
            },
            "language": "hinglish",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )

    result = await service.handle_message(
        ChatRequest(
            user_id="user-new-widow-empathy",
            message="Mere pati ki maut ho gayi hai, mujhe widow pension ke baare mein batayiye.",
        )
    )

    session = await get_session_store().get("user-new-widow-empathy")
    assert "dukh" in result.text.lower() or "sorry" in result.text.lower()
    assert "age" in result.text.lower()
    assert session is not None
    assert session.user_profile.life_event == "DEATH_IN_FAMILY"


@pytest.mark.asyncio
async def test_llm_inferred_widow_context_does_not_reask_gender() -> None:
    """If the LLM infers widow context, the bot should move on to the next missing field."""
    store = get_session_store()
    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "answer_field",
            "life_event": "DEATH_IN_FAMILY",
            "extracted_fields": {
                "gender": "female",
                "marital_status": "widowed",
            },
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )

    result = await service.handle_message(
        ChatRequest(user_id="user-llm-widow-inference", message="My husband died in an accident.")
    )

    session = await store.get("user-llm-widow-inference")
    assert "age" in result.text.lower()
    assert "male or female" not in result.text.lower()
    assert session is not None
    assert session.user_profile.gender == "female"
    assert session.user_profile.marital_status == "widowed"
    assert session.currently_asking == "age"


@pytest.mark.asyncio
async def test_rule_based_spouse_loss_fallback_completes_gender_for_mixed_script_widow() -> None:
    """Mixed-script widow wording should not re-ask gender when the LLM misses it."""
    store = get_session_store()
    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "answer_field",
            "life_event": "DEATH_IN_FAMILY",
            "extracted_fields": {
                "marital_status": "widowed",
            },
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )

    result = await service.handle_message(
        ChatRequest(
            user_id="user-mixed-script-widow-fallback",
            message="Mere pati ki death ho gayi hai aur mujhe pension chahiye.",
        )
    )

    session = await store.get("user-mixed-script-widow-fallback")
    assert "male or female" not in result.text.lower()
    assert "age" in result.text.lower()
    assert session is not None
    assert session.user_profile.gender == "female"
    assert session.user_profile.marital_status == "widowed"
    assert session.currently_asking == "age"


@pytest.mark.asyncio
async def test_unlocked_hinglish_turn_does_not_drift_to_english_reply() -> None:
    """Strong Hinglish turns should stay Hinglish in unlocked sessions."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-unlocked-hinglish-drift",
            state=ConversationState.PROFILE_COLLECTION,
            user_profile=UserProfile(
                life_event="EDUCATION",
                age=18,
                category="SC",
            ),
            language_preference="hinglish",
            language_locked=False,
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "answer_field",
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": (
                "Yes, there are scholarships available for your daughter. "
                "Could you share the approximate annual family income?"
            ),
        }
    )
    mock_ai = AsyncMock()
    mock_ai.generate_response = AsyncMock(
        return_value=(
            "Haan, aapki beti ke liye scholarship options available hain. "
            "Approx annual family income bata dijiye."
        )
    )

    with patch(
        "src.services.response_generator.get_ai_orchestrator",
        return_value=mock_ai,
    ):
        result = await service.handle_message(
            ChatRequest(
                user_id="user-unlocked-hinglish-drift",
                message="Meri beti ke liye scholarship ya college support bhi mil sakta hai kya?",
            )
        )

    session = await store.get("user-unlocked-hinglish-drift")
    assert service.llm.analyze_message.await_args.kwargs["session_language"] == "hinglish"
    assert result.language == "hinglish"
    assert mock_ai.generate_response.await_count == 1
    assert result.text.startswith("Haan, aapki beti ke liye")
    assert "beti ke liye" in result.text.lower()
    assert session is not None
    assert session.language_preference == "hinglish"
    assert session.language_locked is False


@pytest.mark.asyncio
async def test_unlocked_hinglish_income_turn_preserves_scheme_list_language() -> None:
    """Income collection turns should keep the active Hinglish language on scheme presentation."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-unlocked-hinglish-income",
            state=ConversationState.PROFILE_COLLECTION,
            user_profile=UserProfile(
                life_event="EDUCATION",
                age=18,
                category="SC",
            ),
            messages=[
                Message(role="assistant", content="Applicant ke family ki approx annual income kitni hai?"),
            ],
            currently_asking="annual_income",
            language_preference="hinglish",
            language_locked=False,
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "answer_field",
            "life_event": None,
            "extracted_fields": {"annual_income": 120000},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )
    match_schemes = AsyncMock(
        return_value=[
            SchemeMatch(
                scheme=_make_scheme("SCH-EDU-LIST", life_event="EDUCATION", name="Education Loan Scheme - Delhi"),
                eligibility_match={"age": True, "gender": True, "category": True, "income": True},
                deterministic_score=0.9,
            )
        ]
    )

    with patch("src.services.conversation.scheme_matcher.match_schemes", match_schemes), patch(
        "src.services.conversation.format_inline_keyboard",
        return_value=[[{"text": "Education Loan Scheme - Delhi", "callback_data": "scheme:SCH-EDU-LIST"}]],
    ):
        result = await service.handle_message(
            ChatRequest(
                user_id="user-unlocked-hinglish-income",
                message="Family income 120000 per year hai.",
            )
        )

    session = await store.get("user-unlocked-hinglish-income")
    assert service.llm.analyze_message.await_args.kwargs["session_language"] == "hinglish"
    assert result.language == "hinglish"
    assert result.next_state == ConversationState.SCHEME_PRESENTATION.value
    assert "Aapke liye ye schemes mili hain" in result.text
    assert session is not None
    assert session.language_preference == "hinglish"
    assert session.language_locked is False


@pytest.mark.asyncio
async def test_explicit_topic_switch_ignores_bogus_scheme_selection_and_updates_need() -> None:
    """Topic switches should not get trapped by a hallucinated selected_scheme_id."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-topic-switch-live-shape",
            state=ConversationState.SCHEME_PRESENTATION,
            user_profile=UserProfile(
                life_event="DEATH_IN_FAMILY",
                gender="female",
                marital_status="widowed",
                age=35,
                category="SC",
                annual_income=90000,
            ),
            presented_schemes=[
                {
                    "id": "SCH-WIDOW",
                    "name": "Delhi Pension Scheme to Women in Distress (Widow Pension)",
                    "name_hindi": "दिल्ली महिला संकट पेंशन योजना",
                }
            ],
            language_preference="hinglish",
            language_locked=False,
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": None,
            "life_event": "EDUCATION",
            "extracted_fields": {},
            "language": "hinglish",
            "selected_scheme_id": "education",
            "response_text": None,
        }
    )
    match_schemes = AsyncMock(return_value=[])

    with patch("src.services.conversation.scheme_matcher.match_schemes", match_schemes):
        result = await service.handle_message(
            ChatRequest(
                user_id="user-topic-switch-live-shape",
                message="Ab mujhe meri beti ke liye education scheme dekhni hai, widow pension nahin.",
            )
        )

    session = await store.get("user-topic-switch-live-shape")
    assert match_schemes.await_count == 1
    called_profile = match_schemes.await_args.kwargs["profile"]
    assert called_profile.life_event == "EDUCATION"
    assert result.next_state == ConversationState.PROFILE_COLLECTION.value
    assert session is not None
    assert session.user_profile.life_event == "EDUCATION"
    assert session.selected_scheme_id is None
    assert session.presented_schemes == []


@pytest.mark.asyncio
async def test_explicit_topic_switch_suppresses_old_scheme_details_action() -> None:
    """A topic-switch turn should not reopen the old single presented scheme."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-topic-switch-old-details",
            state=ConversationState.SCHEME_PRESENTATION,
            user_profile=UserProfile(
                life_event="DEATH_IN_FAMILY",
                gender="female",
                marital_status="widowed",
                age=35,
                category="SC",
                annual_income=90000,
            ),
            presented_schemes=[
                {
                    "id": "SCH-DELHI-003",
                    "name": "Delhi Pension Scheme to Women in Distress (Widow Pension)",
                    "name_hindi": "दिल्ली महिला संकट पेंशन योजना",
                }
            ],
            language_preference="hinglish",
            language_locked=False,
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "request_details",
            "life_event": "DEATH_IN_FAMILY",
            "extracted_fields": {},
            "language": "hinglish",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )
    match_schemes = AsyncMock(return_value=[])

    with patch("src.services.conversation.scheme_matcher.match_schemes", match_schemes), patch(
        "src.services.conversation._build_scheme_details_text",
        AsyncMock(return_value="DETAILS"),
    ) as details_mock:
        result = await service.handle_message(
            ChatRequest(
                user_id="user-topic-switch-old-details",
                message="Ab mujhe meri beti ke liye education scheme dekhni hai, widow pension nahin.",
            )
        )

    session = await store.get("user-topic-switch-old-details")
    assert details_mock.await_count == 0
    assert match_schemes.await_count == 1
    called_profile = match_schemes.await_args.kwargs["profile"]
    assert called_profile.life_event == "EDUCATION"
    assert result.next_state == ConversationState.PROFILE_COLLECTION.value
    assert session is not None
    assert session.user_profile.life_event == "EDUCATION"
    assert session.selected_scheme_id is None
    assert session.presented_schemes == []


@pytest.mark.asyncio
async def test_spouse_conflict_in_llm_reply_falls_back_to_deterministic_question() -> None:
    """A wrong-spouse LLM reply should be discarded in favor of a safe deterministic prompt."""
    store = get_session_store()
    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "answer_field",
            "life_event": "DEATH_IN_FAMILY",
            "extracted_fields": {
                "gender": "female",
                "marital_status": "widowed",
            },
            "language": "hinglish",
            "selected_scheme_id": None,
            "response_text": (
                "Aapki patni ki maut ho gayi hai, mujhe samajh aaya. "
                "Kya aap apna district bata sakti hain?"
            ),
        }
    )

    result = await service.handle_message(
        ChatRequest(
            user_id="user-spouse-conflict-reply",
            message="Mere pati ki maut ho gayi hai, mujhe widow pension ke baare mein batayiye.",
        )
    )

    session = await store.get("user-spouse-conflict-reply")
    assert "patni" not in result.text.lower()
    assert result.text != (
        "Aapki patni ki maut ho gayi hai, mujhe samajh aaya. "
        "Kya aap apna district bata sakti hain?"
    )
    assert session is not None
    assert session.user_profile.gender == "female"
    assert session.user_profile.marital_status == "widowed"


@pytest.mark.asyncio
async def test_guardian_context_does_not_overinfer_married_for_child_beneficiary() -> None:
    """Guardian wording should not create a married applicant profile by default."""
    store = get_session_store()
    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "answer_field",
            "life_event": "EDUCATION",
            "extracted_fields": {
                "age": 18,
                "gender": "female",
                "marital_status": "married",
            },
            "language": "hinglish",
            "selected_scheme_id": None,
            "response_text": "Kya aap district bata sakti hain?",
        }
    )

    await service.handle_message(
        ChatRequest(
            user_id="user-guardian-overinference",
            message=(
                "Meri beti 18 saal ki hai aur usko college scholarship chahiye. "
                "Main uski maa hoon, meri age 35 hai."
            ),
        )
    )

    session = await store.get("user-guardian-overinference")
    assert session is not None
    assert session.user_profile.age == 18
    assert session.user_profile.gender == "female"
    assert session.user_profile.marital_status is None


@pytest.mark.asyncio
async def test_no_match_topic_switch_updates_life_event_and_reruns_matching() -> None:
    """A new topic should replace the stored life event and retry matching."""
    store = get_session_store()
    seed = Session(
        user_id="user-topic",
        state=ConversationState.PROFILE_COLLECTION,
        user_profile=UserProfile(
            life_event="EDUCATION",
            age=19,
            category="OBC",
            annual_income=400000,
        ),
        language_preference="en",
        awaiting_profile_change=True,
    )
    await store.save(seed)

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "answer_field",
            "life_event": "HOUSING",
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )
    match_schemes = AsyncMock(return_value=[])

    with patch("src.services.conversation.scheme_matcher.match_schemes", match_schemes):
        result = await service.handle_message(
            ChatRequest(user_id="user-topic", message="Now I need housing help instead")
        )

    session = await store.get("user-topic")
    assert result.next_state == ConversationState.PROFILE_COLLECTION.value
    assert match_schemes.await_count == 1
    called_profile = match_schemes.await_args.kwargs["profile"]
    assert called_profile.life_event == "HOUSING"
    assert session is not None
    assert session.user_profile.life_event == "HOUSING"
    assert session.awaiting_profile_change is True


@pytest.mark.asyncio
async def test_ai_relevance_gate_clarifies_cross_domain_candidate() -> None:
    """Low-confidence cross-domain results should trigger a clarification instead of presentation."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-ai-gate",
            state=ConversationState.PROFILE_COLLECTION,
            user_profile=UserProfile(
                life_event="HOUSING",
                age=25,
                category="OBC",
                annual_income=500000,
            ),
            language_preference="en",
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "none",
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )
    service.llm.judge_scheme_relevance = AsyncMock(
        return_value={
            "should_clarify": True,
            "clarification_question": "I understood that you want housing assistance. Do you want housing schemes specifically?",
            "overall_confidence": 0.2,
            "candidate_scores": [
                {
                    "scheme_id": "SCH-EDU",
                    "relevance_score": 0.1,
                    "topic_match": False,
                    "reason": "This candidate is education-related, not housing-related.",
                }
            ],
        }
    )
    match_schemes = AsyncMock(
        return_value=[
            SchemeMatch(
                scheme=_make_scheme("SCH-EDU", life_event="EDUCATION"),
                eligibility_match={"age": True, "income": True, "category": True},
                deterministic_score=0.8,
            )
        ]
    )

    with patch("src.services.conversation.scheme_matcher.match_schemes", match_schemes):
        result = await service.handle_message(
            ChatRequest(user_id="user-ai-gate", message="show me the schemes")
        )

    session = await store.get("user-ai-gate")
    assert result.next_state == ConversationState.SITUATION_UNDERSTANDING.value
    assert "housing" in result.text.lower()
    assert session is not None
    assert session.currently_asking == "life_event"
    assert session.presented_schemes == []


@pytest.mark.asyncio
async def test_relevance_judge_uses_active_need_summary_not_raw_income_reply() -> None:
    """AI relevance judging should see the active goal, not only the bare field answer."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-active-need-summary",
            state=ConversationState.PROFILE_COLLECTION,
            user_profile=UserProfile(
                life_event="DEATH_IN_FAMILY",
                gender="female",
                marital_status="widowed",
                age=23,
            ),
            messages=[
                Message(role="assistant", content="What is the applicant's approximate annual family income?"),
            ],
            currently_asking="annual_income",
            language_preference="en",
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "answer_field",
            "life_event": None,
            "extracted_fields": {"annual_income": 50000},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )
    service.llm.judge_scheme_relevance = AsyncMock(
        return_value={
            "should_clarify": False,
            "clarification_question": None,
            "overall_confidence": 0.92,
            "candidate_scores": [
                {
                    "scheme_id": "SCH-WIDOW",
                    "relevance_score": 0.92,
                    "topic_match": True,
                    "reason": "Widow-support scheme matches the active need and profile.",
                }
            ],
        }
    )
    match_schemes = AsyncMock(
        return_value=[
            SchemeMatch(
                scheme=_make_scheme(
                    "SCH-WIDOW",
                    life_event="DEATH_IN_FAMILY",
                    name="Widow Pension",
                    genders=["female"],
                    categories=["all"],
                ),
                eligibility_match={"age": True, "gender": True, "income": True},
                deterministic_score=0.85,
            )
        ]
    )

    with patch("src.services.conversation.scheme_matcher.match_schemes", match_schemes), patch(
        "src.services.conversation.format_inline_keyboard",
        return_value=[[{"text": "Widow Pension", "callback_data": "scheme:SCH-WIDOW"}]],
    ):
        await service.handle_message(
            ChatRequest(user_id="user-active-need-summary", message="50K INR")
        )

    judge_message = service.llm.judge_scheme_relevance.await_args.kwargs["user_message"]
    assert judge_message != "50K INR"
    assert "Need area: DEATH_IN_FAMILY" in judge_message
    assert "Marital status: widowed" in judge_message
    assert "Annual income: ₹50000" in judge_message


@pytest.mark.asyncio
async def test_clear_deterministic_match_skips_ai_relevance_judge() -> None:
    """High-confidence deterministic ranking should avoid an unnecessary AI judge call."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-skip-ai-judge",
            state=ConversationState.SCHEME_MATCHING,
            user_profile=UserProfile(
                life_event="HOUSING",
                age=29,
                annual_income=350000,
            ),
            language_preference="en",
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "none",
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )
    service.llm.judge_scheme_relevance = AsyncMock()
    match_schemes = AsyncMock(
        return_value=[
            SchemeMatch(
                scheme=_make_scheme("SCH-HIGH", life_event="HOUSING", name="Strong Housing Match"),
                eligibility_match={"age": True, "income": True},
                deterministic_score=0.97,
            ),
            SchemeMatch(
                scheme=_make_scheme("SCH-LOW", life_event="HOUSING", name="Weak Housing Match"),
                eligibility_match={"age": True, "income": True},
                deterministic_score=0.60,
            ),
        ]
    )

    with patch("src.services.conversation.scheme_matcher.match_schemes", match_schemes), patch(
        "src.services.conversation.format_inline_keyboard",
        return_value=[[{"text": "Strong Housing Match", "callback_data": "scheme:SCH-HIGH"}]],
    ):
        result = await service.handle_message(
            ChatRequest(user_id="user-skip-ai-judge", message="show me housing schemes")
        )

    assert result.next_state == ConversationState.SCHEME_PRESENTATION.value
    assert "strong housing match" in result.text.lower()
    assert service.llm.judge_scheme_relevance.await_count == 0


@pytest.mark.asyncio
async def test_followup_about_secondary_scheme_resolves_in_live_conversation() -> None:
    """Natural follow-ups about a secondary presented scheme should not fall back to the selected one."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-secondary-scheme",
            state=ConversationState.APPLICATION_HELP,
            user_profile=UserProfile(
                life_event="HOUSING",
                age=24,
                category="OBC",
                annual_income=500000,
            ),
            selected_scheme_id="SCH-DELHI-001",
            presented_schemes=[
                {
                    "id": "SCH-DELHI-001",
                    "name": "Pradhan Mantri Awas Yojana - Urban 2.0 (PMAY-U 2.0)",
                    "name_hindi": "प्रधानमंत्री आवास योजना - शहरी 2.0",
                },
                {
                    "id": "SCH-DELHI-006",
                    "name": "Education Loan Scheme - Delhi",
                    "name_hindi": "शिक्षा ऋण योजना - दिल्ली",
                },
            ],
            language_preference="en",
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "request_details",
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )

    education_scheme = _make_scheme(
        "SCH-DELHI-006",
        life_event="EDUCATION",
        name="Education Loan Scheme - Delhi",
        name_hindi="शिक्षा ऋण योजना - दिल्ली",
    )
    with patch(
        "src.services.conversation.scheme_repo.get_scheme_by_id",
        AsyncMock(return_value=education_scheme),
    ):
        result = await service.handle_message(
            ChatRequest(
                user_id="user-secondary-scheme",
                message="May I know why you suggested education loan scheme when I was asking for housing schemes?",
            )
        )

    session = await store.get("user-secondary-scheme")
    assert "education loan scheme" in result.text.lower()
    assert session is not None
    assert session.selected_scheme_id == "SCH-DELHI-006"


@pytest.mark.asyncio
async def test_scheme_term_question_gets_direct_answer_instead_of_card_replay() -> None:
    """Scheme-term follow-ups should be answered directly, not by replaying details."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-income-band",
            state=ConversationState.SCHEME_DETAILS,
            user_profile=UserProfile(
                life_event="HOUSING",
                age=24,
                annual_income=500000,
            ),
            selected_scheme_id="SCH-DELHI-001",
            language_preference="en",
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "request_details",
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )

    pmay_scheme = _make_scheme(
        "SCH-DELHI-001",
        life_event="HOUSING",
        name="Pradhan Mantri Awas Yojana - Urban 2.0 (PMAY-U 2.0)",
        income_by_category={"EWS": 300000, "LIG": 600000, "MIG": 900000},
        categories=["EWS", "LIG", "MIG"],
    )
    with patch(
        "src.services.conversation.scheme_repo.get_scheme_by_id",
        AsyncMock(return_value=pmay_scheme),
    ):
        result = await service.handle_message(
            ChatRequest(
                user_id="user-income-band",
                message="What does income band mean? What's LIG or MIG here?",
            )
        )

    assert "income band means" in result.text.lower()
    assert "lig" in result.text.lower()
    assert "₹6 lakh" in result.text
    assert "what would you like next" not in result.text.lower()


@pytest.mark.asyncio
async def test_justify_question_uses_scheme_answer_path() -> None:
    """Why-this-scheme questions should use the grounded answer path, not the static details card."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-justify-answer",
            state=ConversationState.SCHEME_DETAILS,
            user_profile=UserProfile(
                life_event="DEATH_IN_FAMILY",
                age=23,
                gender="female",
                marital_status="widowed",
                annual_income=50000,
            ),
            selected_scheme_id="SCH-DELHI-003",
            language_preference="en",
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "request_details",
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )

    with patch(
        "src.services.conversation.scheme_repo.get_scheme_by_id",
        AsyncMock(return_value=_make_scheme("SCH-DELHI-003", life_event="DEATH_IN_FAMILY", genders=["female"], categories=["all"])),
    ), patch(
        "src.services.conversation.response_generator.generate_scheme_question_response",
        AsyncMock(return_value="I suggested this because you said you are widowed and your income is below ₹1 lakh."),
    ) as answer_mock:
        result = await service.handle_message(
            ChatRequest(
                user_id="user-justify-answer",
                message="Please justify your decision of choosing this scheme",
            )
        )

    assert "widowed" in result.text.lower()
    assert answer_mock.await_count == 1


@pytest.mark.asyncio
async def test_language_switch_justify_question_still_uses_scheme_answer_path() -> None:
    """Explicit language requests should not block scheme Q&A routing."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-language-justify-answer",
            state=ConversationState.SCHEME_DETAILS,
            user_profile=UserProfile(
                life_event="HOUSING",
                age=24,
                annual_income=500000,
            ),
            selected_scheme_id="SCH-DELHI-001",
            language_preference="hi",
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "change_language",
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )

    with patch(
        "src.services.conversation.scheme_repo.get_scheme_by_id",
        AsyncMock(return_value=_make_scheme("SCH-DELHI-001", life_event="HOUSING")),
    ), patch(
        "src.services.conversation.response_generator.generate_scheme_question_response",
        AsyncMock(return_value="I suggested this because your housing need and income fit the scheme rules."),
    ) as answer_mock:
        result = await service.handle_message(
            ChatRequest(
                user_id="user-language-justify-answer",
                message="In english, can you justify the decision for suggesting me this scheme?",
            )
        )

    session = await store.get("user-language-justify-answer")
    assert "housing need" in result.text.lower()
    assert result.language == "en"
    assert answer_mock.await_count == 1
    assert session is not None
    assert session.language_preference == "en"


@pytest.mark.asyncio
async def test_language_switch_translation_request_uses_scheme_answer_path() -> None:
    """Translate-this-detail follow-ups should not fall back to the static card."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-translate-scheme-answer",
            state=ConversationState.SCHEME_DETAILS,
            user_profile=UserProfile(
                life_event="HOUSING",
                age=24,
                annual_income=500000,
            ),
            messages=[
                Message(
                    role="assistant",
                    content=(
                        "In this scheme, income band means the annual family income bracket "
                        "used to decide which segment applies."
                    ),
                ),
            ],
            selected_scheme_id="SCH-DELHI-001",
            language_preference="en",
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "change_language",
            "life_event": None,
            "extracted_fields": {},
            "language": "hi",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )

    with patch(
        "src.services.conversation.scheme_repo.get_scheme_by_id",
        AsyncMock(return_value=_make_scheme("SCH-DELHI-001", life_event="HOUSING")),
    ), patch(
        "src.services.conversation.response_generator.generate_scheme_question_response",
        AsyncMock(return_value="इस योजना में आय वर्ग का मतलब वार्षिक पारिवारिक आय का समूह है।"),
    ) as answer_mock:
        result = await service.handle_message(
            ChatRequest(
                user_id="user-translate-scheme-answer",
                message="Can you provide these details in Hindi language as well?",
            )
        )

    assert "आय वर्ग" in result.text
    assert answer_mock.await_count == 1


@pytest.mark.asyncio
async def test_scheme_eligibility_question_with_profile_update_stays_on_answer_path() -> None:
    """Scheme-context eligibility questions should not get overridden by rematching."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-scheme-eligibility-followup",
            state=ConversationState.SCHEME_DETAILS,
            user_profile=UserProfile(
                life_event="DEATH_IN_FAMILY",
                age=45,
                gender="female",
                annual_income=50000,
            ),
            selected_scheme_id="SCH-DELHI-003",
            language_preference="en",
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "request_details",
            "life_event": None,
            "extracted_fields": {"category": "General"},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )
    match_schemes = AsyncMock()

    with patch(
        "src.services.conversation.scheme_repo.get_scheme_by_id",
        AsyncMock(
            return_value=_make_scheme(
                "SCH-DELHI-003",
                life_event="DEATH_IN_FAMILY",
                name="Delhi Pension Scheme to Women in Distress (Widow Pension)",
                genders=["female"],
                categories=["all"],
            )
        ),
    ), patch(
        "src.services.conversation.scheme_matcher.match_schemes",
        match_schemes,
    ), patch(
        "src.services.conversation.response_generator.generate_scheme_question_response",
        AsyncMock(
            return_value=(
                "General category does not disqualify her for this scheme. "
                "Based on the details shared so far, she appears eligible on the age, gender, and income checks."
            )
        ),
    ) as answer_mock:
        result = await service.handle_message(
            ChatRequest(
                user_id="user-scheme-eligibility-followup",
                message=(
                    "What's the eligibility criteria for this scheme? "
                    "My mother belongs to general category does she qualify?"
                ),
            )
        )

    session = await store.get("user-scheme-eligibility-followup")
    assert result.next_state == ConversationState.SCHEME_DETAILS.value
    assert "general category" in result.text.lower()
    assert answer_mock.await_count == 1
    assert match_schemes.await_count == 0
    assert session is not None
    assert session.state == ConversationState.SCHEME_DETAILS
    assert session.user_profile.category == "General"


@pytest.mark.asyncio
async def test_low_context_field_answer_skips_relevance_clarification_loop() -> None:
    """A parsed field reply should present deterministic matches instead of re-confirming the topic."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-income-loop",
            state=ConversationState.PROFILE_COLLECTION,
            user_profile=UserProfile(
                life_event="HOUSING",
                age=24,
            ),
            messages=[
                Message(role="assistant", content="What is the applicant's approximate annual family income?"),
            ],
            currently_asking="annual_income",
            language_preference="en",
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "answer_field",
            "life_event": None,
            "extracted_fields": {"annual_income": 500000},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )
    service.llm.judge_scheme_relevance = AsyncMock(
        return_value={
            "should_clarify": True,
            "clarification_question": "I understood that you want housing assistance. Do you want housing schemes specifically?",
            "overall_confidence": 0.2,
            "candidate_scores": [],
        }
    )
    match_schemes = AsyncMock(
        return_value=[
            SchemeMatch(
                scheme=_make_scheme("SCH-HOUSE", life_event="HOUSING", name="Housing Relief Scheme"),
                eligibility_match={"age": True, "income": True},
                deterministic_score=0.85,
            )
        ]
    )

    with patch("src.services.conversation.scheme_matcher.match_schemes", match_schemes), patch(
        "src.services.conversation.format_inline_keyboard",
        return_value=[[{"text": "Housing Relief Scheme", "callback_data": "scheme:SCH-HOUSE"}]],
    ):
        result = await service.handle_message(
            ChatRequest(user_id="user-income-loop", message="Roughly 5 lakhs")
        )

    query_text = match_schemes.await_args.kwargs["query_text"]
    judge_message = service.llm.judge_scheme_relevance.await_args.kwargs["user_message"]
    assert result.next_state == ConversationState.SCHEME_PRESENTATION.value
    assert "housing relief scheme" in result.text.lower()
    assert "Need area: HOUSING" in query_text
    assert "Need area: HOUSING" in judge_message
    assert service.llm.judge_scheme_relevance.await_count == 1


@pytest.mark.asyncio
async def test_unlocked_hinglish_field_style_reply_preserves_language_for_matching() -> None:
    """Longer field-style replies should keep unlocked Hinglish sessions stable."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-hinglish-field-style",
            state=ConversationState.PROFILE_COLLECTION,
            user_profile=UserProfile(
                life_event="DEATH_IN_FAMILY",
                gender="female",
                marital_status="widowed",
            ),
            messages=[
                Message(
                    role="assistant",
                    content=(
                        "Aapko kis tarah ki madad chahiye? "
                        "Jaise housing, health, education, ya rojgaar?"
                    ),
                ),
            ],
            currently_asking="life_event",
            language_preference="hinglish",
            language_locked=False,
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "answer_field",
            "life_event": None,
            "extracted_fields": {
                "age": 35,
                "category": "SC",
                "annual_income": 70000,
            },
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )
    match_schemes = AsyncMock(
        return_value=[
            SchemeMatch(
                scheme=_make_scheme(
                    "SCH-WIDOW",
                    life_event="DEATH_IN_FAMILY",
                    name="Widow Pension",
                    name_hindi="Widow Pension",
                    genders=["female"],
                    categories=["all"],
                ),
                eligibility_match={"age": True, "gender": True, "income": True},
                deterministic_score=0.9,
            )
        ]
    )

    with patch("src.services.conversation.scheme_matcher.match_schemes", match_schemes):
        result = await service.handle_message(
            ChatRequest(
                user_id="user-hinglish-field-style",
                message="I am 35, SC, and family income is around 70000 yearly.",
            )
        )

    session = await store.get("user-hinglish-field-style")
    assert service.llm.analyze_message.await_args.kwargs["session_language"] == "hinglish"
    assert result.language == "hinglish"
    assert "Aapke liye ye schemes mili hain" in result.text
    assert session is not None
    assert session.language_preference == "hinglish"


@pytest.mark.asyncio
async def test_scheme_question_response_includes_last_assistant_answer_for_translation() -> None:
    """Scheme Q&A context should include the previous assistant reply for translation requests."""
    session = Session(
        user_id="user-response-translation-context",
        state=ConversationState.SCHEME_DETAILS,
        user_profile=UserProfile(
            life_event="HOUSING",
            age=24,
            annual_income=500000,
        ),
        messages=[
            Message(
                role="assistant",
                content=(
                    "In this scheme, income band means the annual family income bracket "
                    "used to decide which segment applies."
                ),
            ),
        ],
        language_preference="hi",
    )
    scheme = _make_scheme(
        "SCH-DELHI-001",
        life_event="HOUSING",
        income_by_category={"EWS": 300000, "LIG": 600000, "MIG": 900000},
        categories=["EWS", "LIG", "MIG"],
    )

    with patch(
        "src.services.response_generator.generate_response",
        AsyncMock(return_value="अनुवादित उत्तर"),
    ) as generate_mock:
        result = await response_generator.generate_scheme_question_response(
            session,
            scheme,
            session.user_profile,
            "Can you say this in Hindi as well?",
            "hi",
        )

    context = generate_mock.await_args.args[1]
    assert result == "अनुवादित उत्तर"
    assert "income band means" in context["last_assistant_response"].lower()


@pytest.mark.asyncio
async def test_scheme_question_response_answers_justification_deterministically() -> None:
    """Why-this-scheme answers should use grounded deterministic reasons instead of the LLM."""
    session = Session(
        user_id="user-grounded-reasons",
        state=ConversationState.SCHEME_DETAILS,
        user_profile=UserProfile(
            life_event="HOUSING",
            age=24,
            annual_income=400000,
        ),
        language_preference="en",
    )
    scheme = Scheme(
        id="SCH-GROUNDED",
        name="Housing Support Scheme",
        name_hindi="हाउसिंग सहायता योजना",
        department="Test Department",
        department_hindi="परीक्षण विभाग",
        level="state",
        description="Income-based housing support.",
        description_hindi="आय आधारित हाउसिंग सहायता।",
        benefits_amount=250000,
        benefits_frequency="one-time",
        eligibility=EligibilityCriteria(
            min_age=None,
            max_age=None,
            max_income=500000,
            genders=["all"],
            categories=["all"],
        ),
        life_events=["HOUSING"],
        documents_required=[],
        application_url="https://example.com/apply",
        offline_process="Visit the district office",
    )

    with patch(
        "src.services.response_generator.generate_response",
        AsyncMock(return_value="LLM fallback should not be used"),
    ) as generate_mock:
        result = await response_generator.generate_scheme_question_response(
            session,
            scheme,
            session.user_profile,
            "Why did you suggest this scheme?",
            "en",
        )

    assert "grounded reasons" in result.lower()
    assert "max income" in result.lower()
    assert "user age" not in result.lower()
    assert generate_mock.await_count == 0


@pytest.mark.asyncio
async def test_scheme_question_response_answers_eligibility_deterministically() -> None:
    """Eligibility questions should use grounded deterministic text instead of the LLM."""
    session = Session(
        user_id="user-deterministic-eligibility-answer",
        state=ConversationState.SCHEME_DETAILS,
        user_profile=UserProfile(
            life_event="DEATH_IN_FAMILY",
            age=45,
            gender="female",
            category="General",
            annual_income=50000,
        ),
        selected_scheme_id="SCH-DELHI-003",
        language_preference="en",
    )
    scheme = Scheme(
        id="SCH-DELHI-003",
        name="Delhi Pension Scheme to Women in Distress (Widow Pension)",
        name_hindi="दिल्ली महिला विपत्ति पेंशन योजना (विधवा पेंशन)",
        department="Department of Women and Child Development",
        department_hindi="महिला एवं बाल विकास विभाग",
        level="state",
        description="Monthly assistance for women in distress.",
        description_hindi="विपत्ति में महिलाओं के लिए मासिक सहायता।",
        benefits_amount=2500,
        benefits_frequency="monthly",
        eligibility=EligibilityCriteria(
            min_age=18,
            max_income=100000,
            genders=["female"],
            categories=["all"],
        ),
        life_events=["DEATH_IN_FAMILY"],
        documents_required=[],
        application_url="https://example.com/apply",
        offline_process="Apply online",
    )

    with patch(
        "src.services.response_generator.generate_response",
        AsyncMock(return_value="LLM fallback should not be used"),
    ) as generate_mock:
        result = await response_generator.generate_scheme_question_response(
            session,
            scheme,
            session.user_profile,
            "What's the eligibility criteria for this scheme? My mother belongs to general category, does she qualify?",
            "en",
        )

    assert "general category" in result.lower()
    assert "does not have a caste-category restriction" in result.lower()
    assert "appears eligible" in result.lower()
    assert generate_mock.await_count == 0


def test_truncate_at_sentence_handles_hindi_danda() -> None:
    """Hindi descriptions should truncate at sentence boundaries instead of mid-thought."""
    text = "यह पहला वाक्य है। यह दूसरा वाक्य है जिसे बाद में काटना चाहिए।"
    assert _truncate_at_sentence(text, 25) == "यह पहला वाक्य है।"


@pytest.mark.asyncio
async def test_application_request_uses_step_by_step_guidance_when_steps_exist() -> None:
    """Application-step asks should use `scheme.application_steps`, not only the offline summary."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-application-steps",
            state=ConversationState.SCHEME_DETAILS,
            user_profile=UserProfile(
                life_event="HEALTH_CRISIS",
                age=24,
                annual_income=250000,
            ),
            selected_scheme_id="SCH-DAK",
            language_preference="en",
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "request_application",
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )

    scheme = _make_scheme(
        "SCH-DAK",
        life_event="HEALTH_CRISIS",
        name="Delhi Arogya Kosh (DAK)",
        name_hindi="दिल्ली आरोग्य कोष (डीएके)",
    ).model_copy(
        update={
            "application_url": None,
            "application_steps": [
                "Step 1: Collect the DAK application form.",
                "Step 2: Submit it in person with the required documents.",
            ],
            "offline_process": "Visit the Patient Welfare Cell with original documents.",
        }
    )

    with patch(
        "src.services.conversation.scheme_repo.get_scheme_by_id",
        AsyncMock(return_value=scheme),
    ):
        result = await service.handle_message(
            ChatRequest(user_id="user-application-steps", message="application steps")
        )

    assert result.next_state == ConversationState.APPLICATION_HELP.value
    assert "Step 1:" in result.text
    assert "Step 2:" in result.text
    assert "Patient Welfare Cell" in result.text


@pytest.mark.asyncio
async def test_application_help_followup_question_stays_on_answer_path() -> None:
    """Question-shaped application follow-ups should not replay the same application card."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-application-loop",
            state=ConversationState.APPLICATION_HELP,
            user_profile=UserProfile(
                life_event="HEALTH_CRISIS",
                age=24,
                annual_income=250000,
            ),
            selected_scheme_id="SCH-DAK",
            language_preference="en",
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "request_application",
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )

    scheme = _make_scheme(
        "SCH-DAK",
        life_event="HEALTH_CRISIS",
        name="Delhi Arogya Kosh (DAK)",
        name_hindi="दिल्ली आरोग्य कोष (डीएके)",
    ).model_copy(
        update={
            "application_url": None,
            "application_steps": [
                "Step 1: Collect the DAK application form.",
                "Step 2: Submit it in person with the required documents.",
            ],
            "offline_process": "Visit the Patient Welfare Cell with original documents.",
        }
    )

    with patch(
        "src.services.conversation.scheme_repo.get_scheme_by_id",
        AsyncMock(return_value=scheme),
    ), patch(
        "src.services.conversation.response_generator.generate_scheme_question_response",
        AsyncMock(return_value="The first step is to collect the DAK application form."),
    ) as answer_mock:
        result = await service.handle_message(
            ChatRequest(
                user_id="user-application-loop",
                message="What is the first application step?",
            )
        )

    assert result.next_state == ConversationState.APPLICATION_HELP.value
    assert "first step" in result.text.lower()
    assert answer_mock.await_count == 1


@pytest.mark.asyncio
async def test_procedure_request_routes_to_application_help() -> None:
    """Procedure/process wording should enter application help deterministically."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-procedure-route",
            state=ConversationState.SCHEME_DETAILS,
            user_profile=UserProfile(
                life_event="BUSINESS_STARTUP",
                age=25,
                category="General",
                annual_income=250000,
            ),
            selected_scheme_id="SCH-5",
            language_preference="en",
            language_locked=True,
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": None,
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )

    with patch(
        "src.services.conversation._build_application_help_text",
        AsyncMock(return_value="📝 How to Apply for Scheme SCH-5:\n\nStep 1: Submit the form."),
    ), patch(
        "src.services.conversation._build_scheme_details_text",
        AsyncMock(return_value="Scheme details"),
    ):
        result = await service.handle_message(
            ChatRequest(user_id="user-procedure-route", message="procedure")
        )

    assert result.next_state == ConversationState.APPLICATION_HELP.value
    assert "How to Apply" in result.text


@pytest.mark.asyncio
async def test_rgsry_application_followup_ignores_echoed_active_scheme_selection() -> None:
    """Echoed same-scheme selections must not replay RGSRY scheme details."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-rgsry-echoed-selection",
            state=ConversationState.SCHEME_DETAILS,
            user_profile=UserProfile(
                life_event="JOB_LOSS",
                age=24,
                category="General",
                annual_income=100000,
            ),
            selected_scheme_id="SCH-DELHI-005",
            presented_schemes=[
                {
                    "id": "SCH-DELHI-005",
                    "name": "Rajiv Gandhi Swavlamban Rojgar Yojana (RGSRY)",
                    "name_hindi": "राजीव गांधी स्वावलंबन रोज़गार योजना (आरजीएसआरवाई)",
                }
            ],
            language_preference="hinglish",
            language_locked=True,
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": None,
            "life_event": None,
            "extracted_fields": {},
            "language": "hinglish",
            "selected_scheme_id": "SCH-DELHI-005",
            "response_text": None,
        }
    )

    with patch(
        "src.services.conversation._build_application_help_text",
        AsyncMock(return_value="APP HELP"),
    ) as application_mock, patch(
        "src.services.conversation._build_scheme_details_text",
        AsyncMock(return_value="DETAILS"),
    ) as details_mock:
        result = await service.handle_message(
            ChatRequest(
                user_id="user-rgsry-echoed-selection",
                message="mujhe application steps dekhne hai",
            )
        )

    assert result.next_state == ConversationState.APPLICATION_HELP.value
    assert result.text == "APP HELP"
    assert application_mock.await_count == 1
    assert details_mock.await_count == 0


@pytest.mark.asyncio
async def test_locked_english_scheme_followup_rewrites_hindi_application_reply() -> None:
    """Locked English sessions should rewrite stray Hindi scheme follow-up replies."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-english-scheme-rewrite",
            state=ConversationState.SCHEME_DETAILS,
            user_profile=UserProfile(
                life_event="JOB_LOSS",
                age=25,
                category="General",
                annual_income=250000,
            ),
            selected_scheme_id="SCH-5",
            language_preference="en",
            language_locked=True,
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "request_application",
            "life_event": None,
            "extracted_fields": {},
            "language": "hi",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )
    mock_ai = AsyncMock()
    mock_ai.generate_response = AsyncMock(
        return_value=(
            "📝 How to Apply for Rajiv Gandhi Swavlamban Rojgar Yojana (RGSRY):\n\n"
            "Step 1: Submit the application form."
        )
    )

    with patch(
        "src.services.conversation._build_application_help_text",
        AsyncMock(
            return_value=(
                "📝 योजना के लिए आवेदन:\n\n"
                "चरण 1: आवेदन पत्र जमा करें।"
            )
        ),
    ), patch(
        "src.services.response_generator.get_ai_orchestrator",
        return_value=mock_ai,
    ):
        result = await service.handle_message(
            ChatRequest(
                user_id="user-english-scheme-rewrite",
                message="I would like to see the application procedure for this scheme.",
            )
        )

    assert result.next_state == ConversationState.APPLICATION_HELP.value
    assert "How to Apply for Rajiv Gandhi Swavlamban Rojgar Yojana" in result.text
    assert "चरण" not in result.text
    assert mock_ai.generate_response.await_count == 1


@pytest.mark.asyncio
async def test_same_scheme_question_with_echoed_selection_stays_on_answer_path() -> None:
    """Same-scheme question follow-ups should still use the answer path."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-rgsry-question-echo",
            state=ConversationState.APPLICATION_HELP,
            user_profile=UserProfile(
                life_event="JOB_LOSS",
                age=24,
                category="General",
                annual_income=100000,
            ),
            selected_scheme_id="SCH-DELHI-005",
            presented_schemes=[
                {
                    "id": "SCH-DELHI-005",
                    "name": "Rajiv Gandhi Swavlamban Rojgar Yojana (RGSRY)",
                    "name_hindi": "राजीव गांधी स्वावलंबन रोज़गार योजना (आरजीएसआरवाई)",
                }
            ],
            language_preference="en",
            language_locked=True,
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": None,
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": "SCH-DELHI-005",
            "response_text": None,
        }
    )

    with patch(
        "src.services.conversation.response_generator.generate_scheme_question_response",
        AsyncMock(return_value="The first step is to complete Part A."),
    ) as answer_mock, patch(
        "src.services.conversation.scheme_repo.get_scheme_by_id",
        AsyncMock(
            return_value=_make_scheme(
                "SCH-DELHI-005",
                life_event="JOB_LOSS",
                name="Rajiv Gandhi Swavlamban Rojgar Yojana (RGSRY)",
                name_hindi="राजीव गांधी स्वावलंबन रोज़गार योजना (आरजीएसआरवाई)",
            )
        ),
    ), patch(
        "src.services.conversation._build_application_help_text",
        AsyncMock(return_value="APP HELP"),
    ) as application_mock:
        result = await service.handle_message(
            ChatRequest(
                user_id="user-rgsry-question-echo",
                message="What is the first application step?",
            )
        )

    assert result.next_state == ConversationState.APPLICATION_HELP.value
    assert "first step" in result.text.lower()
    assert answer_mock.await_count == 1
    assert application_mock.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("followup_text", "expected_state", "builder_name", "sentinel"),
    [
        (
            "show the required documents",
            ConversationState.DOCUMENT_GUIDANCE,
            "_build_document_guidance_text",
            "DOC HELP",
        ),
        (
            "show rejection warnings",
            ConversationState.REJECTION_WARNINGS,
            "_build_rejection_warnings_text",
            "REJECTION HELP",
        ),
        (
            "show application steps",
            ConversationState.APPLICATION_HELP,
            "_build_application_help_text",
            "APP HELP",
        ),
    ],
    ids=["documents", "rejections", "application"],
)
@pytest.mark.parametrize(
    "scheme_seed",
    ACTIVE_SCHEME_SEEDS,
    ids=[seed["id"] for seed in ACTIVE_SCHEME_SEEDS],
)
async def test_same_scheme_followup_routing_works_for_every_active_scheme(
    scheme_seed: dict[str, str],
    followup_text: str,
    expected_state: ConversationState,
    builder_name: str,
    sentinel: str,
) -> None:
    """Every active scheme should honor explicit subview asks despite echoed selection ids."""
    store = get_session_store()
    await store.save(
        Session(
            user_id=f"user-{scheme_seed['id']}-{expected_state.value}",
            state=ConversationState.SCHEME_DETAILS,
            user_profile=UserProfile(
                life_event="JOB_LOSS",
                age=24,
                category="General",
                annual_income=100000,
            ),
            selected_scheme_id=scheme_seed["id"],
            presented_schemes=[scheme_seed],
            language_preference="en",
            language_locked=True,
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": None,
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": scheme_seed["id"],
            "response_text": None,
        }
    )

    patch_targets = {
        "_build_document_guidance_text": AsyncMock(return_value="DOC HELP"),
        "_build_rejection_warnings_text": AsyncMock(return_value="REJECTION HELP"),
        "_build_application_help_text": AsyncMock(return_value="APP HELP"),
        "_build_scheme_details_text": AsyncMock(return_value="DETAILS"),
    }

    with patch(
        "src.services.conversation._build_document_guidance_text",
        patch_targets["_build_document_guidance_text"],
    ), patch(
        "src.services.conversation._build_rejection_warnings_text",
        patch_targets["_build_rejection_warnings_text"],
    ), patch(
        "src.services.conversation._build_application_help_text",
        patch_targets["_build_application_help_text"],
    ), patch(
        "src.services.conversation._build_scheme_details_text",
        patch_targets["_build_scheme_details_text"],
    ):
        result = await service.handle_message(
            ChatRequest(
                user_id=f"user-{scheme_seed['id']}-{expected_state.value}",
                message=followup_text,
            )
        )

    assert result.next_state == expected_state.value
    assert result.text == sentinel
    assert patch_targets[builder_name].await_count == 1
    assert patch_targets["_build_scheme_details_text"].await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scheme_seed",
    ACTIVE_SCHEME_SEEDS,
    ids=[f"{seed['id']}-rejection-question" for seed in ACTIVE_SCHEME_SEEDS],
)
async def test_rejection_followup_question_stays_on_answer_path_for_every_active_scheme(
    scheme_seed: dict[str, str],
) -> None:
    """Every active scheme should keep rejection follow-up questions on the answer path."""
    store = get_session_store()
    user_id = f"user-{scheme_seed['id']}-rejection-question"
    await store.save(
        Session(
            user_id=user_id,
            state=ConversationState.REJECTION_WARNINGS,
            user_profile=UserProfile(
                life_event="JOB_LOSS",
                age=24,
                category="General",
                annual_income=100000,
            ),
            selected_scheme_id=scheme_seed["id"],
            presented_schemes=[scheme_seed],
            language_preference="en",
            language_locked=True,
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": None,
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": scheme_seed["id"],
            "response_text": None,
        }
    )

    with patch(
        "src.services.conversation.response_generator.generate_scheme_question_response",
        AsyncMock(return_value="The top rejection risk is incomplete or inconsistent information."),
    ) as answer_mock, patch(
        "src.services.conversation.scheme_repo.get_scheme_by_id",
        AsyncMock(
            return_value=_make_scheme(
                scheme_seed["id"],
                life_event="JOB_LOSS",
                name=scheme_seed["name"],
                name_hindi=scheme_seed["name_hindi"],
            )
        ),
    ), patch(
        "src.services.conversation._build_rejection_warnings_text",
        AsyncMock(return_value="REJECTION CARD"),
    ) as warnings_mock:
        result = await service.handle_message(
            ChatRequest(
                user_id=user_id,
                message="Which rejection risk should I be most careful about?",
            )
        )

    assert result.next_state == ConversationState.REJECTION_WARNINGS.value
    assert "rejection risk" in result.text.lower()
    assert answer_mock.await_count == 1
    assert warnings_mock.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scheme_seed",
    ACTIVE_SCHEME_SEEDS,
    ids=[f"{seed['id']}-affirmative" for seed in ACTIVE_SCHEME_SEEDS],
)
async def test_affirmative_rejection_followup_enters_application_help_for_every_active_scheme(
    scheme_seed: dict[str, str],
) -> None:
    """Every active scheme should allow a short affirmative reply to move into application help."""
    store = get_session_store()
    user_id = f"user-{scheme_seed['id']}-affirmative"
    await store.save(
        Session(
            user_id=user_id,
            state=ConversationState.REJECTION_WARNINGS,
            user_profile=UserProfile(
                life_event="JOB_LOSS",
                age=24,
                category="General",
                annual_income=100000,
            ),
            selected_scheme_id=scheme_seed["id"],
            presented_schemes=[scheme_seed],
            language_preference="en",
            language_locked=True,
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": None,
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": scheme_seed["id"],
            "response_text": None,
        }
    )

    with patch(
        "src.services.conversation._build_application_help_text",
        AsyncMock(return_value="APP HELP"),
    ) as application_mock, patch(
        "src.services.conversation._build_rejection_warnings_text",
        AsyncMock(return_value="REJECTION CARD"),
    ) as warnings_mock:
        result = await service.handle_message(
            ChatRequest(user_id=user_id, message="yes")
        )

    assert result.next_state == ConversationState.APPLICATION_HELP.value
    assert result.text == "APP HELP"
    assert application_mock.await_count == 1
    assert warnings_mock.await_count == 0


@pytest.mark.asyncio
async def test_hindi_document_guidance_translates_english_document_fields() -> None:
    """Hindi document guidance should not leak raw English document metadata."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-doc-hindi",
            state=ConversationState.SCHEME_DETAILS,
            user_profile=UserProfile(
                life_event="HEALTH_CRISIS",
                age=24,
                annual_income=250000,
            ),
            selected_scheme_id="SCH-DAK",
            language_preference="hi",
            language_locked=True,
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "request_details",
            "life_event": None,
            "extracted_fields": {},
            "language": "hi",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )

    scheme = _make_scheme(
        "SCH-DAK",
        life_event="HEALTH_CRISIS",
        name="Delhi Arogya Kosh (DAK)",
        name_hindi="दिल्ली आरोग्य कोष (डीएके)",
    ).model_copy(update={"documents_required": ["DOC-INCOME"]})
    document_chain = DocumentChain(
        document=Document(
            id="DOC-INCOME",
            name="Income Certificate",
            name_hindi="आय प्रमाण पत्र",
            issuing_authority="District Office",
            online_portal=None,
            prerequisites=[],
            fee=None,
            fee_bpl=None,
            processing_time="7 days",
            validity_period=None,
            format_requirements=[],
            common_mistakes=[],
            delhi_offices=[],
        ),
        prerequisites=[],
        depth=0,
    )
    mock_ai = AsyncMock()
    mock_ai.generate_response = AsyncMock(
        return_value=(
            "📄 दिल्ली आरोग्य कोष (डीएके) के दस्तावेज:\n\n"
            "1. आय प्रमाण पत्र\n"
            "   🏛️ कहाँ से: जिला कार्यालय\n"
            "   📋 समय: 7 दिन\n\n"
            "अगर चाहें तो मैं common rejection warnings भी बता सकता हूँ।"
        )
    )

    with patch(
        "src.services.conversation.scheme_repo.get_scheme_by_id",
        AsyncMock(return_value=scheme),
    ), patch(
        "src.services.conversation.document_resolver.resolve_documents_for_scheme",
        AsyncMock(return_value=[document_chain]),
    ), patch(
        "src.services.response_generator.get_ai_orchestrator",
        return_value=mock_ai,
    ):
        result = await service.handle_message(
            ChatRequest(user_id="user-doc-hindi", message="दस्तावेज")
        )

    assert result.next_state == ConversationState.DOCUMENT_GUIDANCE.value
    assert "जिला कार्यालय" in result.text
    assert "District Office" not in result.text
    assert mock_ai.generate_response.await_count == 1


@pytest.mark.asyncio
async def test_document_guidance_followup_question_stays_on_answer_path() -> None:
    """Question-shaped document follow-ups should not replay the full document list."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-document-loop",
            state=ConversationState.DOCUMENT_GUIDANCE,
            user_profile=UserProfile(
                life_event="HEALTH_CRISIS",
                age=24,
                annual_income=250000,
            ),
            selected_scheme_id="SCH-DAK",
            language_preference="en",
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "request_details",
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )

    with patch(
        "src.services.conversation.scheme_repo.get_scheme_by_id",
        AsyncMock(return_value=_make_scheme("SCH-DAK", life_event="HEALTH_CRISIS")),
    ), patch(
        "src.services.conversation.response_generator.generate_scheme_question_response",
        AsyncMock(return_value="The affidavit / self-declaration is the document that needs two witnesses."),
    ) as answer_mock:
        result = await service.handle_message(
            ChatRequest(
                user_id="user-document-loop",
                message="Which document needs two witnesses?",
            )
        )

    assert result.next_state == ConversationState.DOCUMENT_GUIDANCE.value
    assert "two witnesses" in result.text.lower()
    assert answer_mock.await_count == 1


@pytest.mark.asyncio
async def test_hindi_rejection_guidance_prefers_hindi_rule_content() -> None:
    """Hindi rejection warnings should prefer Hindi rule content over English prevention tips."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-rejection-hi",
            state=ConversationState.SCHEME_DETAILS,
            user_profile=UserProfile(
                life_event="HEALTH_CRISIS",
                age=24,
                annual_income=250000,
            ),
            selected_scheme_id="SCH-DAK",
            language_preference="hi",
            language_locked=True,
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "request_details",
            "life_event": None,
            "extracted_fields": {},
            "language": "hi",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )

    warning = RejectionRule(
        id="RULE-DAK-001",
        scheme_id="SCH-DAK",
        rule_type="procedural",
        description="Application will be rejected if documents are incomplete.",
        description_hindi="यदि दस्तावेज अधूरे हैं तो आवेदन अस्वीकार हो जाएगा।",
        severity="critical",
        prevention_tip="Bring every document together at submission time.",
        examples=[],
    )

    with patch(
        "src.services.conversation.scheme_repo.get_scheme_by_id",
        AsyncMock(return_value=_make_scheme("SCH-DAK", life_event="HEALTH_CRISIS")),
    ), patch(
        "src.services.conversation.rejection_engine.get_rejection_warnings",
        AsyncMock(return_value=[warning]),
    ):
        result = await service.handle_message(
            ChatRequest(user_id="user-rejection-hi", message="rejection warnings")
        )

    assert result.next_state == ConversationState.REJECTION_WARNINGS.value
    assert "यदि दस्तावेज अधूरे हैं" in result.text
    assert "Bring every document together" not in result.text


@pytest.mark.asyncio
async def test_document_request_moves_to_document_guidance() -> None:
    """Document follow-ups should enter the dedicated document guidance state."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-docs",
            state=ConversationState.SCHEME_DETAILS,
            user_profile=UserProfile(
                life_event="EDUCATION",
                age=19,
                category="OBC",
                annual_income=400000,
            ),
            selected_scheme_id="SCH-1",
            language_preference="en",
        )
    )

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "question",
            "action": "request_details",
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )

    with patch("src.services.conversation.scheme_repo.get_scheme_by_id", AsyncMock(return_value=_make_scheme("SCH-1"))), patch(
        "src.services.conversation.document_resolver.resolve_documents_for_scheme",
        AsyncMock(return_value=[]),
    ):
        result = await service.handle_message(
            ChatRequest(user_id="user-docs", message="show the required documents")
        )

    session = await store.get("user-docs")
    assert result.next_state == ConversationState.DOCUMENT_GUIDANCE.value
    assert "document" in result.text.lower()
    assert session is not None
    assert session.state == ConversationState.DOCUMENT_GUIDANCE


@pytest.mark.asyncio
async def test_switch_scheme_inside_details_uses_new_selection() -> None:
    """Selecting another scheme while already in DETAILS should update the active scheme."""
    store = get_session_store()
    seed = Session(
        user_id="user-switch",
        state=ConversationState.SCHEME_DETAILS,
        user_profile=UserProfile(
            life_event="EDUCATION",
            age=19,
            category="OBC",
            annual_income=400000,
        ),
        selected_scheme_id="SCH-1",
        presented_schemes=[
            {"id": "SCH-1", "name": "Scheme SCH-1", "name_hindi": "योजना SCH-1"},
            {"id": "SCH-2", "name": "Scheme SCH-2", "name_hindi": "योजना SCH-2"},
        ],
        language_preference="en",
    )
    await store.save(seed)

    service = ConversationService(db_pool=AsyncMock())
    service.llm.analyze_message = AsyncMock(
        return_value={
            "intent": "selection",
            "action": "switch_scheme",
            "life_event": None,
            "extracted_fields": {},
            "language": "en",
            "selected_scheme_id": None,
            "response_text": None,
        }
    )

    async def fake_scheme_lookup(pool, scheme_id):  # type: ignore[no-untyped-def]
        return _make_scheme(scheme_id)

    with patch("src.services.conversation.scheme_repo.get_scheme_by_id", AsyncMock(side_effect=fake_scheme_lookup)), patch(
        "src.services.conversation.document_resolver.resolve_documents_for_scheme",
        AsyncMock(return_value=[]),
    ), patch(
        "src.services.conversation.rejection_engine.get_rejection_warnings",
        AsyncMock(return_value=[]),
    ):
        result = await service.handle_message(
            ChatRequest(user_id="user-switch", message="2")
        )

    session = await store.get("user-switch")
    assert result.next_state == ConversationState.SCHEME_DETAILS.value
    assert "SCH-2" in result.text
    assert session is not None
    assert session.selected_scheme_id == "SCH-2"
