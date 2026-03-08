"""Regression tests for multi-turn conversation bugs seen in Telegram exports."""
from unittest.mock import AsyncMock, patch

import pytest

from src.db.session_store import InMemorySessionStore, configure_session_store, get_session_store
from src.models.api import ChatRequest
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
    with patch("src.services.conversation.scheme_repo.get_scheme_by_id", AsyncMock(return_value=scheme)), patch(
        "src.services.conversation.document_resolver.resolve_documents_for_scheme",
        AsyncMock(return_value=[]),
    ), patch(
        "src.services.conversation.rejection_engine.get_rejection_warnings",
        AsyncMock(return_value=[]),
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
