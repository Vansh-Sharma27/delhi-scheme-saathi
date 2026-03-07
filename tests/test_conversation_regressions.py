"""Regression tests for multi-turn conversation bugs seen in Telegram exports."""
from unittest.mock import AsyncMock, patch

import pytest

from src.db.session_store import InMemorySessionStore, configure_session_store, get_session_store
from src.models.api import ChatRequest
from src.models.session import ConversationState, Session, UserProfile
from src.models.scheme import EligibilityCriteria, Scheme, SchemeMatch
from src.services.conversation import (
    ConversationService,
    _infer_text_language,
    _resolve_scheme_from_text,
)
from src.services import session_manager


def _make_scheme(
    scheme_id: str,
    *,
    language: str = "en",
    life_event: str = "EDUCATION",
) -> Scheme:
    """Create a simple scheme object for plain-text rendering tests."""
    return Scheme(
        id=scheme_id,
        name=f"Scheme {scheme_id}",
        name_hindi=f"योजना {scheme_id}",
        department="Test Department",
        department_hindi="परीक्षण विभाग",
        level="state",
        benefits_amount=250000,
        benefits_frequency="one-time",
        eligibility=EligibilityCriteria(
            min_age=18,
            max_age=None,
            max_income=500000,
            categories=["OBC", "General"],
        ),
        life_events=[life_event],
        description="A test scheme description for verification.",
        description_hindi="सत्यापन के लिए परीक्षण योजना विवरण।",
        documents_required=[],
        application_url="https://example.com/apply",
        offline_process="Visit the district office",
    )


@pytest.fixture(autouse=True)
def reset_session_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a fresh in-memory session store and disable summarization noise."""
    configure_session_store(InMemorySessionStore())
    monkeypatch.setattr(session_manager, "SUMMARY_INTERVAL", 999)


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
async def test_unlocked_previous_language_does_not_bias_english_turn() -> None:
    """Observed language should drive unlocked sessions on later turns."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-unlocked-language",
            state=ConversationState.UNDERSTANDING,
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
    assert "category" in skipped.text.lower()
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
        state=ConversationState.DETAILS,
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
    assert result.next_state == ConversationState.DETAILS.value
    assert result.language == "hi"
    assert "आवेदन" not in result.text.splitlines()[0]
    assert session is not None
    assert session.state == ConversationState.DETAILS
    assert session.language_preference == "hi"
    assert session.language_locked is True


@pytest.mark.asyncio
async def test_start_over_action_resets_state_but_keeps_locked_language() -> None:
    """Natural-language restart requests should fully reset the flow."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-restart",
            state=ConversationState.DETAILS,
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


@pytest.mark.asyncio
async def test_no_match_guard_does_not_rerun_matching_without_profile_change() -> None:
    """No-match recovery should not loop into matching again on filler replies."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-no-match-guard",
            state=ConversationState.UNDERSTANDING,
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
    assert result.next_state == ConversationState.UNDERSTANDING.value
    assert "matching your profile" in result.text.lower()


@pytest.mark.asyncio
async def test_details_profile_change_clears_selection_and_rematches() -> None:
    """Eligibility changes inside details should invalidate the old scheme."""
    store = get_session_store()
    await store.save(
        Session(
            user_id="user-details-rematch",
            state=ConversationState.DETAILS,
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
    assert result.next_state == ConversationState.PRESENTING.value
    assert "SCH-2" in result.text
    assert session is not None
    assert session.selected_scheme_id is None
    assert session.presented_schemes[0]["id"] == "SCH-2"


def test_plain_english_scheme_message_stays_english() -> None:
    """The word 'scheme' alone should not force Hinglish auto-detection."""
    assert _infer_text_language("Can you explain this scheme in English?") == "en"


@pytest.mark.asyncio
async def test_no_match_topic_switch_updates_life_event_and_reruns_matching() -> None:
    """A new topic should replace the stored life event and retry matching."""
    store = get_session_store()
    seed = Session(
        user_id="user-topic",
        state=ConversationState.UNDERSTANDING,
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
    assert result.next_state == ConversationState.UNDERSTANDING.value
    assert match_schemes.await_count == 1
    called_profile = match_schemes.await_args.kwargs["profile"]
    assert called_profile.life_event == "HOUSING"
    assert session is not None
    assert session.user_profile.life_event == "HOUSING"
    assert session.awaiting_profile_change is True


@pytest.mark.asyncio
async def test_switch_scheme_inside_details_uses_new_selection() -> None:
    """Selecting another scheme while already in DETAILS should update the active scheme."""
    store = get_session_store()
    seed = Session(
        user_id="user-switch",
        state=ConversationState.DETAILS,
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
    assert result.next_state == ConversationState.DETAILS.value
    assert "SCH-2" in result.text
    assert session is not None
    assert session.selected_scheme_id == "SCH-2"
