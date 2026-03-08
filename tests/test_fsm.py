"""Tests for FSM transitions."""

import pytest

from src.models.session import ConversationState, Session, UserProfile
from src.services.fsm import (
    FSMTransitionError,
    can_transition,
    determine_next_state,
    transition,
)


class TestFSMTransitions:
    """Tests for FSM state transitions."""

    def test_greeting_to_situation_understanding(self):
        assert can_transition(
            ConversationState.GREETING,
            ConversationState.SITUATION_UNDERSTANDING,
        )

    def test_situation_to_profile_collection(self):
        assert can_transition(
            ConversationState.SITUATION_UNDERSTANDING,
            ConversationState.PROFILE_COLLECTION,
        )

    def test_scheme_presentation_to_scheme_details(self):
        assert can_transition(
            ConversationState.SCHEME_PRESENTATION,
            ConversationState.SCHEME_DETAILS,
        )

    def test_scheme_details_to_application_help(self):
        assert can_transition(
            ConversationState.SCHEME_DETAILS,
            ConversationState.APPLICATION_HELP,
        )

    def test_invalid_greeting_to_scheme_details(self):
        assert not can_transition(
            ConversationState.GREETING,
            ConversationState.SCHEME_DETAILS,
        )

    def test_transition_updates_state(self):
        session = Session(user_id="user123", state=ConversationState.GREETING)
        new_session = transition(session, ConversationState.SITUATION_UNDERSTANDING)
        assert session.state == ConversationState.GREETING
        assert new_session.state == ConversationState.SITUATION_UNDERSTANDING

    def test_invalid_transition_raises(self):
        session = Session(user_id="user123", state=ConversationState.GREETING)
        with pytest.raises(FSMTransitionError):
            transition(session, ConversationState.SCHEME_DETAILS)


class TestDetermineNextState:
    """Tests for automatic state determination."""

    def test_greeting_stays_for_greeting_intent(self):
        next_state = determine_next_state(
            current_state=ConversationState.GREETING,
            profile=UserProfile(),
            intent="greeting",
        )
        assert next_state == ConversationState.GREETING

    def test_greeting_with_topic_moves_to_profile_collection(self):
        next_state = determine_next_state(
            current_state=ConversationState.GREETING,
            profile=UserProfile(life_event="HOUSING"),
            intent="question",
        )
        assert next_state == ConversationState.PROFILE_COLLECTION

    def test_situation_understanding_without_topic_stays(self):
        next_state = determine_next_state(
            current_state=ConversationState.SITUATION_UNDERSTANDING,
            profile=UserProfile(),
            intent="question",
        )
        assert next_state == ConversationState.SITUATION_UNDERSTANDING

    def test_profile_collection_with_complete_profile_triggers_matching(self):
        profile = UserProfile(
            life_event="HOUSING",
            age=28,
            category="SC",
            annual_income=300000,
        )
        next_state = determine_next_state(
            current_state=ConversationState.PROFILE_COLLECTION,
            profile=profile,
            intent="question",
        )
        assert next_state == ConversationState.SCHEME_MATCHING

    def test_scheme_matching_with_results_moves_to_presentation(self):
        next_state = determine_next_state(
            current_state=ConversationState.SCHEME_MATCHING,
            profile=UserProfile(life_event="HOUSING"),
            intent="question",
            has_schemes=True,
        )
        assert next_state == ConversationState.SCHEME_PRESENTATION

    def test_scheme_matching_without_results_returns_to_collection(self):
        next_state = determine_next_state(
            current_state=ConversationState.SCHEME_MATCHING,
            profile=UserProfile(life_event="HOUSING"),
            intent="question",
            has_schemes=False,
        )
        assert next_state == ConversationState.PROFILE_COLLECTION

    def test_scheme_presentation_with_selection_moves_to_details(self):
        next_state = determine_next_state(
            current_state=ConversationState.SCHEME_PRESENTATION,
            profile=UserProfile(life_event="HOUSING"),
            intent="selection",
            selected_scheme_id="SCH-001",
        )
        assert next_state == ConversationState.SCHEME_DETAILS

    def test_requested_document_view_is_respected(self):
        next_state = determine_next_state(
            current_state=ConversationState.SCHEME_DETAILS,
            profile=UserProfile(life_event="EDUCATION"),
            intent="question",
            has_selected_scheme=True,
            requested_state=ConversationState.DOCUMENT_GUIDANCE,
        )
        assert next_state == ConversationState.DOCUMENT_GUIDANCE

    def test_requested_rejection_view_is_respected(self):
        next_state = determine_next_state(
            current_state=ConversationState.DOCUMENT_GUIDANCE,
            profile=UserProfile(life_event="EDUCATION"),
            intent="question",
            has_selected_scheme=True,
            requested_state=ConversationState.REJECTION_WARNINGS,
        )
        assert next_state == ConversationState.REJECTION_WARNINGS

    def test_details_apply_request_moves_to_application_help(self):
        next_state = determine_next_state(
            current_state=ConversationState.SCHEME_DETAILS,
            profile=UserProfile(life_event="EDUCATION"),
            intent="question",
            action="request_application",
            has_selected_scheme=True,
        )
        assert next_state == ConversationState.APPLICATION_HELP

    def test_handoff_with_known_profile_returns_to_presentation(self):
        profile = UserProfile(
            life_event="HOUSING",
            age=28,
            category="OBC",
            annual_income=300000,
        )
        next_state = determine_next_state(
            current_state=ConversationState.CSC_HANDOFF,
            profile=profile,
            intent="question",
        )
        assert next_state == ConversationState.SCHEME_PRESENTATION

    def test_goodbye_resets_to_greeting(self):
        profile = UserProfile(life_event="HOUSING")
        for state in ConversationState:
            next_state = determine_next_state(
                current_state=state,
                profile=profile,
                intent="goodbye",
            )
            assert next_state == ConversationState.GREETING
