"""Tests for FSM transitions."""

import pytest

from src.models.session import ConversationState, Session, UserProfile
from src.services.fsm import (
    can_transition,
    determine_next_state,
    get_valid_transitions,
    transition,
    FSMTransitionError,
)


class TestFSMTransitions:
    """Tests for FSM state transitions."""

    def test_greeting_to_understanding(self):
        """Test GREETING -> UNDERSTANDING is valid."""
        assert can_transition(
            ConversationState.GREETING,
            ConversationState.UNDERSTANDING
        )

    def test_understanding_to_matching(self):
        """Test UNDERSTANDING -> MATCHING is valid."""
        assert can_transition(
            ConversationState.UNDERSTANDING,
            ConversationState.MATCHING
        )

    def test_invalid_greeting_to_details(self):
        """Test GREETING -> DETAILS is invalid."""
        assert not can_transition(
            ConversationState.GREETING,
            ConversationState.DETAILS
        )

    def test_presenting_to_details(self):
        """Test PRESENTING -> DETAILS (scheme selected) is valid."""
        assert can_transition(
            ConversationState.PRESENTING,
            ConversationState.DETAILS
        )

    def test_details_to_application(self):
        """Test DETAILS -> APPLICATION is valid."""
        assert can_transition(
            ConversationState.DETAILS,
            ConversationState.APPLICATION
        )

    def test_transition_updates_state(self):
        """Test transition() returns new session with updated state."""
        session = Session(user_id="user123", state=ConversationState.GREETING)
        new_session = transition(session, ConversationState.UNDERSTANDING)

        # Original unchanged
        assert session.state == ConversationState.GREETING

        # New session has new state
        assert new_session.state == ConversationState.UNDERSTANDING

    def test_invalid_transition_raises(self):
        """Test invalid transition raises FSMTransitionError."""
        session = Session(user_id="user123", state=ConversationState.GREETING)

        with pytest.raises(FSMTransitionError):
            transition(session, ConversationState.DETAILS)


class TestDetermineNextState:
    """Tests for automatic state determination."""

    def test_greeting_to_understanding(self):
        """Test greeting intent transitions to understanding."""
        profile = UserProfile()
        next_state = determine_next_state(
            current_state=ConversationState.GREETING,
            profile=profile,
            intent="greeting",
        )
        assert next_state == ConversationState.UNDERSTANDING

    def test_understanding_with_life_event_triggers_matching(self):
        """Test understanding with complete profile triggers matching."""
        profile = UserProfile(life_event="HOUSING")
        next_state = determine_next_state(
            current_state=ConversationState.UNDERSTANDING,
            profile=profile,
            intent="question",
        )
        assert next_state == ConversationState.MATCHING

    def test_matching_with_schemes_to_presenting(self):
        """Test matching with schemes transitions to presenting."""
        profile = UserProfile(life_event="HOUSING")
        next_state = determine_next_state(
            current_state=ConversationState.MATCHING,
            profile=profile,
            intent="question",
            has_schemes=True,
        )
        assert next_state == ConversationState.PRESENTING

    def test_matching_without_schemes_to_handoff(self):
        """Test matching without schemes transitions to handoff."""
        profile = UserProfile(life_event="HOUSING")
        next_state = determine_next_state(
            current_state=ConversationState.MATCHING,
            profile=profile,
            intent="question",
            has_schemes=False,
        )
        assert next_state == ConversationState.HANDOFF

    def test_presenting_with_selection_to_details(self):
        """Test presenting with scheme selection transitions to details."""
        profile = UserProfile(life_event="HOUSING")
        next_state = determine_next_state(
            current_state=ConversationState.PRESENTING,
            profile=profile,
            intent="selection",
            selected_scheme_id="SCH-001",
        )
        assert next_state == ConversationState.DETAILS

    def test_goodbye_resets_to_greeting(self):
        """Test goodbye intent always goes to greeting."""
        profile = UserProfile(life_event="HOUSING")

        for state in ConversationState:
            next_state = determine_next_state(
                current_state=state,
                profile=profile,
                intent="goodbye",
            )
            assert next_state == ConversationState.GREETING
