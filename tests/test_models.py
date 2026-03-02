"""Tests for Pydantic models."""

import pytest

from src.models.scheme import EligibilityCriteria, Scheme
from src.models.session import ConversationState, Session, UserProfile


class TestUserProfile:
    """Tests for UserProfile model."""

    def test_create_empty_profile(self):
        """Test creating empty profile."""
        profile = UserProfile()
        assert profile.age is None
        assert profile.life_event is None
        assert profile.is_complete_for_matching is False

    def test_profile_with_life_event(self):
        """Test profile with life event is complete for matching."""
        profile = UserProfile(life_event="HOUSING")
        assert profile.is_complete_for_matching is True

    def test_profile_merge(self):
        """Test immutable profile merge."""
        profile1 = UserProfile(age=25, gender="male")
        profile2 = UserProfile(category="SC", annual_income=100000)

        merged = profile1.merge_with(profile2)

        # Original profiles unchanged
        assert profile1.category is None
        assert profile2.age is None

        # Merged has all values
        assert merged.age == 25
        assert merged.gender == "male"
        assert merged.category == "SC"
        assert merged.annual_income == 100000

    def test_profile_merge_override(self):
        """Test that merge only overrides with non-None values."""
        profile1 = UserProfile(age=25, gender="male", category="OBC")
        profile2 = UserProfile(age=30, category=None)

        merged = profile1.merge_with(profile2)

        assert merged.age == 30  # Overridden
        assert merged.category == "OBC"  # Not overridden (was None in profile2)

    def test_completeness_score(self):
        """Test completeness score calculation."""
        empty = UserProfile()
        assert empty.completeness_score == 0

        partial = UserProfile(age=25, life_event="HOUSING")
        assert partial.completeness_score == 4  # age=2 + life_event=2

        full = UserProfile(
            age=25,
            gender="male",
            category="SC",
            annual_income=100000,
            employment_status="employed",
            life_event="HOUSING",
        )
        assert full.completeness_score == 10


class TestSession:
    """Tests for Session model."""

    def test_create_session(self):
        """Test creating new session."""
        session = Session(user_id="user123")

        assert session.user_id == "user123"
        assert session.state == ConversationState.GREETING
        assert len(session.messages) == 0

    def test_add_message(self):
        """Test adding message to session."""
        session = Session(user_id="user123")
        new_session = session.add_message("user", "Hello")

        # Original unchanged
        assert len(session.messages) == 0

        # New session has message
        assert len(new_session.messages) == 1
        assert new_session.messages[0].content == "Hello"

    def test_sliding_window(self):
        """Test message sliding window (max 10)."""
        session = Session(user_id="user123")

        # Add 15 messages
        for i in range(15):
            session = session.add_message("user", f"Message {i}")

        # Only last 10 kept
        assert len(session.messages) == 10
        assert session.messages[0].content == "Message 5"
        assert session.messages[-1].content == "Message 14"

    def test_with_state(self):
        """Test immutable state update."""
        session = Session(user_id="user123")
        new_session = session.with_state(ConversationState.UNDERSTANDING)

        # Original unchanged
        assert session.state == ConversationState.GREETING

        # New session has new state
        assert new_session.state == ConversationState.UNDERSTANDING


class TestEligibilityCriteria:
    """Tests for EligibilityCriteria model."""

    def test_from_db_empty(self):
        """Test creating from empty dict."""
        criteria = EligibilityCriteria.from_db({})
        assert criteria.min_age is None
        assert criteria.genders == ["all"]

    def test_from_db_with_data(self):
        """Test creating from dict with data."""
        data = {
            "min_age": 18,
            "max_age": 60,
            "genders": ["female"],
            "categories": ["SC", "ST"],
            "max_income": 100000,
        }
        criteria = EligibilityCriteria.from_db(data)

        assert criteria.min_age == 18
        assert criteria.max_age == 60
        assert criteria.genders == ["female"]
        assert "SC" in criteria.categories


class TestConversationState:
    """Tests for ConversationState enum."""

    def test_all_states_exist(self):
        """Test all required states exist."""
        states = [
            ConversationState.GREETING,
            ConversationState.UNDERSTANDING,
            ConversationState.MATCHING,
            ConversationState.PRESENTING,
            ConversationState.DETAILS,
            ConversationState.APPLICATION,
            ConversationState.HANDOFF,
        ]
        assert len(states) == 7
