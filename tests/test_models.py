"""Tests for Pydantic models."""

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

    def test_profile_with_only_life_event_not_complete(self):
        """Matching requires core eligibility fields, not just life event."""
        profile = UserProfile(life_event="HOUSING")
        assert profile.is_complete_for_matching is False

    def test_profile_complete_for_matching(self):
        """Profile is complete when required matching fields are present."""
        profile = UserProfile(
            life_event="HOUSING",
            age=30,
            annual_income=250000,
        )
        assert profile.is_complete_for_matching is True

    def test_widow_profile_does_not_require_category_for_matching(self):
        """Widow/death-in-family flows should not block on caste category."""
        profile = UserProfile(
            life_event="DEATH_IN_FAMILY",
            age=23,
            gender="female",
            annual_income=50000,
        )
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
        """Test message sliding window (max 12 / last 6 turns)."""
        session = Session(user_id="user123")

        # Add 15 messages
        for i in range(15):
            session = session.add_message("user", f"Message {i}")

        # Only last 12 kept
        assert len(session.messages) == 12
        assert session.messages[0].content == "Message 3"
        assert session.messages[-1].content == "Message 14"

    def test_with_state(self):
        """Test immutable state update."""
        session = Session(user_id="user123")
        new_session = session.with_state(ConversationState.SITUATION_UNDERSTANDING)

        # Original unchanged
        assert session.state == ConversationState.GREETING

        # New session has new state
        assert new_session.state == ConversationState.SITUATION_UNDERSTANDING

    def test_to_dynamodb_item_serializes_message_timestamps(self):
        """Test DynamoDB serialization converts message datetime fields to strings."""
        session = Session(user_id="user123").add_message("user", "Hello")
        item = session.to_dynamodb_item()

        assert "messages" in item
        assert isinstance(item["messages"][0]["timestamp"], str)


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
        assert criteria.caste_categories == ["SC", "ST"]
        assert criteria.income_segments == []

    def test_income_segments_are_split_from_categories(self):
        """Income-band labels should not be treated as caste categories."""
        criteria = EligibilityCriteria.from_db(
            {
                "categories": ["EWS", "LIG", "MIG"],
                "income_by_category": {
                    "EWS": 300000,
                    "LIG": 600000,
                    "MIG": 900000,
                },
            }
        )
        assert criteria.categories == ["EWS", "LIG", "MIG"]
        assert criteria.caste_categories == []
        assert criteria.income_segments == ["EWS", "LIG", "MIG"]

    def test_all_category_does_not_create_caste_restriction(self):
        """A raw 'all' category means unrestricted, not caste='ALL'."""
        criteria = EligibilityCriteria.from_db({"categories": ["all"]})
        assert criteria.categories == ["all"]
        assert criteria.caste_categories == []
        assert criteria.income_segments == []

    def test_scheme_from_db_row_uses_canonical_life_events(self):
        """Bundled scheme metadata should override stale DB life-event tags."""
        scheme = Scheme.from_db_row(
            {
                "id": "SCH-DELHI-001",
                "name": "PMAY-U 2.0",
                "name_hindi": "पीएमएवाई-यू 2.0",
                "department": "Housing",
                "department_hindi": "आवास",
                "level": "central",
                "description": "Housing scheme",
                "description_hindi": "आवास योजना",
                "eligibility": {},
                "life_events": ["HOUSING"],
                "tags": ["housing"],
                "documents_required": [],
                "rejection_rules": [],
            }
        )

        assert "CHILDBIRTH" in scheme.life_events
        assert "MARRIAGE" in scheme.life_events

    def test_session_deserializes_legacy_understanding_state(self):
        """Legacy UNDERSTANDING sessions should map to the correct new state."""
        item = Session(user_id="legacy-user").to_dynamodb_item()
        item["state"] = "UNDERSTANDING"
        item["user_profile"] = {"life_event": "HOUSING"}

        session = Session.from_dynamodb_item(item)

        assert session.state == ConversationState.PROFILE_COLLECTION

    def test_session_deserializes_legacy_conversation_summary_into_working_memory(self):
        """Legacy summary storage should hydrate the new working-memory field."""
        item = Session(user_id="legacy-memory").to_dynamodb_item()
        item.pop("working_memory", None)
        item["conversation_summary"] = "User needs housing help and has low income."

        session = Session.from_dynamodb_item(item)

        assert session.working_memory.summary == "User needs housing help and has low income."


class TestConversationState:
    """Tests for ConversationState enum."""

    def test_all_states_exist(self):
        """Test all required states exist."""
        states = [
            ConversationState.GREETING,
            ConversationState.SITUATION_UNDERSTANDING,
            ConversationState.PROFILE_COLLECTION,
            ConversationState.SCHEME_MATCHING,
            ConversationState.SCHEME_PRESENTATION,
            ConversationState.SCHEME_DETAILS,
            ConversationState.DOCUMENT_GUIDANCE,
            ConversationState.REJECTION_WARNINGS,
            ConversationState.APPLICATION_HELP,
            ConversationState.CSC_HANDOFF,
        ]
        assert len(states) == 10
