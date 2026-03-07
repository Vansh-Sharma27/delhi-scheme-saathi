"""Session and conversation state models."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ConversationState(str, Enum):
    """FSM states for conversation flow (7-state simplified model)."""

    GREETING = "GREETING"
    UNDERSTANDING = "UNDERSTANDING"  # Situation + profile collection
    MATCHING = "MATCHING"  # Transient: running scheme retrieval
    PRESENTING = "PRESENTING"  # Showing matched schemes
    DETAILS = "DETAILS"  # Deep dive: scheme + docs + rejections
    APPLICATION = "APPLICATION"  # Step-by-step application guidance
    HANDOFF = "HANDOFF"  # Connect to human/CSC


class UserProfile(BaseModel):
    """User profile extracted from conversation (mutable for updates)."""

    age: int | None = None
    gender: str | None = None  # male, female, other
    category: str | None = None  # SC, ST, OBC, General, EWS
    annual_income: int | None = None
    employment_status: str | None = None  # employed, unemployed, self-employed, student
    marital_status: str | None = None  # single, married, widowed, divorced, separated
    life_event: str | None = None  # HOUSING, HEALTH_CRISIS, etc.
    district: str | None = None
    has_bpl_card: bool | None = None
    disability_percentage: int | None = None
    latitude: float | None = None
    longitude: float | None = None

    def merge_with(self, other: "UserProfile") -> "UserProfile":
        """Create new profile merging non-None values from other (immutable merge)."""
        current_data = self.model_dump()
        other_data = other.model_dump()

        # Only override with non-None values
        merged = {
            k: other_data[k] if other_data[k] is not None else current_data[k]
            for k in current_data
        }
        return UserProfile(**merged)

    @property
    def is_complete_for_matching(self) -> bool:
        """Check if profile has minimum required fields for scheme matching."""
        return (
            self.life_event is not None
            and self.age is not None
            and self.category is not None
            and self.annual_income is not None
        )

    @property
    def completeness_score(self) -> int:
        """Score 0-10 indicating profile completeness."""
        score = 0
        if self.age is not None:
            score += 2
        if self.gender is not None:
            score += 1
        if self.category is not None:
            score += 2
        if self.annual_income is not None:
            score += 2
        if self.employment_status is not None:
            score += 1
        if self.life_event is not None:
            score += 2
        return min(score, 10)


class Message(BaseModel, frozen=True):
    """Single conversation message."""

    role: str  # user, assistant, system
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Session(BaseModel):
    """Conversation session (mutable for state updates)."""

    user_id: str
    state: ConversationState = ConversationState.GREETING
    user_profile: UserProfile = Field(default_factory=UserProfile)
    messages: list[Message] = Field(default_factory=list)
    conversation_summary: str | None = None
    discussed_schemes: list[str] = Field(default_factory=list)  # Scheme IDs
    selected_scheme_id: str | None = None
    language_preference: str = "auto"  # auto, hi, en
    language_locked: bool = False
    currently_asking: str | None = None
    skipped_fields: list[str] = Field(default_factory=list)
    awaiting_profile_change: bool = False
    presented_schemes: list[dict[str, str]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def copy_with(self, **updates: Any) -> "Session":
        """Return a deep-copied session with updated fields."""
        data = self.model_dump(round_trip=True)
        data.update(updates)
        data.setdefault("updated_at", datetime.utcnow())
        return Session(**data)

    def add_message(self, role: str, content: str) -> "Session":
        """Add message and return new session (keeping sliding window of 10)."""
        new_message = Message(role=role, content=content)
        messages = list(self.messages)
        messages.append(new_message)

        # Keep last 10 messages (sliding window)
        if len(messages) > 10:
            messages = messages[-10:]

        return self.copy_with(messages=messages, updated_at=datetime.utcnow())

    def with_state(self, new_state: ConversationState) -> "Session":
        """Return new session with updated state."""
        return self.copy_with(state=new_state, updated_at=datetime.utcnow())

    def with_profile(self, profile: UserProfile) -> "Session":
        """Return new session with updated profile."""
        return self.copy_with(user_profile=profile, updated_at=datetime.utcnow())

    def to_dynamodb_item(self) -> dict[str, Any]:
        """Serialize for DynamoDB storage."""
        return {
            "user_id": self.user_id,
            "state": self.state.value,
            "user_profile": self.user_profile.model_dump(),
            # DynamoDB serializer does not support datetime objects directly.
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                }
                for m in self.messages
            ],
            "conversation_summary": self.conversation_summary,
            "discussed_schemes": self.discussed_schemes,
            "selected_scheme_id": self.selected_scheme_id,
            "language_preference": self.language_preference,
            "language_locked": self.language_locked,
            "currently_asking": self.currently_asking,
            "skipped_fields": self.skipped_fields,
            "awaiting_profile_change": self.awaiting_profile_change,
            "presented_schemes": self.presented_schemes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "ttl": int(self.updated_at.timestamp()) + 86400 * 7,  # 7 days TTL
        }

    @classmethod
    def from_dynamodb_item(cls, item: dict[str, Any]) -> "Session":
        """Deserialize from DynamoDB item."""
        messages = [
            Message(
                role=m["role"],
                content=m["content"],
                timestamp=datetime.fromisoformat(m["timestamp"]) if isinstance(m["timestamp"], str) else m["timestamp"],
            )
            for m in item.get("messages", [])
        ]

        return cls(
            user_id=item["user_id"],
            state=ConversationState(item["state"]),
            user_profile=UserProfile(**item.get("user_profile", {})),
            messages=messages,
            conversation_summary=item.get("conversation_summary"),
            discussed_schemes=item.get("discussed_schemes", []),
            selected_scheme_id=item.get("selected_scheme_id"),
            language_preference=item.get("language_preference", "auto"),
            language_locked=item.get("language_locked", False),
            currently_asking=item.get("currently_asking", item.get("metadata", {}).get("currently_asking")),
            skipped_fields=item.get("skipped_fields", item.get("metadata", {}).get("skipped_fields", [])),
            awaiting_profile_change=item.get(
                "awaiting_profile_change",
                item.get("metadata", {}).get("awaiting_profile_change", False),
            ),
            presented_schemes=item.get(
                "presented_schemes",
                item.get("metadata", {}).get("presented_schemes", []),
            ),
            created_at=datetime.fromisoformat(item["created_at"]) if isinstance(item["created_at"], str) else item["created_at"],
            updated_at=datetime.fromisoformat(item["updated_at"]) if isinstance(item["updated_at"], str) else item["updated_at"],
            metadata=item.get("metadata", {}),
        )
