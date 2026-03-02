"""API request and response models."""

from typing import Any

from pydantic import BaseModel, Field

from src.models.document import Document, DocumentChain
from src.models.office import Office
from src.models.rejection_rule import RejectionRule
from src.models.scheme import Scheme, SchemeMatch


class SchemeDetailResponse(BaseModel):
    """Full scheme details response."""

    scheme: Scheme
    documents: list[Document] = Field(default_factory=list)
    rejection_rules: list[RejectionRule] = Field(default_factory=list)
    nearest_offices: list[Office] = Field(default_factory=list)


class SchemeListResponse(BaseModel):
    """List of schemes matching criteria."""

    schemes: list[SchemeMatch]
    total: int
    life_event: str | None = None


class DocumentDetailResponse(BaseModel):
    """Document with procurement chain."""

    document: Document
    prerequisite_chain: list[DocumentChain] = Field(default_factory=list)
    procurement_steps: list[str] = Field(default_factory=list)


class NearestOfficesResponse(BaseModel):
    """Nearest offices response."""

    offices: list[Office]
    query_district: str | None = None
    query_location: tuple[float, float] | None = None


class ChatRequest(BaseModel):
    """Chat request from Telegram webhook."""

    user_id: str
    message: str
    message_type: str = "text"  # text, voice, callback
    callback_data: str | None = None
    voice_file_id: str | None = None
    latitude: float | None = None
    longitude: float | None = None


class ChatResponse(BaseModel):
    """Chat response to send back."""

    text: str
    text_hindi: str | None = None
    audio_url: str | None = None
    schemes: list[SchemeMatch] = Field(default_factory=list)
    documents: list[DocumentChain] = Field(default_factory=list)
    rejection_warnings: list[RejectionRule] = Field(default_factory=list)
    offices: list[Office] = Field(default_factory=list)
    inline_keyboard: list[list[dict[str, str]]] | None = None
    next_state: str | None = None


class TelegramUpdate(BaseModel):
    """Telegram webhook update payload."""

    update_id: int
    message: dict[str, Any] | None = None
    callback_query: dict[str, Any] | None = None

    @property
    def chat_id(self) -> int | None:
        """Extract chat ID from update."""
        if self.message:
            return self.message.get("chat", {}).get("id")
        if self.callback_query:
            return self.callback_query.get("message", {}).get("chat", {}).get("id")
        return None

    @property
    def user_id(self) -> str | None:
        """Extract user ID from update."""
        if self.message:
            return str(self.message.get("from", {}).get("id"))
        if self.callback_query:
            return str(self.callback_query.get("from", {}).get("id"))
        return None

    @property
    def text(self) -> str | None:
        """Extract text content from update."""
        if self.message:
            return self.message.get("text")
        if self.callback_query:
            return self.callback_query.get("data")
        return None

    @property
    def is_voice(self) -> bool:
        """Check if update contains voice message."""
        return bool(self.message and self.message.get("voice"))

    @property
    def voice_file_id(self) -> str | None:
        """Get voice file ID if present."""
        if self.message and self.message.get("voice"):
            return self.message["voice"].get("file_id")
        return None

    @property
    def is_callback(self) -> bool:
        """Check if update is callback query."""
        return self.callback_query is not None
