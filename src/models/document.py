"""Document data model."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel, frozen=True):
    """Required document for scheme application (immutable)."""

    id: str
    name: str
    name_hindi: str
    issuing_authority: str
    alternate_authority: str | None = None
    online_portal: str | None = None
    prerequisites: list[str] = Field(default_factory=list)  # Document IDs
    fee: str | None = None
    fee_bpl: str | None = None
    processing_time: str | None = None
    validity_period: str | None = None
    format_requirements: list[str] = Field(default_factory=list)
    common_mistakes: list[str] = Field(default_factory=list)
    delhi_offices: list[str] = Field(default_factory=list)  # Office IDs
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_db_row(cls, row: Any) -> "Document":
        """Create from database row (asyncpg Record)."""
        return cls(
            id=row["id"],
            name=row["name"],
            name_hindi=row["name_hindi"],
            issuing_authority=row["issuing_authority"],
            alternate_authority=row.get("alternate_authority"),
            online_portal=row.get("online_portal"),
            prerequisites=list(row.get("prerequisites") or []),
            fee=row.get("fee"),
            fee_bpl=row.get("fee_bpl"),
            processing_time=row.get("processing_time"),
            validity_period=row.get("validity_period"),
            format_requirements=list(row.get("format_requirements") or []),
            common_mistakes=list(row.get("common_mistakes") or []),
            delhi_offices=list(row.get("delhi_offices") or []),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )


class DocumentChain(BaseModel, frozen=True):
    """Document with resolved prerequisite chain."""

    document: Document
    prerequisites: list["DocumentChain"] = Field(default_factory=list)
    depth: int = 0

    @property
    def flat_list(self) -> list["Document"]:
        """Get flattened list of all documents in dependency order."""
        result: list[Document] = []
        for prereq in self.prerequisites:
            result.extend(prereq.flat_list)
        result.append(self.document)
        return result
