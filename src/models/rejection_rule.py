"""Rejection rule data model."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RejectionRule(BaseModel, frozen=True):
    """Rejection rule for a scheme (immutable)."""

    id: str
    scheme_id: str
    rule_type: str  # validity, procedural, compliance, timeline, authority
    description: str
    description_hindi: str
    severity: str  # critical, high, warning
    prevention_tip: str
    examples: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_db_row(cls, row: Any) -> "RejectionRule":
        """Create from database row (asyncpg Record)."""
        return cls(
            id=row["id"],
            scheme_id=row["scheme_id"],
            rule_type=row["rule_type"],
            description=row["description"],
            description_hindi=row["description_hindi"],
            severity=row["severity"],
            prevention_tip=row["prevention_tip"],
            examples=list(row.get("examples") or []),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )

    @property
    def severity_order(self) -> int:
        """Numeric severity for sorting (lower = more severe)."""
        return {"critical": 0, "high": 1, "warning": 2}.get(self.severity, 3)
