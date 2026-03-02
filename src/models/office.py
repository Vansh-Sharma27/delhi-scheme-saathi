"""Office data model."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Office(BaseModel, frozen=True):
    """Government office or CSC (immutable)."""

    id: str
    name: str
    type: str  # CSC, SDM, Tehsildar, District, Department
    address: str
    district: str
    pincode: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    phone: str | None = None
    working_hours: str | None = None
    services: list[str] = Field(default_factory=list)  # Document IDs
    fee_structure: dict[str, Any] = Field(default_factory=dict)
    operator_name: str | None = None
    last_verified: datetime | None = None
    distance_km: float | None = None  # Calculated field for nearest queries

    @classmethod
    def from_db_row(cls, row: Any, distance_km: float | None = None) -> "Office":
        """Create from database row (asyncpg Record)."""
        fee_structure = row.get("fee_structure") or {}
        if isinstance(fee_structure, str):
            import json
            fee_structure = json.loads(fee_structure)

        return cls(
            id=row["id"],
            name=row["name"],
            type=row["type"],
            address=row["address"],
            district=row["district"],
            pincode=row.get("pincode"),
            latitude=float(row["latitude"]) if row.get("latitude") else None,
            longitude=float(row["longitude"]) if row.get("longitude") else None,
            phone=row.get("phone"),
            working_hours=row.get("working_hours"),
            services=list(row.get("services") or []),
            fee_structure=fee_structure,
            operator_name=row.get("operator_name"),
            last_verified=row.get("last_verified"),
            distance_km=distance_km,
        )
