"""Office repository with nearest-office queries."""

import math

import asyncpg

from src.models.office import Office


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in kilometers using Haversine formula."""
    R = 6371  # Earth's radius in kilometers

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


async def get_office_by_id(pool: asyncpg.Pool, office_id: str) -> Office | None:
    """Get a single office by ID."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM offices WHERE id = $1",
            office_id
        )
        if row:
            return Office.from_db_row(row)
    return None


async def get_offices_by_district(
    pool: asyncpg.Pool,
    district: str,
    limit: int = 10
) -> list[Office]:
    """Get offices in a district."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM offices
            WHERE district ILIKE $1
            ORDER BY type, name
            LIMIT $2
            """,
            f"%{district}%",
            limit
        )
        return [Office.from_db_row(row) for row in rows]


async def get_nearest_offices(
    pool: asyncpg.Pool,
    latitude: float,
    longitude: float,
    limit: int = 5,
    office_type: str | None = None
) -> list[Office]:
    """Get nearest offices to a location (using Haversine in Python for MVP).

    For production, use PostGIS ST_Distance for better performance.
    """
    async with pool.acquire() as conn:
        query = """
            SELECT * FROM offices
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        """
        params = []

        if office_type:
            query += " AND type = $1"
            params.append(office_type)

        rows = await conn.fetch(query, *params)

        # Calculate distances and sort in Python (16 offices is small enough)
        offices_with_distance = []
        for row in rows:
            office_lat = float(row["latitude"])
            office_lon = float(row["longitude"])
            distance = haversine_distance(latitude, longitude, office_lat, office_lon)
            offices_with_distance.append((distance, row))

        # Sort by distance and take top N
        offices_with_distance.sort(key=lambda x: x[0])
        top_offices = offices_with_distance[:limit]

        return [
            Office.from_db_row(row, distance_km=round(distance, 2))
            for distance, row in top_offices
        ]


async def get_offices_by_service(
    pool: asyncpg.Pool,
    document_id: str,
    district: str | None = None,
    limit: int = 10
) -> list[Office]:
    """Get offices that provide a specific document service."""
    async with pool.acquire() as conn:
        query = """
            SELECT * FROM offices
            WHERE $1 = ANY(services)
        """
        params = [document_id]

        if district:
            query += " AND district ILIKE $2"
            params.append(f"%{district}%")

        query += " ORDER BY type, name LIMIT $" + str(len(params) + 1)
        params.append(limit)

        rows = await conn.fetch(query, *params)
        return [Office.from_db_row(row) for row in rows]


async def get_all_offices(pool: asyncpg.Pool) -> list[Office]:
    """Get all offices."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM offices ORDER BY district, name")
        return [Office.from_db_row(row) for row in rows]
