"""Scheme repository with 3-stage hybrid search."""

from typing import Any

import asyncpg

from src.models.scheme import Scheme, SchemeMatch
from src.models.session import UserProfile


async def get_scheme_by_id(pool: asyncpg.Pool, scheme_id: str) -> Scheme | None:
    """Get a single scheme by ID."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM schemes WHERE id = $1 AND is_active = true",
            scheme_id
        )
        if row:
            return Scheme.from_db_row(row)
    return None


async def get_schemes_by_life_event(
    pool: asyncpg.Pool,
    life_event: str,
    limit: int = 10
) -> list[Scheme]:
    """Get schemes matching a life event."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM schemes
            WHERE is_active = true AND $1 = ANY(life_events)
            ORDER BY benefits_amount DESC NULLS LAST
            LIMIT $2
            """,
            life_event,
            limit
        )
        return [Scheme.from_db_row(row) for row in rows]


async def get_all_schemes(pool: asyncpg.Pool, active_only: bool = True) -> list[Scheme]:
    """Get all schemes."""
    async with pool.acquire() as conn:
        query = "SELECT * FROM schemes"
        if active_only:
            query += " WHERE is_active = true"
        query += " ORDER BY name"
        rows = await conn.fetch(query)
        return [Scheme.from_db_row(row) for row in rows]


async def hybrid_search(
    pool: asyncpg.Pool,
    life_event: str | None,
    profile: UserProfile,
    query_embedding: list[float] | None = None,
    limit: int = 5
) -> list[SchemeMatch]:
    """3-stage hybrid search for scheme matching.

    Stage 1: Filter by life event
    Stage 2: Filter by eligibility (age, income, category)
    Stage 3: Rank by vector similarity (if embedding provided)
    """
    async with pool.acquire() as conn:
        # Build dynamic query based on available filters
        params: list[Any] = []
        conditions = ["is_active = true"]
        param_idx = 1

        # Stage 1: Life event filter
        if life_event:
            conditions.append(f"${param_idx} = ANY(life_events)")
            params.append(life_event)
            param_idx += 1

        # Stage 2: Eligibility filters
        if profile.age is not None:
            # Age within range (NULL means no restriction)
            conditions.append(f"""
                ((eligibility->>'min_age')::int IS NULL OR (eligibility->>'min_age')::int <= ${param_idx})
                AND ((eligibility->>'max_age')::int IS NULL OR (eligibility->>'max_age')::int >= ${param_idx})
            """)
            params.append(profile.age)
            param_idx += 1

        if profile.annual_income is not None:
            # Income below max (NULL means no restriction)
            conditions.append(f"""
                (eligibility->>'max_income')::int IS NULL
                OR (eligibility->>'max_income')::int >= ${param_idx}
            """)
            params.append(profile.annual_income)
            param_idx += 1

        # Stage 3: Vector similarity (if embedding provided)
        if query_embedding and len(query_embedding) > 0:
            embedding_str = f"[{','.join(map(str, query_embedding))}]"
            order_by = f"description_embedding <=> '{embedding_str}'::vector"
            similarity_select = f", 1 - (description_embedding <=> '{embedding_str}'::vector) as similarity"
        else:
            order_by = "benefits_amount DESC NULLS LAST"
            similarity_select = ", 0.0 as similarity"

        where_clause = " AND ".join(conditions)
        query = f"""
            SELECT *{similarity_select}
            FROM schemes
            WHERE {where_clause}
            ORDER BY {order_by}
            LIMIT ${param_idx}
        """
        params.append(limit)

        rows = await conn.fetch(query, *params)

        # Build results with eligibility match details
        results = []
        for row in rows:
            scheme = Scheme.from_db_row(row)
            # Handle None similarity value
            sim_value = row.get("similarity")
            similarity = float(sim_value) if sim_value is not None else 0.0

            # Calculate eligibility match
            eligibility_match = _calculate_eligibility_match(scheme, profile)

            results.append(SchemeMatch(
                scheme=scheme,
                similarity=similarity,
                eligibility_match=eligibility_match
            ))

        return results


def _calculate_eligibility_match(scheme: Scheme, profile: UserProfile) -> dict[str, bool]:
    """Calculate which eligibility criteria the user matches."""
    match = {}
    elig = scheme.eligibility

    # Age check
    if profile.age is not None:
        age_ok = True
        if elig.min_age is not None and profile.age < elig.min_age:
            age_ok = False
        if elig.max_age is not None and profile.age > elig.max_age:
            age_ok = False
        match["age"] = age_ok

    # Gender check
    if profile.gender is not None:
        match["gender"] = (
            "all" in elig.genders
            or profile.gender.lower() in [g.lower() for g in elig.genders]
        )

    # Category check
    if profile.category is not None and elig.categories:
        match["category"] = profile.category.upper() in [c.upper() for c in elig.categories]

    # Income check
    if profile.annual_income is not None:
        income_ok = True
        if elig.max_income is not None and profile.annual_income > elig.max_income:
            income_ok = False
        # Check category-specific income limits
        if profile.category and elig.income_by_category:
            cat_limit = elig.income_by_category.get(profile.category.upper())
            if cat_limit and profile.annual_income > cat_limit:
                income_ok = False
        match["income"] = income_ok

    return match


async def search_schemes_by_text(
    pool: asyncpg.Pool,
    search_text: str,
    limit: int = 10
) -> list[Scheme]:
    """Simple text search in scheme names and descriptions."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM schemes
            WHERE is_active = true
              AND (
                name ILIKE $1 OR name_hindi ILIKE $1
                OR description ILIKE $1 OR description_hindi ILIKE $1
                OR $2 = ANY(tags)
              )
            ORDER BY benefits_amount DESC NULLS LAST
            LIMIT $3
            """,
            f"%{search_text}%",
            search_text.lower(),
            limit
        )
        return [Scheme.from_db_row(row) for row in rows]
