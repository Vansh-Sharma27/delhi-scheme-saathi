"""Scheme repository with 3-stage hybrid search."""

import logging
from typing import Any

import asyncpg

from src.models.scheme import EligibilityCriteria, Scheme, SchemeMatch
from src.models.session import UserProfile
from src.utils.scheme_catalog import (
    get_canonical_life_events,
    get_canonical_scheme_ids_for_life_event,
)

logger = logging.getLogger(__name__)
INCOME_SEGMENT_ORDER = ("EWS", "LIG", "MIG", "HIG")


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
    canonical_ids = get_canonical_scheme_ids_for_life_event(life_event)
    async with pool.acquire() as conn:
        if canonical_ids:
            rows = await conn.fetch(
                """
                SELECT * FROM schemes
                WHERE is_active = true AND id = ANY($1::text[])
                ORDER BY benefits_amount DESC NULLS LAST
                LIMIT $2
                """,
                canonical_ids,
                limit,
            )
        else:
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
            canonical_ids = get_canonical_scheme_ids_for_life_event(life_event)
            if canonical_ids:
                conditions.append(f"id = ANY(${param_idx}::text[])")
                params.append(canonical_ids)
            else:
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


def _lookup_case_insensitive(mapping: dict[str, int], key: str) -> int | None:
    """Return the matching numeric value for a case-insensitive key."""
    target = key.upper()
    for candidate, value in mapping.items():
        if candidate.upper() == target:
            return value
    return None


def _infer_income_segment(
    annual_income: int,
    income_limits: dict[str, int],
) -> str | None:
    """Infer the user's income band from ordered segment thresholds."""
    normalized_limits: list[tuple[str, int]] = []
    for segment, raw_limit in income_limits.items():
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            continue
        normalized_limits.append((segment.upper(), limit))

    if not normalized_limits:
        return None

    ordered: list[tuple[str, int]] = []
    seen: set[str] = set()
    sorted_limits = sorted(normalized_limits, key=lambda item: item[1])
    for segment_name in INCOME_SEGMENT_ORDER:
        for segment, limit in sorted_limits:
            if segment == segment_name and segment not in seen:
                ordered.append((segment, limit))
                seen.add(segment)
    for segment, limit in sorted_limits:
        if segment not in seen:
            ordered.append((segment, limit))
            seen.add(segment)

    for segment, limit in ordered:
        if annual_income <= limit:
            return segment
    return None


def calculate_eligibility_match(scheme: Scheme, profile: UserProfile) -> dict[str, bool]:
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
    if profile.category is not None and elig.caste_categories:
        if not any(category.upper() == "ALL" for category in elig.caste_categories):
            match["category"] = (
                profile.category.upper() in [c.upper() for c in elig.caste_categories]
            )

    # Income check
    if profile.annual_income is not None:
        income_ok = True
        if elig.max_income is not None and profile.annual_income > elig.max_income:
            income_ok = False
        if elig.has_income_segment_restrictions and elig.income_by_category:
            inferred_segment = _infer_income_segment(
                profile.annual_income,
                elig.income_by_category,
            )
            allowed_segments = [segment.upper() for segment in elig.income_segments]
            segment_ok = inferred_segment in allowed_segments if inferred_segment else False
            match["income_segment"] = segment_ok
            income_ok = income_ok and segment_ok
        elif profile.category and elig.income_by_category:
            cat_limit = _lookup_case_insensitive(
                elig.income_by_category,
                profile.category,
            )
            if cat_limit is not None and profile.annual_income > cat_limit:
                income_ok = False
        match["income"] = income_ok

    return match


def _calculate_eligibility_match(scheme: Scheme, profile: UserProfile) -> dict[str, bool]:
    """Backward-compatible private alias."""
    return calculate_eligibility_match(scheme, profile)


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


async def get_scheme_debug_rows(
    pool: asyncpg.Pool,
    scheme_ids: list[str],
) -> list[dict[str, Any]]:
    """Fetch lightweight verification data for specific scheme rows."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, name, life_events, eligibility
            FROM schemes
            WHERE id = ANY($1::text[])
            ORDER BY id
            """,
            scheme_ids,
        )

    debug_rows: list[dict[str, Any]] = []
    for row in rows:
        eligibility = EligibilityCriteria.from_db(row.get("eligibility") or {})
        db_life_events = list(row.get("life_events") or [])
        canonical_life_events = get_canonical_life_events(row["id"])
        debug_rows.append(
            {
                "id": row["id"],
                "name": row["name"],
                "life_events": db_life_events,
                "canonical_life_events": canonical_life_events,
                "life_events_match": not canonical_life_events
                or sorted(db_life_events) == sorted(canonical_life_events),
                "raw_categories": eligibility.categories,
                "caste_categories": eligibility.caste_categories,
                "income_segments": eligibility.income_segments,
                "income_by_category": eligibility.income_by_category,
            }
        )

    if len(debug_rows) != len(scheme_ids):
        missing = sorted(set(scheme_ids) - {row["id"] for row in debug_rows})
        logger.warning("Missing scheme verification rows for: %s", ", ".join(missing))

    return debug_rows
