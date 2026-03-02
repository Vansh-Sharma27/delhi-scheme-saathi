"""Rejection rule repository."""

import asyncpg

from src.models.rejection_rule import RejectionRule


async def get_rules_by_scheme(
    pool: asyncpg.Pool,
    scheme_id: str
) -> list[RejectionRule]:
    """Get all rejection rules for a scheme, sorted by severity."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM rejection_rules
            WHERE scheme_id = $1
            ORDER BY
                CASE severity
                    WHEN 'critical' THEN 0
                    WHEN 'high' THEN 1
                    WHEN 'warning' THEN 2
                END,
                rule_type
            """,
            scheme_id
        )
        return [RejectionRule.from_db_row(row) for row in rows]


async def get_rules_by_ids(
    pool: asyncpg.Pool,
    rule_ids: list[str]
) -> list[RejectionRule]:
    """Get rejection rules by IDs."""
    if not rule_ids:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM rejection_rules
            WHERE id = ANY($1)
            ORDER BY
                CASE severity
                    WHEN 'critical' THEN 0
                    WHEN 'high' THEN 1
                    WHEN 'warning' THEN 2
                END
            """,
            rule_ids
        )
        return [RejectionRule.from_db_row(row) for row in rows]


async def get_critical_rules(
    pool: asyncpg.Pool,
    scheme_id: str
) -> list[RejectionRule]:
    """Get only critical severity rules for a scheme."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM rejection_rules
            WHERE scheme_id = $1 AND severity = 'critical'
            ORDER BY rule_type
            """,
            scheme_id
        )
        return [RejectionRule.from_db_row(row) for row in rows]


async def get_all_rules(pool: asyncpg.Pool) -> list[RejectionRule]:
    """Get all rejection rules."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM rejection_rules
            ORDER BY scheme_id,
                CASE severity
                    WHEN 'critical' THEN 0
                    WHEN 'high' THEN 1
                    WHEN 'warning' THEN 2
                END
            """
        )
        return [RejectionRule.from_db_row(row) for row in rows]
