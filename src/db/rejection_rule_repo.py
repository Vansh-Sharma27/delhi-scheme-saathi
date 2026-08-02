"""Rejection rule repository.

Call sites as of 2026-07-26: only ``get_rules_by_scheme`` has any. It backs
``rejection_engine.get_rejection_warnings`` (the bot's rejection-warnings
view) and the ``GET /api/scheme/{id}`` endpoint.

The other three queries here are deliberately kept but currently uncalled.
They are not leftovers from a deleted feature — they round out the read
surface for rules, and each notes below what would use it. Check with
``rg 'get_rules_by_ids|get_critical_rules|get_all_rules'`` before assuming
this note is still accurate.
"""

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
    """Get rejection rules by IDs, most severe first.

    Currently uncalled. Its counterpart is ``Scheme.rejection_rules``
    (src/models/scheme.py), a list of rule IDs carried on every scheme row
    that nothing reads yet — fetching by those IDs is what this is for, and
    is cheaper than a second scheme-keyed query once a scheme is loaded.

    Its previous caller, ``rejection_engine.get_rules_for_scheme_ids``, was a
    pass-through wrapper with no callers of its own and was removed in the
    dead-code pass.
    """
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
    """Get only critical severity rules for a scheme.

    Currently uncalled. The rejection-warnings view fetches every rule and
    truncates to five in ``views.build_rejection_warnings_text``; this is the
    push-the-filter-into-SQL version, worth switching to if a scheme ever
    carries enough rules for that truncation to drop a critical one.
    """
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
    """Get every rejection rule, grouped by scheme and ordered by severity.

    Currently uncalled. Intended for bulk work rather than a request path —
    seed verification, or an admin/report view. Do not put this behind an API
    endpoint without a limit.
    """
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
