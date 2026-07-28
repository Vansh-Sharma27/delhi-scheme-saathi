"""Rejection rule engine for proactive warnings."""

import logging

import asyncpg

from src.db.rejection_rule_repo import get_rules_by_scheme
from src.models.rejection_rule import RejectionRule
from src.models.session import UserProfile

logger = logging.getLogger(__name__)


async def get_rejection_warnings(
    pool: asyncpg.Pool,
    scheme_id: str,
    profile: UserProfile | None = None,
) -> list[RejectionRule]:
    """Get rejection warnings for a scheme, most severe first.

    ``profile`` is accepted but not yet used: every rule for the scheme is
    returned and callers truncate the list when rendering. The parameter is
    kept so profile-aware filtering can be added without touching callers.
    """
    return await get_rules_by_scheme(pool, scheme_id)
