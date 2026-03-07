"""Verify critical scheme rows in the configured PostgreSQL database."""

from __future__ import annotations

import asyncio
import json
import os

import asyncpg

from src.db.scheme_repo import get_scheme_debug_rows

DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/delhi_scheme_saathi"
DEFAULT_SCHEME_IDS = ["SCH-DELHI-001", "SCH-DELHI-006"]


async def main() -> None:
    """Print normalized debug info for critical scheme rows."""
    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    pool = await asyncpg.create_pool(database_url, min_size=1, max_size=2)
    try:
        rows = await get_scheme_debug_rows(pool, DEFAULT_SCHEME_IDS)
    finally:
        await pool.close()

    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
