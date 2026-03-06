"""Ensure bundled JSON seed data is loaded for local Docker bootstraps."""

from __future__ import annotations

import asyncio
import os

import asyncpg

from scripts.seed_data import seed_database

DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/delhi_scheme_saathi"
AUTO_SEED_ENABLED_VALUES = {"1", "true", "yes", "on"}
READY_CHECK_RETRIES = 30
READY_CHECK_DELAY_SECONDS = 1.0


def auto_seed_enabled(env_value: str | None) -> bool:
    """Return whether AUTO_SEED_DATA is enabled."""
    if env_value is None:
        return False
    return env_value.strip().lower() in AUTO_SEED_ENABLED_VALUES


async def wait_for_database_ready(
    database_url: str,
    retries: int = READY_CHECK_RETRIES,
    delay_seconds: float = READY_CHECK_DELAY_SECONDS,
) -> None:
    """Wait until PostgreSQL is reachable and the schema is initialized."""
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            conn = await asyncpg.connect(database_url)
            try:
                schema_ready = await conn.fetchval(
                    "SELECT to_regclass('public.schemes') IS NOT NULL"
                )
            finally:
                await conn.close()

            if schema_ready:
                return

            last_error = RuntimeError("Database schema is not initialized yet")
        except Exception as exc:  # pragma: no cover - exercised via retry test seams
            last_error = exc

        if attempt < retries:
            print(
                f"Database not ready for seed check "
                f"(attempt {attempt}/{retries}); retrying..."
            )
            await asyncio.sleep(delay_seconds)

    raise RuntimeError("Database did not become ready for seeding") from last_error


async def get_scheme_count(database_url: str) -> int:
    """Return the number of seeded schemes in the database."""
    conn = await asyncpg.connect(database_url)
    try:
        count = await conn.fetchval("SELECT COUNT(*) FROM schemes")
    finally:
        await conn.close()
    return int(count or 0)


async def ensure_seed_data() -> bool:
    """Seed bundled local data exactly once when AUTO_SEED_DATA is enabled."""
    if not auto_seed_enabled(os.getenv("AUTO_SEED_DATA")):
        print("AUTO_SEED_DATA is disabled; skipping local seed bootstrap.")
        return False

    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    await wait_for_database_ready(database_url)

    scheme_count = await get_scheme_count(database_url)
    if scheme_count > 0:
        print(f"Database already seeded (schemes={scheme_count}); skipping.")
        return False

    print("Database is empty; loading bundled seed data...")
    await seed_database()
    return True


if __name__ == "__main__":
    asyncio.run(ensure_seed_data())
