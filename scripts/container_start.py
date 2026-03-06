"""Container entrypoint for local Docker runs."""

from __future__ import annotations

import asyncio
import os

from scripts.ensure_seed_data import ensure_seed_data


def main() -> None:
    """Optionally seed the local database, then exec uvicorn."""
    asyncio.run(ensure_seed_data())

    host = os.getenv("APP_HOST", "0.0.0.0")
    port = os.getenv("APP_PORT", "8000")
    os.execvp(
        "uvicorn",
        ["uvicorn", "src.main:app", "--host", host, "--port", port],
    )


if __name__ == "__main__":
    main()
