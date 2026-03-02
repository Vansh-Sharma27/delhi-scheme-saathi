#!/usr/bin/env python3
"""Generate embeddings for schemes using Voyage AI.

Usage:
    python scripts/generate_embeddings.py

Requires:
    - DATABASE_URL environment variable
    - VOYAGE_API_KEY environment variable
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    """Generate embeddings for all schemes."""
    from dotenv import load_dotenv
    load_dotenv()

    database_url = os.environ.get("DATABASE_URL")
    voyage_key = os.environ.get("VOYAGE_API_KEY")

    if not database_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)
    if not voyage_key:
        print("ERROR: VOYAGE_API_KEY not set")
        sys.exit(1)

    import asyncpg
    from src.integrations.embedding_client import get_embedding_client

    print("Connecting to database...")
    pool = await asyncpg.create_pool(dsn=database_url)

    print("Initializing embedding client...")
    client = get_embedding_client()

    # Get all schemes
    rows = await pool.fetch("SELECT id, description, description_hindi FROM schemes")
    print(f"Found {len(rows)} schemes\n")

    for row in rows:
        scheme_id = row["id"]
        # Combine English and Hindi descriptions for richer embedding
        text = f"{row['description']}\n\n{row['description_hindi']}"

        print(f"Generating embedding for {scheme_id}...")
        try:
            embedding = await client.get_embedding(text)

            # Format as PostgreSQL vector literal
            embedding_str = f"[{','.join(map(str, embedding))}]"

            await pool.execute(
                "UPDATE schemes SET description_embedding = $1::vector WHERE id = $2",
                embedding_str,
                scheme_id
            )
            print(f"  Updated with {len(embedding)}-dim embedding")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Verify
    count = await pool.fetchval(
        "SELECT COUNT(*) FROM schemes WHERE description_embedding IS NOT NULL"
    )
    print(f"\nDone! {count} schemes now have embeddings.")

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
