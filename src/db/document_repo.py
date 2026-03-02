"""Document repository."""

import asyncpg

from src.models.document import Document


async def get_document_by_id(pool: asyncpg.Pool, doc_id: str) -> Document | None:
    """Get a single document by ID."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM documents WHERE id = $1",
            doc_id
        )
        if row:
            return Document.from_db_row(row)
    return None


async def get_documents_by_ids(pool: asyncpg.Pool, doc_ids: list[str]) -> list[Document]:
    """Get multiple documents by IDs."""
    if not doc_ids:
        return []
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM documents WHERE id = ANY($1)",
            doc_ids
        )
        return [Document.from_db_row(row) for row in rows]


async def get_all_documents(pool: asyncpg.Pool) -> list[Document]:
    """Get all documents."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM documents ORDER BY name")
        return [Document.from_db_row(row) for row in rows]


async def get_documents_for_scheme(pool: asyncpg.Pool, scheme_id: str) -> list[Document]:
    """Get all documents required for a scheme."""
    async with pool.acquire() as conn:
        # First get the scheme's document IDs
        scheme_row = await conn.fetchrow(
            "SELECT documents_required FROM schemes WHERE id = $1",
            scheme_id
        )
        if not scheme_row or not scheme_row["documents_required"]:
            return []

        doc_ids = list(scheme_row["documents_required"])
        rows = await conn.fetch(
            "SELECT * FROM documents WHERE id = ANY($1)",
            doc_ids
        )
        return [Document.from_db_row(row) for row in rows]


async def search_documents(pool: asyncpg.Pool, query: str, limit: int = 10) -> list[Document]:
    """Search documents by name."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT * FROM documents
            WHERE name ILIKE $1 OR name_hindi ILIKE $1
            ORDER BY name
            LIMIT $2
            """,
            f"%{query}%",
            limit
        )
        return [Document.from_db_row(row) for row in rows]
