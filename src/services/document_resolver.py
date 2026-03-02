"""Document resolver with recursive DFS for prerequisite chains."""

import logging
from typing import Any

import asyncpg

from src.db.document_repo import get_document_by_id, get_documents_by_ids
from src.models.document import Document, DocumentChain

logger = logging.getLogger(__name__)

# Maximum depth for prerequisite chain (prevent infinite loops)
MAX_CHAIN_DEPTH = 5


async def resolve_document_chain(
    pool: asyncpg.Pool,
    document_id: str,
    visited: set[str] | None = None,
    depth: int = 0,
) -> DocumentChain | None:
    """Resolve document with all prerequisites using DFS.

    Returns DocumentChain with nested prerequisites.
    """
    if visited is None:
        visited = set()

    # Prevent cycles and excessive depth
    if document_id in visited or depth > MAX_CHAIN_DEPTH:
        logger.warning(f"Cycle or max depth reached for document {document_id}")
        return None

    visited.add(document_id)

    # Get the document
    document = await get_document_by_id(pool, document_id)
    if not document:
        logger.warning(f"Document not found: {document_id}")
        return None

    # Recursively resolve prerequisites
    prereq_chains = []
    for prereq_id in document.prerequisites:
        prereq_chain = await resolve_document_chain(
            pool=pool,
            document_id=prereq_id,
            visited=visited.copy(),  # Copy to allow different paths
            depth=depth + 1,
        )
        if prereq_chain:
            prereq_chains.append(prereq_chain)

    return DocumentChain(
        document=document,
        prerequisites=prereq_chains,
        depth=depth,
    )


async def resolve_documents_for_scheme(
    pool: asyncpg.Pool,
    document_ids: list[str],
) -> list[DocumentChain]:
    """Resolve all documents required for a scheme with their prerequisites."""
    chains = []

    for doc_id in document_ids:
        chain = await resolve_document_chain(pool, doc_id)
        if chain:
            chains.append(chain)

    return chains


def get_procurement_order(chain: DocumentChain) -> list[Document]:
    """Get documents in order of procurement (prerequisites first)."""
    return chain.flat_list


def format_document_guide(
    chain: DocumentChain,
    language: str = "hi",
) -> dict[str, Any]:
    """Format document chain for display."""
    doc = chain.document

    # Get procurement steps
    steps = []
    all_docs = get_procurement_order(chain)

    for i, prereq_doc in enumerate(all_docs):
        if prereq_doc.id != doc.id:  # Don't include the main doc in prereqs
            step = {
                "order": i + 1,
                "document": prereq_doc.name_hindi if language == "hi" else prereq_doc.name,
                "where": prereq_doc.issuing_authority,
                "online": prereq_doc.online_portal,
                "fee": prereq_doc.fee,
                "time": prereq_doc.processing_time,
            }
            steps.append(step)

    return {
        "id": doc.id,
        "name": doc.name_hindi if language == "hi" else doc.name,
        "issuing_authority": doc.issuing_authority,
        "alternate_authority": doc.alternate_authority,
        "online_portal": doc.online_portal,
        "fee": doc.fee,
        "fee_bpl": doc.fee_bpl,
        "processing_time": doc.processing_time,
        "validity_period": doc.validity_period,
        "format_requirements": doc.format_requirements,
        "common_mistakes": doc.common_mistakes,
        "prerequisite_steps": steps,
        "total_prerequisites": len(steps),
    }


def generate_document_card(
    doc: Document,
    language: str = "hi",
) -> str:
    """Generate a formatted document card for Telegram."""
    if language == "hi":
        lines = [
            f"📄 *{doc.name_hindi}*",
            f"🏛️ कहाँ से: {doc.issuing_authority}",
        ]
        if doc.online_portal:
            lines.append(f"🌐 ऑनलाइन: {doc.online_portal}")
        if doc.fee:
            lines.append(f"💲 शुल्क: {doc.fee}")
            if doc.fee_bpl:
                lines.append(f"   (BPL: {doc.fee_bpl})")
        if doc.processing_time:
            lines.append(f"⏱️ समय: {doc.processing_time}")
        if doc.common_mistakes:
            lines.append(f"⚠️ ध्यान: {doc.common_mistakes[0]}")
    else:
        lines = [
            f"📄 *{doc.name}*",
            f"🏛️ Where: {doc.issuing_authority}",
        ]
        if doc.online_portal:
            lines.append(f"🌐 Online: {doc.online_portal}")
        if doc.fee:
            lines.append(f"💲 Fee: {doc.fee}")
        if doc.processing_time:
            lines.append(f"⏱️ Time: {doc.processing_time}")
        if doc.common_mistakes:
            lines.append(f"⚠️ Note: {doc.common_mistakes[0]}")

    return "\n".join(lines)
