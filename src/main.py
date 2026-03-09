"""Delhi Scheme Saathi - FastAPI Application.

Voice-first Hindi chatbot helping Delhi residents discover
and apply for government welfare schemes.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.utils.logging_config import configure_logging

# Configure logging
settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

# Database connection pool (initialized on startup)
db_pool: asyncpg.Pool | None = None


async def init_db_pool() -> asyncpg.Pool:
    """Initialize the database connection pool."""
    return await asyncpg.create_pool(
        settings.database_url,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )


async def close_db_pool() -> None:
    """Close the database connection pool."""
    global db_pool
    if db_pool:
        await db_pool.close()
        db_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    global db_pool
    logger.info("Starting Delhi Scheme Saathi...")

    # Initialize database pool
    try:
        db_pool = await init_db_pool()
        logger.info("Database connection pool initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        db_pool = None
    else:
        try:
            from src.db.scheme_repo import get_scheme_debug_rows

            scheme_debug_rows = await get_scheme_debug_rows(
                db_pool,
                ["SCH-DELHI-001", "SCH-DELHI-006"],
            )
            for row in scheme_debug_rows:
                log_method = logger.warning if not row["life_events_match"] else logger.info
                log_method(
                    "Verified scheme row %s db_life_events=%s canonical_life_events=%s caste_categories=%s income_segments=%s",
                    row["id"],
                    row["life_events"],
                    row["canonical_life_events"],
                    row["caste_categories"],
                    row["income_segments"],
                )
        except Exception as e:
            logger.warning(f"Scheme verification logging failed: {e}")

    # Configure session store based on environment
    _configure_session_store()
    await _configure_ai_background_runtime()

    yield

    # Cleanup
    logger.info("Shutting down...")
    await _shutdown_ai_background_runtime()
    await close_db_pool()


def _configure_session_store() -> None:
    """Configure session store based on environment.

    Uses DynamoDB when SESSION_TABLE_NAME is set (production/Lambda),
    otherwise uses in-memory store (local development).
    """
    from src.db.session_store import (
        DynamoDBSessionStore,
        InMemorySessionStore,
        configure_session_store,
    )

    settings = get_settings()

    # Check if running in AWS Lambda or production environment
    if settings.session_table_name and settings.session_table_name != "dss-sessions":
        # DynamoDB configured (production)
        try:
            store = DynamoDBSessionStore(
                table_name=settings.session_table_name,
                region=settings.aws_region,
            )
            configure_session_store(store)
            logger.info(f"Session store: DynamoDB ({settings.session_table_name})")
        except Exception as e:
            logger.warning(f"DynamoDB init failed, using in-memory: {e}")
            configure_session_store(InMemorySessionStore())
    else:
        # Local development - use in-memory store
        configure_session_store(InMemorySessionStore())
        logger.info("Session store: In-memory (local development)")


async def _configure_ai_background_runtime() -> None:
    """Configure and start the background AI worker."""
    from src.services.ai_background import (
        InMemoryAIWorkQueue,
        configure_ai_work_queue,
        create_default_ai_work_queue,
        start_ai_background_worker,
    )

    queue = create_default_ai_work_queue()
    configure_ai_work_queue(queue)
    if queue is None:
        logger.info("AI background queue: disabled")
        return

    if isinstance(queue, InMemoryAIWorkQueue):
        await start_ai_background_worker()
        logger.info("AI background queue: %s (in-process worker started)", queue.__class__.__name__)
        return

    logger.info("AI background queue: %s (external worker expected)", queue.__class__.__name__)


async def _shutdown_ai_background_runtime() -> None:
    """Stop the background AI worker and release queue resources."""
    from src.services.ai_background import stop_ai_background_worker

    await stop_ai_background_worker()


# Create FastAPI app
app = FastAPI(
    title="Delhi Scheme Saathi",
    description="Voice-first Hindi chatbot for Delhi welfare schemes",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint returning database status and scheme count."""
    result: dict[str, Any] = {
        "status": "ok",
        "database": "disconnected",
        "schemes_count": 0,
    }

    if db_pool:
        try:
            async with db_pool.acquire() as conn:
                # Check database connectivity and get scheme count
                count = await conn.fetchval("SELECT COUNT(*) FROM schemes WHERE is_active = true")
                result["database"] = "connected"
                result["schemes_count"] = count or 0
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            result["database"] = f"error: {str(e)}"
            result["status"] = "degraded"

    return result


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint with API info."""
    return {
        "name": "Delhi Scheme Saathi API",
        "version": "0.1.0",
        "docs": "/docs",
    }


def get_db_pool() -> asyncpg.Pool:
    """Get the database connection pool.

    Raises HTTPException if pool is not initialized.
    """
    if not db_pool:
        raise HTTPException(
            status_code=503,
            detail="Database connection not available"
        )
    return db_pool


# =============================================================================
# REST API Endpoints
# =============================================================================

@app.get("/api/scheme/{scheme_id}")
async def get_scheme(scheme_id: str) -> dict[str, Any]:
    """Get full scheme details by ID."""
    from src.db import document_repo, rejection_rule_repo, scheme_repo

    pool = get_db_pool()
    scheme = await scheme_repo.get_scheme_by_id(pool, scheme_id)

    if not scheme:
        raise HTTPException(status_code=404, detail=f"Scheme {scheme_id} not found")

    # Get related documents and rejection rules
    documents = await document_repo.get_documents_for_scheme(pool, scheme_id)
    rejection_rules = await rejection_rule_repo.get_rules_by_scheme(pool, scheme_id)

    return {
        "scheme": scheme.model_dump(),
        "documents": [d.model_dump() for d in documents],
        "rejection_rules": [r.model_dump() for r in rejection_rules],
    }


@app.get("/api/schemes")
async def list_schemes(
    life_event: str | None = None,
    limit: int = 10
) -> dict[str, Any]:
    """List schemes, optionally filtered by life event."""
    from src.db import scheme_repo

    pool = get_db_pool()

    if life_event:
        schemes = await scheme_repo.get_schemes_by_life_event(pool, life_event, limit)
    else:
        schemes = await scheme_repo.get_all_schemes(pool)
        schemes = schemes[:limit]

    return {
        "schemes": [s.model_dump() for s in schemes],
        "total": len(schemes),
        "life_event": life_event,
    }


@app.get("/api/document/{document_id}")
async def get_document(document_id: str) -> dict[str, Any]:
    """Get document details with procurement guidance."""
    from src.db import document_repo, office_repo

    pool = get_db_pool()
    document = await document_repo.get_document_by_id(pool, document_id)

    if not document:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    # Get prerequisite documents
    prereq_docs = []
    if document.prerequisites:
        prereq_docs = await document_repo.get_documents_by_ids(pool, document.prerequisites)

    # Get offices that issue this document
    offices = await office_repo.get_offices_by_service(pool, document_id)

    return {
        "document": document.model_dump(),
        "prerequisites": [d.model_dump() for d in prereq_docs],
        "offices": [o.model_dump() for o in offices],
    }


@app.get("/api/csc/nearest")
async def get_nearest_offices(
    lat: float | None = None,
    lng: float | None = None,
    district: str | None = None,
    office_type: str | None = None,
    limit: int = 5
) -> dict[str, Any]:
    """Get nearest CSC/government offices."""
    from src.db import office_repo

    pool = get_db_pool()

    if lat is not None and lng is not None:
        offices = await office_repo.get_nearest_offices(
            pool, lat, lng, limit, office_type
        )
        query_type = "location"
    elif district:
        offices = await office_repo.get_offices_by_district(pool, district, limit)
        query_type = "district"
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either lat+lng or district parameter"
        )

    return {
        "offices": [o.model_dump() for o in offices],
        "total": len(offices),
        "query_type": query_type,
        "query_district": district,
        "query_location": (lat, lng) if lat and lng else None,
    }


@app.get("/api/life-events")
async def list_life_events() -> dict[str, Any]:
    """List all life event categories."""
    pool = get_db_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT key, display_name, display_name_hindi, aliases FROM life_events_taxonomy ORDER BY key"
        )

    return {
        "life_events": [
            {
                "key": row["key"],
                "display_name": row["display_name"],
                "display_name_hindi": row["display_name_hindi"],
                "aliases": list(row["aliases"] or []),
            }
            for row in rows
        ]
    }


# =============================================================================
# Telegram Webhook
# =============================================================================

@app.post("/webhook/telegram")
async def telegram_webhook(update: dict[str, Any]) -> dict[str, str]:
    """Handle incoming Telegram updates."""
    from src.webhook.handler import handle_telegram_update

    pool = get_db_pool()
    return await handle_telegram_update(update, pool)


# =============================================================================
# Chat API (for testing without Telegram)
# =============================================================================

@app.post("/api/chat")
async def chat_endpoint(request: dict[str, Any]) -> dict[str, Any]:
    """Direct chat endpoint for testing without Telegram.

    Request: {"user_id": "test123", "message": "Namaste"}
    Response: {"response": "...", "next_state": "...", "schemes": [...]}
    """
    from src.models.api import ChatRequest
    from src.services.conversation import ConversationService

    pool = get_db_pool()
    user_id = request.get("user_id", "test_user")
    message = request.get("message", "")

    chat_request = ChatRequest(
        user_id=user_id,
        message=message,
        language="auto"
    )

    service = ConversationService(pool)
    response = await service.handle_message(chat_request)

    return {
        "response": response.text,
        "next_state": response.next_state,
        "schemes": [s.model_dump() if hasattr(s, 'model_dump') else s for s in (response.schemes or [])],
        "documents": [d.model_dump() if hasattr(d, 'model_dump') else d for d in (response.documents or [])],
        "rejection_warnings": [r.model_dump() if hasattr(r, 'model_dump') else r for r in (response.rejection_warnings or [])],
    }
