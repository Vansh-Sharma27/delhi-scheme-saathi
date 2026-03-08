"""AWS Lambda handler for SQS-driven working-memory refresh jobs."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from src.config import get_settings
from src.db.session_store import DynamoDBSessionStore, configure_session_store
from src.services.ai_background import deserialize_work_item, process_work_item

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_runtime_configured = False


def _configure_runtime() -> None:
    """Configure shared dependencies for the worker Lambda."""
    global _runtime_configured

    if _runtime_configured:
        return
    if not settings.session_table_name:
        raise RuntimeError("SESSION_TABLE_NAME is required for memory worker Lambda")

    configure_session_store(
        DynamoDBSessionStore(
            table_name=settings.session_table_name,
            region=settings.aws_region,
        )
    )
    _runtime_configured = True


async def _handle_event(event: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Process an SQS event and report any batch item failures."""
    batch_failures: list[dict[str, str]] = []

    for record in event.get("Records", []):
        message_id = record.get("messageId")
        try:
            payload = json.loads(record["body"])
            item = deserialize_work_item(payload)
            await process_work_item(item)
        except Exception as exc:
            logger.error(
                "Memory worker failed for message_id=%s: %s",
                message_id,
                exc,
                exc_info=True,
            )
            if message_id:
                batch_failures.append({"itemIdentifier": message_id})

    return {"batchItemFailures": batch_failures}


def handler(event: dict[str, Any], context: Any) -> dict[str, list[dict[str, str]]]:
    """Lambda entrypoint for SQS-triggered memory refresh jobs."""
    _configure_runtime()
    return asyncio.run(_handle_event(event))
