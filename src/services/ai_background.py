"""Background queue and worker for non-urgent AI tasks."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

import boto3

from src.config import get_settings
from src.db.session_store import get_session_store

logger = logging.getLogger(__name__)


class AIWorkType(StrEnum):
    """Background AI work types."""

    REFRESH_WORKING_MEMORY = "refresh_working_memory"


@dataclass(slots=True)
class AIWorkItem:
    """Background AI work payload."""

    work_type: AIWorkType
    user_id: str
    turn_count: int
    enqueued_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    receipt_handle: str | None = None


class AIWorkQueue(Protocol):
    """Queue abstraction for async AI jobs."""

    async def enqueue(self, item: AIWorkItem) -> None:
        """Enqueue a work item."""

    async def dequeue(self) -> AIWorkItem | None:
        """Dequeue the next work item."""

    async def ack(self, item: AIWorkItem) -> None:
        """Acknowledge a completed work item."""

    async def close(self) -> None:
        """Close queue resources."""


class InMemoryAIWorkQueue:
    """In-process queue for local development and tests."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[AIWorkItem] = asyncio.Queue()

    async def enqueue(self, item: AIWorkItem) -> None:
        await self._queue.put(item)

    async def dequeue(self) -> AIWorkItem | None:
        return await self._queue.get()

    async def ack(self, item: AIWorkItem) -> None:
        self._queue.task_done()

    async def close(self) -> None:
        return None


class SQSAIWorkQueue:
    """SQS-backed queue for shared background AI jobs across instances."""

    def __init__(self, queue_url: str, region: str) -> None:
        self._queue_url = queue_url
        self._client = boto3.client("sqs", region_name=region)

    async def enqueue(self, item: AIWorkItem) -> None:
        body = json.dumps(serialize_work_item(item))
        await asyncio.to_thread(
            self._client.send_message,
            QueueUrl=self._queue_url,
            MessageBody=body,
        )

    async def dequeue(self) -> AIWorkItem | None:
        response = await asyncio.to_thread(
            self._client.receive_message,
            QueueUrl=self._queue_url,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
            VisibilityTimeout=60,
        )
        messages = response.get("Messages", [])
        if not messages:
            return None

        raw_message = messages[0]
        body = json.loads(raw_message["Body"])
        return deserialize_work_item(
            body,
            receipt_handle=raw_message.get("ReceiptHandle"),
        )

    async def ack(self, item: AIWorkItem) -> None:
        if not item.receipt_handle:
            return
        await asyncio.to_thread(
            self._client.delete_message,
            QueueUrl=self._queue_url,
            ReceiptHandle=item.receipt_handle,
        )

    async def close(self) -> None:
        return None


_ai_work_queue: AIWorkQueue | None = None
_worker_task: asyncio.Task[None] | None = None


def serialize_work_item(item: AIWorkItem) -> dict[str, str | int]:
    """Serialize a work item into a queue-safe payload."""
    return {
        "work_type": item.work_type.value,
        "user_id": item.user_id,
        "turn_count": item.turn_count,
        "enqueued_at": item.enqueued_at.isoformat(),
    }


def deserialize_work_item(
    payload: dict[str, object],
    *,
    receipt_handle: str | None = None,
) -> AIWorkItem:
    """Deserialize a queue payload into a typed work item."""
    enqueued_at_raw = payload.get("enqueued_at")
    if isinstance(enqueued_at_raw, str):
        enqueued_at = datetime.fromisoformat(enqueued_at_raw)
    else:
        enqueued_at = datetime.now(UTC)

    return AIWorkItem(
        work_type=AIWorkType(str(payload["work_type"])),
        user_id=str(payload["user_id"]),
        turn_count=int(payload.get("turn_count", 0)),
        enqueued_at=enqueued_at,
        receipt_handle=receipt_handle,
    )


def configure_ai_work_queue(queue: AIWorkQueue | None) -> None:
    """Set the active AI work queue."""
    global _ai_work_queue
    _ai_work_queue = queue


def get_ai_work_queue() -> AIWorkQueue | None:
    """Return the configured AI work queue."""
    return _ai_work_queue


def create_default_ai_work_queue() -> AIWorkQueue | None:
    """Create queue from settings."""
    settings = get_settings()
    if not settings.ai_memory_queue_enabled:
        return None
    if settings.ai_memory_queue_backend == "sqs" and settings.ai_memory_queue_url:
        return SQSAIWorkQueue(settings.ai_memory_queue_url, settings.aws_region)
    return InMemoryAIWorkQueue()


async def enqueue_memory_refresh(user_id: str, turn_count: int) -> bool:
    """Enqueue a working-memory refresh for a session."""
    queue = get_ai_work_queue()
    if queue is None:
        return False
    await queue.enqueue(
        AIWorkItem(
            work_type=AIWorkType.REFRESH_WORKING_MEMORY,
            user_id=user_id,
            turn_count=turn_count,
        )
    )
    return True


async def process_work_item(item: AIWorkItem) -> None:
    """Execute a background AI work item."""
    from src.services import session_manager
    from src.services.ai_orchestrator import get_ai_orchestrator

    store = get_session_store()
    session = await store.get(item.user_id)
    if session is None:
        return
    if item.work_type != AIWorkType.REFRESH_WORKING_MEMORY:
        return
    if not session.pending_memory_job:
        return
    queue_lag_ms = max(
        0.0,
        (datetime.now(UTC) - item.enqueued_at).total_seconds() * 1000,
    )

    try:
        memory = await get_ai_orchestrator().refresh_working_memory(
            session,
            queue_lag_ms=queue_lag_ms,
        )
        session = session_manager.apply_working_memory(
            session,
            memory,
            refreshed_turn=session.completed_turn_count,
        )
    except Exception as exc:
        logger.error(
            "Background memory refresh failed for user=%s turn=%s: %s",
            item.user_id,
            item.turn_count,
            exc,
            exc_info=True,
        )
        session = session_manager.set_pending_memory_job(session, False)
    else:
        session = session_manager.set_pending_memory_job(session, False)

    await store.save(session)


async def _worker_loop(queue: AIWorkQueue) -> None:
    """Continuously process background AI jobs."""
    while True:
        item = await queue.dequeue()
        if item is None:
            continue
        try:
            await process_work_item(item)
        finally:
            await queue.ack(item)


async def start_ai_background_worker() -> None:
    """Start the background AI worker if a queue is configured."""
    global _worker_task

    queue = get_ai_work_queue()
    if queue is None or _worker_task is not None:
        return
    _worker_task = asyncio.create_task(_worker_loop(queue), name="ai-background-worker")
    logger.info("Started AI background worker")


async def stop_ai_background_worker() -> None:
    """Stop the background AI worker and close queue resources."""
    global _worker_task

    if _worker_task is not None:
        _worker_task.cancel()
        with suppress(asyncio.CancelledError):
            await _worker_task
        _worker_task = None

    queue = get_ai_work_queue()
    if queue is not None:
        await queue.close()
    configure_ai_work_queue(None)
