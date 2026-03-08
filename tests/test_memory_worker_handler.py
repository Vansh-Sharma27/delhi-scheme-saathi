"""Tests for the SQS-based working-memory Lambda handler."""

import json
from unittest.mock import AsyncMock, patch

from src.memory_worker_handler import handler
from src.services.ai_background import AIWorkType


def test_memory_worker_handler_processes_sqs_records() -> None:
    """Worker Lambda should deserialize queue records and process them."""
    process_item = AsyncMock()
    event = {
        "Records": [
            {
                "messageId": "msg-1",
                "body": json.dumps(
                    {
                        "work_type": "refresh_working_memory",
                        "user_id": "user-123",
                        "turn_count": 7,
                    }
                ),
            }
        ]
    }

    with patch("src.memory_worker_handler._configure_runtime"), patch(
        "src.memory_worker_handler.process_work_item",
        process_item,
    ):
        result = handler(event, context=None)

    assert result == {"batchItemFailures": []}
    item = process_item.await_args.args[0]
    assert item.work_type == AIWorkType.REFRESH_WORKING_MEMORY
    assert item.user_id == "user-123"
    assert item.turn_count == 7


def test_memory_worker_handler_reports_batch_failures() -> None:
    """Failed records should be returned for SQS retry instead of crashing the batch."""
    process_item = AsyncMock(side_effect=RuntimeError("boom"))
    event = {
        "Records": [
            {
                "messageId": "msg-2",
                "body": json.dumps(
                    {
                        "work_type": "refresh_working_memory",
                        "user_id": "user-456",
                        "turn_count": 11,
                    }
                ),
            }
        ]
    }

    with patch("src.memory_worker_handler._configure_runtime"), patch(
        "src.memory_worker_handler.process_work_item",
        process_item,
    ):
        result = handler(event, context=None)

    assert result == {"batchItemFailures": [{"itemIdentifier": "msg-2"}]}
