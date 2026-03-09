"""Shared logging configuration helpers."""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_logging(level_name: str) -> int:
    """Ensure the root logger emits at the requested level.

    AWS Lambda frequently installs handlers before app import time, which makes
    ``logging.basicConfig`` a no-op. We still need INFO-level telemetry like
    ``llm_usage`` to reach CloudWatch, so we explicitly raise the root logger
    and any pre-existing handlers to the requested level.
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, format=LOG_FORMAT)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)
    for handler in root_logger.handlers:
        handler.setLevel(level)
        if handler.formatter is None:
            handler.setFormatter(formatter)

    return level
