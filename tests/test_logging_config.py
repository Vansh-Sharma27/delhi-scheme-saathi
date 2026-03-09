"""Tests for shared logging configuration."""

from __future__ import annotations

import logging

from src.utils.logging_config import configure_logging


def test_configure_logging_updates_existing_root_handlers() -> None:
    """Preconfigured handlers should still emit INFO logs after setup."""
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    test_handler = logging.StreamHandler()
    test_handler.setLevel(logging.ERROR)
    root_logger.addHandler(test_handler)
    root_logger.setLevel(logging.WARNING)

    try:
        configure_logging("INFO")

        assert root_logger.level == logging.INFO
        assert test_handler.level == logging.INFO
        assert test_handler.formatter is not None
    finally:
        root_logger.removeHandler(test_handler)
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_level)
