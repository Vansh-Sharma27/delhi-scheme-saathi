"""Tests for shared logging configuration."""

from __future__ import annotations

import logging
import sys

import httpx
import pytest

from src.utils import logging_config
from src.utils.logging_config import REDACTED, RedactingFilter, configure_logging

# Structurally a Telegram token but never a live one.
FAKE_BOT_TOKEN = "123456789:AA-fake-token-used-only-in-tests"


def _telegram_status_error() -> httpx.HTTPStatusError:
    """Build the exception a failing Telegram call actually raises.

    httpx puts the full request URL in the message, and the bot token is part
    of that URL — the mechanism this redaction exists to defeat.
    """
    request = httpx.Request(
        "POST", f"https://api.telegram.org/bot{FAKE_BOT_TOKEN}/sendMessage"
    )
    try:
        httpx.Response(400, request=request).raise_for_status()
    except httpx.HTTPStatusError as exc:
        return exc
    raise AssertionError("raise_for_status() did not raise")


def _record(msg: str, args: object = (), exc_info: object = None) -> logging.LogRecord:
    return logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=exc_info,
    )


@pytest.fixture
def only_bot_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat the fake bot token as the only configured secret."""
    monkeypatch.setattr(logging_config, "_secret_values", lambda: (FAKE_BOT_TOKEN,))


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


def test_filter_redacts_bot_token_from_exception_argument(only_bot_token: None) -> None:
    """`logger.error("...: %s", exc)` must not write the token to the sink."""
    record = _record("Failed to send Telegram message: %s", (_telegram_status_error(),))

    assert RedactingFilter().filter(record) is True

    message = record.getMessage()
    assert FAKE_BOT_TOKEN not in message
    assert REDACTED in message


def test_filter_redacts_bot_token_from_traceback(only_bot_token: None) -> None:
    """The exc_info=True path renders the exception separately and also leaks."""
    try:
        raise _telegram_status_error()
    except httpx.HTTPStatusError:
        record = _record("send failed", exc_info=sys.exc_info())

    assert RedactingFilter().filter(record) is True

    assert record.exc_text is not None
    assert FAKE_BOT_TOKEN not in record.exc_text
    assert REDACTED in record.exc_text


def test_filter_leaves_clean_records_untouched(only_bot_token: None) -> None:
    """Records without secrets keep their original msg and args for handlers."""
    record = _record("scheme matched for %s in %d ms", ("SCH-DELHI-001", 12))

    assert RedactingFilter().filter(record) is True

    assert record.msg == "scheme matched for %s in %d ms"
    assert record.args == ("SCH-DELHI-001", 12)
    assert record.getMessage() == "scheme matched for SCH-DELHI-001 in 12 ms"


def test_configure_logging_installs_one_redaction_filter() -> None:
    """Repeated setup must not stack duplicate filters on the same handler."""
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    test_handler = logging.StreamHandler()
    root_logger.addHandler(test_handler)

    try:
        configure_logging("INFO")
        configure_logging("INFO")

        redacting = [f for f in test_handler.filters if isinstance(f, RedactingFilter)]
        assert len(redacting) == 1
    finally:
        root_logger.removeHandler(test_handler)
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_level)


def test_secret_values_picks_up_configured_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The filter must read real settings, not just whatever tests inject."""
    from src.config import get_settings

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", FAKE_BOT_TOKEN)
    get_settings.cache_clear()
    logging_config._secret_values.cache_clear()

    try:
        assert FAKE_BOT_TOKEN in logging_config._secret_values()
        assert logging_config.redact(f"url=.../bot{FAKE_BOT_TOKEN}/send") == (
            f"url=.../bot{REDACTED}/send"
        )
    finally:
        get_settings.cache_clear()
        logging_config._secret_values.cache_clear()


def test_secret_values_ignores_short_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Short keys would match ordinary words and corrupt unrelated lines."""
    from src.config import get_settings

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "abc")
    get_settings.cache_clear()
    logging_config._secret_values.cache_clear()

    try:
        assert "abc" not in logging_config._secret_values()
    finally:
        get_settings.cache_clear()
        logging_config._secret_values.cache_clear()
