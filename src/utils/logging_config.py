"""Shared logging configuration helpers."""

from __future__ import annotations

import logging
from functools import lru_cache

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

REDACTED = "***REDACTED***"

# Values shorter than this are too generic to substitute safely — a 4-character
# key would match ordinary words and corrupt unrelated log lines.
_MIN_SECRET_LENGTH = 8


@lru_cache(maxsize=1)
def _secret_values() -> tuple[str, ...]:
    """Configured credentials that must never reach a log sink.

    Read lazily so importing this module does not require loadable settings,
    and cached because it is consulted on every log record. Tests that change
    the environment should call ``_secret_values.cache_clear()`` alongside
    ``get_settings.cache_clear()``.
    """
    from src.config import get_settings

    settings = get_settings()
    candidates = (
        settings.telegram_bot_token,
        settings.telegram_webhook_secret,
        settings.chat_api_key,
        settings.xai_api_key,
        settings.jina_api_key,
        settings.voyage_api_key,
        settings.sarvam_api_key,
        settings.bhashini_api_key,
        settings.bhashini_ulca_api_key,
    )
    unique = {value for value in candidates if len(value) >= _MIN_SECRET_LENGTH}
    # Longest first, so a secret that contains a shorter one is replaced whole.
    return tuple(sorted(unique, key=len, reverse=True))


def redact(text: str) -> str:
    """Replace every configured secret occurring in ``text``."""
    for secret in _secret_values():
        text = text.replace(secret, REDACTED)
    return text


class RedactingFilter(logging.Filter):
    """Strip credentials from log records before a handler emits them.

    Secrets reach logs indirectly rather than through careless call sites. The
    Telegram bot token is part of every Telegram request URL, and httpx embeds
    the full URL in ``HTTPStatusError``; logging such an exception — as a
    ``%s`` argument or via ``exc_info`` — would otherwise write the token to
    CloudWatch. Filtering centrally covers both, and covers call sites added
    later that do not think about it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not _secret_values():
            return True

        message = record.getMessage()
        redacted = redact(message)
        if redacted != message:
            # Arguments are already interpolated into the redacted text.
            # Keeping them would reintroduce the secret at format time.
            record.msg = redacted
            record.args = ()

        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact(record.exc_text)

        return True


def install_redaction(handler: logging.Handler) -> None:
    """Attach the redaction filter to ``handler`` exactly once.

    Handler level rather than logger level: a filter on a logger only sees
    records logged directly to it, not records propagated from module loggers,
    which is where nearly all of this application's output originates.
    """
    if not any(isinstance(existing, RedactingFilter) for existing in handler.filters):
        handler.addFilter(RedactingFilter())


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
        install_redaction(handler)

    return level
