# ADR-0002: Redact credentials in the logging layer, not at call sites

**Date**: 2026-07-26
**Status**: accepted
**Deciders**: Vansh Sharma

## Context

The Telegram bot token is part of every Telegram request URL
(`https://api.telegram.org/bot{token}/...`). When `raise_for_status()` fails,
httpx builds an `HTTPStatusError` whose message embeds the full request URL, and
`src/webhook/handler.py` logs that exception at ERROR with `exc_info=True`. A
single failed `sendMessage` — a blocked user, a 429, a malformed reply — writes
full bot credentials to the log sink, which on AWS is CloudWatch.

The leak is not careless logging. The call sites look correct; the credential
arrives inside an exception constructed elsewhere. Any future code that logs a
Telegram error has the same problem by default.

## Decision

A `RedactingFilter` in `src/utils/logging_config.py` scrubs every credential
configured in `Settings` from log records before a handler emits them, covering
both the interpolated message and the rendered `exc_info` traceback.
`configure_logging` attaches it to each root handler.

## Alternatives Considered

### Alternative 1: Fix the two call sites that leak
- **Pros**: smallest possible diff; obvious and local.
- **Cons**: addresses the symptom, not the cause.
- **Why not**: the next `logger.error("...: %s", exc)` on a Telegram path reintroduces it, and nothing signals that to the author. It also leaves the other eight configured credentials unprotected if they ever reach a log the same way.

### Alternative 2: Redact at the raise site inside telegram.py
- **Pros**: closest to where the token is known; no logging machinery involved.
- **Cons**: only covers Telegram, and only the exception message.
- **Why not**: with `exc_info=True` the formatter renders the traceback separately, so the URL still reaches the sink through that path. It would also mean wrapping every `raise_for_status()` call in the client.

### Alternative 3: Replace handler formatters with a redacting Formatter
- **Pros**: a formatter sees the final string including the traceback, so redaction is trivially complete.
- **Cons**: overwrites whatever formatter is already installed.
- **Why not**: AWS Lambda installs its own handlers and formatting before app import — the reason `configure_logging` exists in this shape. Replacing them would discard Lambda's request-ID context. A filter leaves formatting untouched.

## Consequences

### Positive
- Covers all nine credentials in `Settings`, and call sites written later that never think about it.
- Redaction is verified end to end: the log line that previously carried the token now reads `bot***REDACTED***`, traceback included.

### Negative
- Every log record pays a substring scan per configured secret. The set is small and cached with `lru_cache`.
- When a secret is found, the record's `args` are collapsed into the already-interpolated message, so structured-logging consumers would lose the argument split for those records only.

### Risks
- The filter must be attached to *handlers*, not to the root logger: a filter on a logger only sees records logged directly to it, not records propagated from module loggers, which is where nearly all output here originates. Attaching it in the wrong place would silently do nothing. `tests/test_logging_config.py` covers installation and both leak paths.
- Values shorter than eight characters are ignored deliberately, since a short key would match ordinary words and corrupt unrelated lines. A genuinely short credential would not be redacted.
- Secrets are read once and cached; tests that change the environment must call `_secret_values.cache_clear()` alongside `get_settings.cache_clear()`.
