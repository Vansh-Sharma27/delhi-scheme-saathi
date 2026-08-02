# ADR-0001: Namespace /api/chat sessions rather than relying on authentication

**Date**: 2026-07-26
**Status**: accepted
**Deciders**: Vansh Sharma

## Context

`POST /api/chat` takes `user_id` from an unauthenticated request body and
`session_manager.get_or_create_session` keys sessions on that value, in the same
store the Telegram webhook uses. The route is deployed, not local-only
(`sam-template.yaml`). Telegram user IDs are numeric and guessable, and one is
published as an example in `README.md`.

Anyone able to reach the endpoint could therefore resume a real applicant's
conversation: read back their matched schemes, and continue or corrupt the
profile extracted from it — age, income, caste category. The endpoint is
documented as a testing tool, but nothing confined it to test data.

## Decision

`/api/chat` prefixes every caller-supplied ID with `CHAT_SESSION_PREFIX`
(`api:`) before it reaches the session store, so its sessions occupy a separate
keyspace from Telegram's. An optional `CHAT_API_KEY`, checked against the
`X-API-Key` header with a constant-time comparison, gates the endpoint on top of
that.

## Alternatives Considered

### Alternative 1: Require an API key, keep the shared keyspace
- **Pros**: preserves the ability to drive any session, including a live user's, from the endpoint; single mechanism.
- **Cons**: protection depends entirely on deployment configuration being correct.
- **Why not**: `CHAT_API_KEY` defaults to empty, so this leaves the app one unset variable away from full exposure — the same fail-open shape as `TELEGRAM_WEBHOOK_SECRET`, which is the defect this work started from. Neither `docker-compose.yml` nor `sam-template.yaml` currently passes such values through, so the default is the realistic case.

### Alternative 2: Remove the route from sam-template.yaml, keep it local-only
- **Pros**: eliminates the internet-facing surface completely.
- **Cons**: loses the documented debugging path in any deployed environment.
- **Why not**: `CLAUDE.md` prescribes driving `POST /api/chat` over going through Telegram when reproducing conversation bugs. Deleting the deployed route removes that capability exactly where it is hardest to replace.

### Alternative 3: A separate session store for API traffic
- **Pros**: strongest possible isolation; no shared substrate at all.
- **Cons**: a second store to configure, seed and keep consistent across the local and AWS paths, which already do not share assumptions.
- **Why not**: delivers the same guarantee as a key prefix at meaningfully higher operational cost.

## Consequences

### Positive
- The cross-user read is structurally impossible, not merely disallowed. It holds even when `CHAT_API_KEY` is unset, which is the default.
- Namespacing and authentication are independent layers, so a configuration mistake degrades one without removing the other.
- Test traffic can no longer pollute production sessions, which was possible before even without malice.

### Negative
- A live Telegram session can no longer be inspected or driven through this endpoint. `scripts/fork_session.py` covers the legitimate need by copying a session into the `api:` keyspace for replay.
- The stored ID differs from the ID the caller supplied, which is mildly surprising when reading the session store directly.

### Risks
- Someone restores the old behaviour while debugging a user-reported issue, reintroducing the vulnerability. Mitigated by documenting the rationale in `docs/API.md`, in the endpoint docstring, and here, and by regression tests in `tests/test_main.py` asserting a Telegram ID never addresses a Telegram session.
- The endpoint remains unauthenticated by default, so it can still consume LLM and embedding quota. Rate limiting is not implemented and is tracked separately.
