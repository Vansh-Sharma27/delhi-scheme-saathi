# Architecture Decision Records

Why things are the way they are. Each record states the problem, the decision,
what else was considered, and what it costs — so a decision does not have to be
reconstructed from the code months later.

Add one when a choice has real alternatives and a rationale that is not obvious
from reading the result. Skip it for anything a reader would infer anyway.
Superseded records stay in place and link to their replacement rather than being
edited or deleted.

Use [template.md](template.md) for new entries.

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](0001-namespace-chat-sessions.md) | Namespace /api/chat sessions rather than relying on authentication | accepted | 2026-07-26 |
| [0002](0002-redact-credentials-in-logging-layer.md) | Redact credentials in the logging layer, not at call sites | accepted | 2026-07-26 |
| [0003](0003-webhook-secret-in-secrets-manager.md) | Webhook secret in Secrets Manager, with no CloudFormation default | proposed | 2026-07-26 |
| [0004](0004-split-conversation-into-package.md) | Split conversation.py into a package with a named-phase pipeline | accepted | 2026-07-26 |
