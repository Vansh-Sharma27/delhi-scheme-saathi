# ADR-0003: Webhook secret in Secrets Manager, with no CloudFormation default

**Date**: 2026-07-26
**Status**: proposed
**Deciders**: Vansh Sharma

Not implemented. The project is pre-production and local-development only, so
the change is deliberately deferred. This records the shape it should take, so
the reasoning does not have to be reconstructed later.

## Context

`src/main.py` verifies `X-Telegram-Bot-Api-Secret-Token` only when
`TELEGRAM_WEBHOOK_SECRET` is non-empty. Neither `docker-compose.yml` nor
`sam-template.yaml` passes that variable to the container, so the check is
inactive everywhere and `/webhook/telegram` accepts any caller. Setting it in
`.env` alone changes nothing, because the app container never sees `.env`.

Closing this means two decisions: how the secret is supplied at deploy time, and
where it is stored.

## Decision

When this is implemented, the secret is stored in AWS Secrets Manager, and the
CloudFormation parameter carries **no default**, so a deploy that omits it fails
rather than silently producing an unprotected webhook.

## Alternatives Considered

### Alternative 1: CloudFormation parameter with `Default: ""`
- **Pros**: existing deploys and `scripts/rapid_redeploy.sh` keep working untouched.
- **Cons**: a deploy that forgets the value produces a stack that looks configured and is not.
- **Why not**: it reproduces the current vulnerability in a form that is harder to notice, since the template now mentions the secret. A missing security control should break the deploy loudly, not the security quietly.

### Alternative 2: CloudFormation parameter with `NoEcho: true`, no Secrets Manager
- **Pros**: matches how `DatabaseUrl`, `XaiApiKey` and `TelegramBotToken` are already handled; no new service.
- **Cons**: `NoEcho` masks the value in the console but it remains part of the stack.
- **Why not**: readable by anyone holding `cloudformation:GetTemplate` or `DescribeStacks`, and rotating it requires a redeploy. Consistency with the existing parameters is not worth extending a weaker pattern to a control whose whole purpose is authentication.

## Consequences

### Positive
- The secret is rotatable without a redeploy, and is not recoverable from stack metadata.
- A forgotten value surfaces at deploy time, when it is cheap to fix.

### Negative
- Every deploy path must supply the parameter, including `scripts/rapid_redeploy.sh`, which will need updating at the same time.
- Introduces a dependency on Secrets Manager and the IAM policy to read from it.

### Risks
- **Ordering is critical and one direction breaks the bot.** Once the env var is set, the app requires the header on every update, but Telegram only sends it if the secret was registered via `setWebhook(secret_token=...)`. Deploy first and every update returns 403 — the bot goes silently dead until the webhook is re-registered. Registering first is harmless, because the app ignores the header while the variable is unset. Safe order: register with Telegram, then deploy.
- `scripts/set_telegram_webhook.py` posts only `{"url": url}` and cannot send `secret_token`, even though `TelegramClient.set_webhook` accepts it. That script needs a `--secret` flag before any of this is possible.
- Two ways to wire the value remain open. A CloudFormation `{{resolve:secretsmanager:...}}` dynamic reference is simpler but resolves at deploy time and lands the value in a Lambda environment variable, readable via `lambda:GetFunctionConfiguration`. Fetching at runtime with boto3 keeps it out of the environment entirely, at the cost of an IAM policy and cold-start caching. Pick when implementing.
