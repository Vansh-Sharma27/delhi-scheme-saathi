# Delhi Scheme Saathi: AWS Operator Handoff (1-Page)

Last updated: 2026-03-05 UTC

## Purpose

Use this checklist to resume deployment/testing quickly and safely.

## Current State

- AWS profile: `delhi-sso` (SSO-based)
- Region: `ap-south-1`
- Stack: `delhi-scheme-saathi`
- DB instance: `dss-postgres`
- Last known API endpoint: `https://jk6mrqfa2d.execute-api.ap-south-1.amazonaws.com/prod`
- Core local status: tests passing (`89 passed`)

## Resume in 5 Steps

1. Re-auth AWS SSO:
```bash
export AWS_PROFILE=delhi-sso
aws sso login --profile delhi-sso --use-device-code
aws sts get-caller-identity --profile delhi-sso
```

2. Start RDS and wait:
```bash
aws rds start-db-instance --db-instance-identifier dss-postgres --region ap-south-1 --profile delhi-sso
aws rds wait db-instance-available --db-instance-identifier dss-postgres --region ap-south-1 --profile delhi-sso
```

3. Redeploy + reset Telegram test session:
```bash
cd /home/ubuntu/ai-for-bharat/delhi-scheme-saathi
./scripts/rapid_redeploy.sh --user-id 780045592
```

4. Re-set webhook (if deleted):
```bash
curl -sS -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://jk6mrqfa2d.execute-api.ap-south-1.amazonaws.com/prod/webhook/telegram","drop_pending_updates":true}'
```

5. Smoke test:
```bash
API="https://jk6mrqfa2d.execute-api.ap-south-1.amazonaws.com/prod"
curl -sS "$API/health"
curl -sS -X POST "$API/api/chat" -H "Content-Type: application/json" -d '{"user_id":"test123","message":"Namaste"}'
```

## What Was Fixed

- SAM deployment blockers (template path, API Gateway logging role, reserved env var, rollback workflow)
- Lambda runtime startup (`Mangum` lifespan)
- Automatic SQS + worker Lambda deployment for async working-memory refresh jobs
- Bedrock model/profile configuration and IAM scope
- DynamoDB datetime serialization crash
- LLM fallback reliability (Bedrock -> Grok)
- FSM loop edge cases in `DETAILS`
- Category extraction false positives (`SC/ST` from substrings)
- Profile collection gating (match only after core fields)
- Telegram markdown artifact cleanup in plain-text sends

## Known Follow-up

- Parent/guardian flow is still single-profile.  
  If user says “for my son,” system now asks for applicant/beneficiary details, but there is no separate dependent profile model yet.

## Pause Again (Cost Save)

```bash
aws rds stop-db-instance --db-instance-identifier dss-postgres --region ap-south-1 --profile delhi-sso
curl -sS "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/deleteWebhook?drop_pending_updates=true"
```
