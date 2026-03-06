# Delhi Scheme Saathi AWS Deployment Brief

Last updated: 2026-03-05 18:00 UTC

## 1) Goal

Deploy `delhi-scheme-saathi` on AWS with:
- RDS PostgreSQL (Mumbai region)
- SAM stack (Lambda + API Gateway + DynamoDB + S3 + CloudWatch)
- Telegram webhook integration
- Bedrock Nova 2 Lite as primary LLM, Grok as fallback

---

## 2) What Happened (Chronological)

## 2.1 Infrastructure setup and early blockers

1. AWS profile and access worked with `delhi-sso`.
2. RDS creation failed with engine version `16.1`:
   - Error: `Cannot find version 16.1 for postgres`
   - Fix: use an available version from `describe-db-engine-versions` (for this region, 16.x versions like `16.6+` were available).
3. RDS password validation failed:
   - Error: `MasterUserPassword ... besides '/', '@', '"', ' '`
   - Fix: use password without forbidden characters.
4. `psql` connection string error:
   - Used `@https://...` in host part, causing host resolution failure on `https`.
   - Fix: use plain host, no protocol in the DB host segment.

## 2.2 SAM build and packaging issues

1. `sam build` failed because default template name was wrong:
   - Error: `template.yml not found`
   - Fix: `sam build -t sam-template.yaml`
2. Runtime mismatch:
   - Error: Python 3.11 not found for Lambda build.
   - Fix: install Python 3.11 and recreate venv.
3. Dependency resolution failed during host build:
   - Error around `numpy==2.4.2(wheel)`.
   - Fix: build in container: `sam build -t sam-template.yaml --use-container`.
4. Large package rollback:
   - Error: Lambda unzipped size > 262144000 bytes.
   - Fix: deploy from built template (`.aws-sam/build/template.yaml`) and avoid packaging local venv/artifacts.

## 2.3 SAM deploy failures and fixes

1. Lambda env var failure:
   - Error: reserved key `AWS_REGION` cannot be set.
   - Fix: removed `AWS_REGION` from function env variables in SAM template.
2. Stack update blocked:
   - Error: stack in `ROLLBACK_COMPLETE` cannot be updated.
   - Fix: delete stack, then redeploy.
3. API Gateway stage creation failed:
   - Error: CloudWatch Logs role ARN must be set.
   - Fix: added API Gateway account-level CloudWatch role resources in template.

## 2.4 Runtime failures after successful stack creation

1. Health endpoint showed DB disconnected:
   - Cause: FastAPI startup was not running under Mangum.
   - Fix: set `Mangum(app, lifespan="auto")`.
2. Bedrock model invocation issue:
   - Error: Nova 2 Lite on-demand call unsupported; required inference profile.
   - Fix: support `BEDROCK_MODEL` as profile ID/ARN, and set to inference profile.
3. Bedrock IAM access denied:
   - Error: Lambda role not authorized for `bedrock:InvokeModel`.
   - Fix: broadened IAM resource scope to foundation-model and inference-profile ARNs.
4. DynamoDB session save crash:
   - Error: unsupported `datetime.datetime` type.
   - Fix: serialize message timestamps to ISO strings before `put_item`.
5. JSON serialization errors in LLM path:
   - Error: `Object of type datetime is not JSON serializable`.
   - Fix: use `json.dumps(..., default=str)` in Bedrock and Grok clients.

## 2.5 Conversation quality bugs identified from DynamoDB sessions

1. Looping in scheme details:
   - Symptom: repeated "Which scheme would you like to know more about?"
   - Cause: FSM moved `DETAILS -> PRESENTING` on selection intent without new ID.
   - Fix: keep `DETAILS` when session already has selected scheme.
2. Wrong category extraction (critical):
   - Symptom: category changed to `SC`/`ST` without explicit user statement.
   - Cause: substring matching (`"sc"` in "schemes", `"st"` in "study").
   - Fix: regex word-boundary/phrase category matching.
3. Profile collection skipped too early:
   - Symptom: bot stopped asking profile questions and jumped to matching.
   - Cause: matching considered complete with life_event only.
   - Fix: require core fields for matching (`life_event + age + category + annual_income`).
4. Assistant output polluted extraction:
   - Symptom: previously generated assistant text influenced field extraction.
   - Fix: use user-only conversation history for analysis extraction calls.
5. Telegram raw markdown artifacts:
   - Symptom: `###`, `**` visible in bot messages.
   - Fix: normalize markdown artifacts before sending Telegram text.
6. Beneficiary wording confusion:
   - Symptom: "your age" asked when user said "for my son".
   - Fix: changed prompts to "applicant/beneficiary age/category/income".

---

## 3) Code and Template Changes Made

## 3.1 SAM template (`sam-template.yaml`)
- Added missing API events (`/api/life-events`, `/api/chat`).
- Removed reserved env var injection (`AWS_REGION`).
- Added `BedrockModel` parameter and `BEDROCK_MODEL` env variable.
- Updated Bedrock IAM permissions for model/inference-profile resources.
- Added API Gateway CloudWatch role/account resources.
- Set `UseBedrock` default to `true`.

## 3.2 Runtime and integrations
- `src/lambda_handler.py`: enabled lifespan startup.
- `src/models/session.py`: DynamoDB-safe timestamp serialization; stricter matching completeness.
- `src/integrations/bedrock_client.py`: safe JSON serialization for non-primitive values.
- `src/integrations/grok_client.py`: safe JSON serialization and proper exception re-raise for fallback handling.
- `src/integrations/embedding_client.py` + `src/services/scheme_matcher.py`:
  - safer fallback behavior when embeddings fail/unavailable.
- `src/services/session_manager.py` + `src/services/conversation.py`:
  - user-only analysis history
  - deterministic extraction and life-event fallback
  - explicit full session reset on goodbye
  - improved state transition inputs.
- `src/services/fsm.py`: fixed MATCHING and DETAILS transition edge cases.
- `src/services/profile_extractor.py`: fixed category regex and beneficiary-focused questions.
- `src/webhook/handler.py`: text cleanup before Telegram send.

## 3.3 Dependencies and docs
- Removed `voyageai` package from `requirements.txt` and `pyproject.toml` (not needed by current integration path).
- Added rapid redeploy workflow docs in README.

## 3.4 Automation added
- New script: `delhi-scheme-saathi/scripts/rapid_redeploy.sh`
  - Rebuild + redeploy SAM
  - Resolve session table from stack outputs
  - Delete one DynamoDB session by Telegram user id
  - Verify deletion
  - Optional health check

---

## 4) Testing Status

Local test status (latest run):
- `pytest -q` -> **89 passed**

Added/updated tests cover:
- FSM transitions and loop prevention
- DynamoDB timestamp serialization
- Telegram markdown cleanup
- Category extraction false-positive prevention
- LLM/embedding fallback behavior

---

## 5) Last Known AWS Runtime Status

As last verified during deployment/testing:
- Stack `delhi-scheme-saathi` was successfully created/updated.
- Last known API endpoint:
  - `https://jk6mrqfa2d.execute-api.ap-south-1.amazonaws.com/prod`
- Health endpoint had reached:
  - `{"status":"ok","database":"connected","schemes_count":5}`
- Telegram webhook endpoint was configured and receiving updates.

Current immediate status at pause/resume point:
- RDS had been stopped to save cost.
- On resume, `aws rds start-db-instance` initially failed due expired SSO token.
- Next required step: re-login with SSO, then start/wait for DB availability.

---

## 6) Resume Checklist (Fast Path)

1. Re-authenticate AWS SSO:
   - `aws sso login --profile delhi-sso --use-device-code`
2. Start DB:
   - `aws rds start-db-instance --db-instance-identifier dss-postgres --region ap-south-1 --profile delhi-sso`
   - `aws rds wait db-instance-available --db-instance-identifier dss-postgres --region ap-south-1 --profile delhi-sso`
3. Run rapid redeploy + session reset:
   - `cd /home/ubuntu/ai-for-bharat/delhi-scheme-saathi`
   - `./scripts/rapid_redeploy.sh --user-id 780045592`
4. Re-set Telegram webhook (if needed) and re-test key flows.

---

## 7) Open Items

1. Re-run end-to-end Telegram conversation after latest redeploy to confirm:
   - no category drift (`SC/ST` false positives gone)
   - no markdown artifacts
   - profile collection remains in UNDERSTANDING until core fields are present.
2. Optional product enhancement:
   - Add explicit dependent profile model (`beneficiary_relation`, `beneficiary_age`) for parent/guardian scenarios.
