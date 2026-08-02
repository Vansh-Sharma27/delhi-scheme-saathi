# Delhi Scheme Saathi

A voice-first Hindi chatbot that helps Delhi residents discover and apply for government welfare schemes. The bot provides personalized scheme recommendations based on user profiles, document procurement guidance, and rejection prevention tips.

## Features

- **Conversational Interface**: Natural Hindi/English/Hinglish conversations via Telegram
- **Life Event Detection**: Automatically identifies user situations (housing, health crisis, widowhood, etc.)
- **Smart Scheme Matching**: 3-stage hybrid search combining SQL filters and semantic vector similarity
- **Profile Extraction**: Extracts age, income, category, and other details from natural conversation
- **Document Guidance**: Shows which documents are needed and how to obtain them
- **Rejection Prevention**: Proactive warnings about common application mistakes

## What This Bot Can Do

For a short bilingual note for judges and demo reviewers, see [docs/WHAT_THIS_BOT_CAN_DO.md](docs/WHAT_THIS_BOT_CAN_DO.md).
In Telegram, try `/help` first. The bot now also exposes `/start`, `/help`, and `/language` in the Telegram command menu when commands are synced.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3.11, FastAPI |
| Database | PostgreSQL 16 + pgvector |
| LLM | Grok (xAI) via OpenAI-compatible API; AWS Bedrock (Nova) when `USE_BEDROCK=true` |
| Embeddings | Jina AI (`jina-embeddings-v3`) primary, Voyage AI (`voyage-multilingual-2`, 1024-dim) fallback |
| Voice | Sarvam AI primary, Bhashini fallback |
| Messaging | Telegram Bot API |
| Containerization | Docker, Docker Compose |

## Project Structure

```
delhi-scheme-saathi/
├── src/
│   ├── models/          # Pydantic data models
│   ├── db/              # Database repositories
│   ├── services/        # Business logic (FSM, matching, extraction)
│   ├── integrations/    # External APIs (LLM, embeddings, Telegram)
│   ├── prompts/         # LLM prompt templates
│   ├── utils/           # Validators, keyboards, logging, scheme catalog
│   └── webhook/         # Telegram webhook handler
├── data/                # Seed data (schemes, documents, offices)
├── scripts/             # Database seeding and utilities
├── tests/               # Unit and integration tests
└── docs/                # Documentation
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- API keys for: xAI (Grok), Voyage AI, Telegram Bot

### Setup

1. Clone and configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys
```

2. Start services:
```bash
docker compose up -d
```

On a fresh local volume, the app container now auto-seeds the bundled scheme data during startup.

3. Verify health:
```bash
curl http://localhost:8000/health
# {"status":"ok","database":"connected","schemes_count":5}
```

4. Generate embeddings (first time only):
```bash
docker exec -it dss-app python3 scripts/generate_embeddings.py
```

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for detailed setup instructions.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check with database status |
| `GET /api/schemes` | List schemes, filter by life event |
| `GET /api/scheme/{id}` | Full scheme details with documents |
| `GET /api/document/{id}` | Document procurement guide |
| `GET /api/csc/nearest` | Nearest government offices |
| `GET /api/life-events` | List of life event categories |
| `POST /api/chat` | Direct chat endpoint (for testing); sessions are namespaced separately from Telegram |
| `POST /webhook/telegram` | Telegram webhook handler |

See [docs/API.md](docs/API.md) for complete API documentation.

## Architecture

The system uses a 10-state finite state machine (FSM) to manage conversation flow:

```
GREETING → SITUATION_UNDERSTANDING → PROFILE_COLLECTION → SCHEME_MATCHING
         → SCHEME_PRESENTATION → SCHEME_DETAILS → CSC_HANDOFF
```

From `SCHEME_DETAILS` the user can move freely between the four per-scheme
views — `SCHEME_DETAILS`, `DOCUMENT_GUIDANCE`, `REJECTION_WARNINGS` and
`APPLICATION_HELP` — or back to the scheme list.

Scheme matching uses a 3-stage hybrid approach:
1. **SQL Filter**: Life event and eligibility criteria
2. **Vector Search**: Semantic similarity using pgvector
3. **Ranking**: Combined score with eligibility match details

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture, and
[docs/adr/](docs/adr/README.md) for the decisions behind it — what was chosen,
what was rejected, and why.

## Data

The system includes seed data for:
- 5 welfare schemes (PMAY-U Housing, Widow Pension, Health, Education Loan, Self-Employment)
- 29 documents with prerequisites and procurement guides
- 16 government offices with locations and services
- 46 rejection rules with prevention tips
- 10 life event categories

## Testing

Run unit tests:
```bash
pytest tests/ -v
```

Rapid AWS redeploy + Telegram session reset:
```bash
./scripts/rapid_redeploy.sh --user-id 780045592
```
This will rebuild, redeploy, delete that user session from DynamoDB, and run `/health`.

Test the chat API directly:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "message": "Namaste"}'
```

`user_id` here is caller-supplied and unauthenticated, so it is stored under an
`api:` prefix. A conversation started through this endpoint is therefore a
separate session from the Telegram chat with the same ID — passing a real
Telegram user ID does not open that user's session. Add `-H "X-API-Key: ..."`
when `CHAT_API_KEY` is set.

To reproduce something a real user hit, copy their session into that keyspace
and replay against the copy:
```bash
python scripts/fork_session.py show 780045592          # inspect
python scripts/fork_session.py fork 780045592 --to repro-turn-12
```
Requires a shared session store (DynamoDB); the local in-memory store lives
inside the app process and is not reachable from a separate command.

Scan dependencies for known vulnerabilities:
```bash
python -m pip_audit -r requirements.txt
```

## Environment Variables

See `.env.example` for the full list. The ones that matter most:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `XAI_API_KEY` | xAI API key for Grok LLM |
| `JINA_API_KEY` | Jina AI key for embeddings (primary) |
| `VOYAGE_API_KEY` | Voyage AI key for embeddings (fallback) |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token |
| `SARVAM_API_KEY` | Sarvam AI key for voice; without it voice is disabled |
| `BHASHINI_API_KEY` | Bhashini voice fallback (with `BHASHINI_USER_ID`, `BHASHINI_ULCA_API_KEY`) |
| `USE_BEDROCK` | `true` routes the LLM through AWS Bedrock with Grok as fallback |
| `AI_MEMORY_QUEUE_BACKEND` | `in_memory` locally, `sqs` for shared AWS queue |
| `AI_MEMORY_QUEUE_URL` | SQS queue URL for async working-memory jobs |
| `LOG_LEVEL` | Logging level (INFO, DEBUG) |

Access control — all default to open, which is intended only for local use:

| Variable | Description |
|----------|-------------|
| `TELEGRAM_WEBHOOK_SECRET` | Checked against `X-Telegram-Bot-Api-Secret-Token`. **Empty disables the check**, so `/webhook/telegram` accepts any caller |
| `CHAT_API_KEY` | Required in `X-API-Key` on `/api/chat` when set. Empty leaves the endpoint open |
| `CORS_ALLOWED_ORIGINS` | Comma-separated allowed origins |

Neither `docker-compose.yml` nor `sam-template.yaml` currently passes
`TELEGRAM_WEBHOOK_SECRET` or `CHAT_API_KEY` to the container, so setting them
in `.env` alone has no effect on those deployments. Wire them into whichever
deployment path you use before exposing the service publicly.

Configured credentials are stripped from log output by
`src/utils/logging_config.py` before any handler emits a record. This matters
because the Telegram bot token is part of every Telegram request URL, and httpx
includes that URL in the exceptions the webhook handler logs.

## License

MIT License
