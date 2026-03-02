# Delhi Scheme Saathi

A voice-first Hindi chatbot that helps Delhi residents discover and apply for government welfare schemes. The bot provides personalized scheme recommendations based on user profiles, document procurement guidance, and rejection prevention tips.

## Features

- **Conversational Interface**: Natural Hindi/English/Hinglish conversations via Telegram
- **Life Event Detection**: Automatically identifies user situations (housing, health crisis, widowhood, etc.)
- **Smart Scheme Matching**: 3-stage hybrid search combining SQL filters and semantic vector similarity
- **Profile Extraction**: Extracts age, income, category, and other details from natural conversation
- **Document Guidance**: Shows which documents are needed and how to obtain them
- **Rejection Prevention**: Proactive warnings about common application mistakes

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | Python 3.11, FastAPI |
| Database | PostgreSQL 16 + pgvector |
| LLM | Grok (xAI) via OpenAI-compatible API |
| Embeddings | Voyage AI (voyage-multilingual-2, 1024-dim) |
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
│   ├── utils/           # Validators, formatters, helpers
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
| `POST /api/chat` | Direct chat endpoint (for testing) |
| `POST /webhook/telegram` | Telegram webhook handler |

See [docs/API.md](docs/API.md) for complete API documentation.

## Architecture

The system uses a 7-state finite state machine (FSM) to manage conversation flow:

```
GREETING → UNDERSTANDING → MATCHING → PRESENTING → DETAILS → APPLICATION → HANDOFF
```

Scheme matching uses a 3-stage hybrid approach:
1. **SQL Filter**: Life event and eligibility criteria
2. **Vector Search**: Semantic similarity using pgvector
3. **Ranking**: Combined score with eligibility match details

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture.

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

Test the chat API directly:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "message": "Namaste"}'
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `XAI_API_KEY` | xAI API key for Grok LLM |
| `VOYAGE_API_KEY` | Voyage AI key for embeddings |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token |
| `LOG_LEVEL` | Logging level (INFO, DEBUG) |

## License

MIT License
