# Architecture

This document describes the technical architecture of Delhi Scheme Saathi.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Telegram                                │
│                     (User Interface)                            │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Webhook
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Server                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Webhook    │  │   REST API   │  │    Chat Endpoint     │  │
│  │   Handler    │  │  /api/...    │  │     /api/chat        │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │              │
│         └─────────────────┼──────────────────────┘              │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 Conversation Service                      │  │
│  │  ┌─────────┐  ┌───────────┐  ┌────────────────────────┐  │  │
│  │  │   FSM   │  │  Profile  │  │   Response Generator   │  │  │
│  │  │ Engine  │  │ Extractor │  │                        │  │  │
│  │  └────┬────┘  └─────┬─────┘  └───────────┬────────────┘  │  │
│  │       │             │                     │               │  │
│  │       └─────────────┼─────────────────────┘               │  │
│  └──────────────────────┼────────────────────────────────────┘  │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Service Layer                          │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │  │
│  │  │   Scheme    │  │  Document   │  │   Rejection     │   │  │
│  │  │   Matcher   │  │  Resolver   │  │   Engine        │   │  │
│  │  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘   │  │
│  └─────────┼────────────────┼──────────────────┼────────────┘  │
│            │                │                  │                │
│            └────────────────┼──────────────────┘                │
│                             ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                  Repository Layer                         │  │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────┐ │  │
│  │  │  Schemes   │ │  Documents │ │  Offices   │ │Sessions│ │  │
│  │  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └───┬────┘ │  │
│  └────────┼──────────────┼──────────────┼────────────┼──────┘  │
└───────────┼──────────────┼──────────────┼────────────┼──────────┘
            │              │              │            │
            ▼              ▼              ▼            ▼
┌───────────────────────────────────┐  ┌──────────────────────────┐
│         PostgreSQL + pgvector     │  │    In-Memory Store       │
│  ┌─────────┐ ┌─────────┐ ┌─────┐  │  │    (Sessions)            │
│  │ Schemes │ │Documents│ │Offi │  │  │                          │
│  │ +vector │ │         │ │ ces │  │  │                          │
│  └─────────┘ └─────────┘ └─────┘  │  └──────────────────────────┘
└───────────────────────────────────┘

            │                              │
            ▼                              ▼
┌───────────────────────────────────────────────────────────────────┐
│                        External APIs                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │   xAI (Grok)    │  │   Voyage AI     │  │   Telegram API  │   │
│  │   LLM Client    │  │   Embeddings    │  │   Bot Client    │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Webhook Handler (`src/webhook/handler.py`)

Receives Telegram updates and routes them to the conversation service:
- Parses incoming messages (text, voice, callbacks)
- Sends typing indicators
- Dispatches responses back to Telegram

### 2. Conversation Service (`src/services/conversation/`)

Main orchestrator that handles the conversation flow. One user message is
one turn, and `service.ConversationService.handle_message` runs the same
pipeline every time: load session → short-circuit commands and callbacks →
analyse → settle language → update profile → pick next state → render →
persist.

The package is split by what each layer needs to know, and dependencies run
strictly in this order:

| Module | Responsibility |
|--------|----------------|
| `language.py` | What language to read the message in and reply in |
| `intents.py` | What the user is asking for, from the message alone |
| `scheme_reference.py` | Which scheme the user means ("2", "second", by name) |
| `turn_policy.py` | Decisions that also need session and profile state |
| `views.py` | The plain text sent back to the user |
| `service.py` | The turn pipeline itself |

The LLM proposes and the deterministic layers dispose: anything the LLM
returns that conflicts with the plain meaning of the user's own words is
overridden, because a fluent wrong answer is worse here than a plain one.

### 3. FSM Engine (`src/services/fsm.py`)

10-state finite state machine managing conversation flow:

| State | Purpose |
|-------|---------|
| GREETING | Welcome message, initial interaction |
| SITUATION_UNDERSTANDING | Establish which life event brought the user here |
| PROFILE_COLLECTION | Collect age, gender, category, income |
| SCHEME_MATCHING | Transient state while searching schemes |
| SCHEME_PRESENTATION | Display matched schemes |
| SCHEME_DETAILS | Deep dive into selected scheme |
| DOCUMENT_GUIDANCE | Documents required, and where to get them |
| REJECTION_WARNINGS | Common reasons this application gets rejected |
| APPLICATION_HELP | Step-by-step application guidance |
| CSC_HANDOFF | Transfer to a service centre |

`ConversationState` also defines short aliases (`UNDERSTANDING`, `MATCHING`,
`PRESENTING`, `DETAILS`, `APPLICATION`, `HANDOFF`) that point at the states
above, kept for older code paths and tests.

State transitions:
```
GREETING ─────▶ SITUATION_UNDERSTANDING ─────▶ PROFILE_COLLECTION
                                                       │
           ┌───────────────────────────────────────────┘
           │ (profile complete)
           ▼
   SCHEME_MATCHING ────▶ SCHEME_PRESENTATION ────▶ SCHEME_DETAILS
           │                                              │
           │ (no schemes)                                 ▼
           ▼                            DOCUMENT_GUIDANCE / REJECTION_WARNINGS
   PROFILE_COLLECTION                        / APPLICATION_HELP
   (or SITUATION_UNDERSTANDING                        │
    when the topic is unknown)                        ▼
                                                 CSC_HANDOFF
```

The four scheme views reachable from `SCHEME_DETAILS` transition freely
between one another and back to `SCHEME_PRESENTATION`; see
`fsm.get_valid_transitions` for the exact table.

### 4. Profile Extractor (`src/services/profile_extractor.py`)

Extracts user information from natural language:
- Rule-based regex patterns for common formats
- LLM-based extraction for complex cases
- Supports Hindi, English, and Hinglish

Extracted fields:
- Age, gender, marital status
- Category (SC/ST/OBC/General/EWS)
- Annual income, employment status
- Life event, district, BPL status

### 5. Scheme Matcher (`src/services/scheme_matcher.py`)

3-stage hybrid search for scheme matching:

**Stage 1: Life Event Filter**
```sql
WHERE $life_event = ANY(life_events)
```

**Stage 2: Eligibility Filter**
```sql
AND (eligibility->>'min_age')::int <= $age
AND (eligibility->>'max_income')::int >= $income
```

**Stage 3: Vector Similarity**
```sql
ORDER BY description_embedding <=> $query_embedding
```

### 6. Document Resolver (`src/services/document_resolver.py`)

Resolves document prerequisite chains using DFS:
```
Income Certificate
  └── Aadhaar Card
  └── Ration Card
      └── Aadhaar Card (already resolved)
```

Returns flat list in procurement order.

### 7. LLM Client (`src/integrations/llm_client.py`)

OpenAI-compatible client for xAI's Grok:
- Single-call message analysis (intent, life event, profile extraction)
- Response generation with context
- Conversation summarization

Uses structured JSON output for reliable parsing.

### 8. Embedding Client (`src/integrations/embedding_client.py`)

Voyage AI client for multilingual embeddings:
- Model: `voyage-multilingual-2`
- Dimension: 1024
- Used for semantic scheme search

## Database Schema

### Core Tables

**schemes**
- Primary scheme data with eligibility as JSONB
- `description_embedding` as vector(1024) for similarity search
- HNSW index for fast vector queries

**documents**
- Document metadata with prerequisites
- Procurement guidance (fees, authorities, processing time)

**offices**
- Government office locations
- Services offered, working hours

**rejection_rules**
- Common rejection reasons by scheme
- Severity levels (critical, high, warning)
- Prevention tips

### Indexes

```sql
-- Vector similarity search (HNSW)
CREATE INDEX idx_schemes_embedding ON schemes
    USING hnsw (description_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Life event filtering
CREATE INDEX idx_schemes_life_events ON schemes USING GIN (life_events);

-- Eligibility JSONB queries
CREATE INDEX idx_schemes_eligibility ON schemes USING GIN (eligibility);
```

## Data Models

### Pydantic Models (Immutable)

All domain models use `frozen=True` for immutability:

```python
class Scheme(BaseModel, frozen=True):
    id: str
    name: str
    eligibility: EligibilityCriteria
    ...

class UserProfile(BaseModel):
    age: int | None = None
    category: str | None = None
    ...

    def merge_with(self, other: "UserProfile") -> "UserProfile":
        # Immutable merge - returns new instance
```

### Session Management

Sessions track conversation state:
- Current FSM state
- User profile (accumulated)
- Message history (sliding window of 10)
- Discussed schemes

## Request Flow

1. **Telegram webhook** receives update
2. **Handler** extracts message and user info
3. **Conversation service** loads session
4. **LLM** analyzes message (intent, entities, life event)
5. **Profile extractor** updates user profile
6. **FSM** determines next state
7. **Scheme matcher** finds relevant schemes (if needed)
8. **Response generator** creates bilingual response
9. **Session** saved with updated state
10. **Telegram client** sends response

## Configuration

Environment-based configuration via Pydantic Settings:

```python
class Settings(BaseSettings, frozen=True):
    database_url: str
    xai_api_key: str
    voyage_api_key: str
    telegram_bot_token: str
    log_level: str = "INFO"
```

## Error Handling

- Database errors return 503 Service Unavailable
- LLM failures fall back to generic responses
- Embedding failures disable vector search (SQL-only matching)
- All errors logged with context

## Performance Considerations

- Connection pooling (asyncpg, min=2, max=10)
- HNSW index for O(log n) vector search
- Session sliding window limits memory
- Async throughout for concurrency
