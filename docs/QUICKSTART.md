# Quick Start Guide

This guide walks you through setting up Delhi Scheme Saathi locally.

## Prerequisites

- **Docker** (20.10+) and **Docker Compose** (v2)
- **Python 3.10+** (for running scripts outside container)
- API keys:
  - [xAI API Key](https://x.ai/) for Grok LLM
  - [Voyage AI Key](https://www.voyageai.com/) for embeddings
  - [Telegram Bot Token](https://core.telegram.org/bots#creating-a-new-bot) from BotFather

## Step 1: Clone and Configure

```bash
git clone git@github.com:your-org/delhi-scheme-saathi.git
cd delhi-scheme-saathi

# Copy environment template
cp .env.example .env
```

Edit `.env` with your API keys:

```env
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/delhi_scheme_saathi
XAI_API_KEY=xai-your-key-here
XAI_BASE_URL=https://api.x.ai/v1
VOYAGE_API_KEY=pa-your-key-here
TELEGRAM_BOT_TOKEN=123456:ABC-your-token-here
LOG_LEVEL=INFO
```

## Step 2: Start Services

```bash
# Start PostgreSQL and the application
docker compose up -d

# Check container status
docker compose ps

# View logs
docker compose logs -f app
```

The services will be available at:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **PostgreSQL**: localhost:5434

## Step 3: Verify Setup

```bash
# Health check
curl http://localhost:8000/health

# Expected response:
# {"status":"ok","database":"connected","schemes_count":5}
```

If `schemes_count` is 0, the database may need seeding (see Step 4).

## Step 4: Seed Database (if needed)

The database is automatically seeded on first startup via Docker init scripts. If you need to reseed:

```bash
# Stop containers and remove volume
docker compose down -v

# Restart (will reseed on startup)
docker compose up -d
```

## Step 5: Generate Embeddings

Generate semantic embeddings for scheme search:

```bash
# Using Python directly (with .env loaded)
pip install python-dotenv asyncpg voyageai

python3 << 'EOF'
import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

import asyncpg
import voyageai

async def generate():
    pool = await asyncpg.create_pool(
        'postgresql://postgres:postgres@localhost:5434/delhi_scheme_saathi'
    )
    client = voyageai.Client(api_key=os.environ['VOYAGE_API_KEY'])

    rows = await pool.fetch("SELECT id, description, description_hindi FROM schemes")
    for row in rows:
        text = f"{row['description']}\n\n{row['description_hindi']}"
        result = client.embed([text], model="voyage-multilingual-2")
        embedding = result.embeddings[0]
        embedding_str = f"[{','.join(map(str, embedding))}]"
        await pool.execute(
            "UPDATE schemes SET description_embedding = $1::vector WHERE id = $2",
            embedding_str, row['id']
        )
        print(f"Generated embedding for {row['id']}")
    await pool.close()

asyncio.run(generate())
EOF
```

## Step 6: Test the Chat API

```bash
# Send a greeting
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "message": "Namaste"}'

# Ask about widow pension
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "message": "Mera pati guzar gaye, pension chahiye"}'
```

## Step 7: Set Up Telegram Webhook (Optional)

For production Telegram integration:

```bash
# Using ngrok for local testing
ngrok http 8000

# Set webhook (replace with your ngrok URL)
curl "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook?url=https://xxx.ngrok.io/webhook/telegram"

# Verify webhook
curl "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getWebhookInfo"
```

## Running Tests

```bash
# Create virtual environment and install dev dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

# Run tests
export DATABASE_URL="postgresql://postgres:postgres@localhost:5434/delhi_scheme_saathi"
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## Step 8: Voice Integration with Sarvam AI (Optional)

Sarvam AI provides high-quality Indian language voice services with 1000 free credits on signup.

### Getting Sarvam AI API Keys

1. Visit [Sarvam AI Console](https://console.sarvam.ai/)
2. Sign up for a developer account (1000 free credits included)
3. Create an API subscription key from the dashboard
4. Add to your `.env` file:

```env
SARVAM_API_KEY=your-api-subscription-key
```

### Alternative: Bhashini (Fallback)

If Sarvam AI is unavailable, you can use Bhashini (requires paid subscription):
1. Visit [Bhashini ULCA Portal](https://bhashini.gov.in/ulca)
2. Register for a developer account
3. Add to `.env`:

```env
BHASHINI_API_KEY=your-api-key
BHASHINI_USER_ID=your-user-id
```

### Testing Voice Integration

```bash
# With virtual environment activated
python scripts/test_voice_integration.py
```

The script will:
- Test TTS (text-to-speech) in Hindi using Sarvam AI
- Test STT (speech-to-text) transcription
- Test language detection

### Voice Features

When configured, the bot will:
- Accept Hindi voice messages via Telegram
- Transcribe voice to text using Sarvam AI ASR (Saaras v3 model with transcribe mode)
- Respond with both text and audio (TTS via bulbul:v3 model)

**Supported Languages:** Hindi, English, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Odia

**Note:** Voice features gracefully degrade to text-only when no voice API is configured.

## Stopping Services

```bash
# Stop containers (preserves data)
docker compose stop

# Stop and remove containers
docker compose down

# Stop and remove containers + data
docker compose down -v
```

## Troubleshooting

### Database Connection Failed
```bash
# Check if postgres is running
docker compose ps postgres

# Check postgres logs
docker compose logs postgres
```

### Port Already in Use
```bash
# Check what's using port 5434
lsof -i :5434

# Edit docker-compose.yml to use different port
```

### LLM API Errors
- Verify `XAI_API_KEY` is correct
- Check rate limits on your xAI account
- View app logs: `docker compose logs app`

### Embeddings Not Working
- Verify `VOYAGE_API_KEY` is correct
- Check if embeddings exist: `docker exec dss-postgres psql -U postgres -d delhi_scheme_saathi -c "SELECT id, description_embedding IS NOT NULL as has_emb FROM schemes;"`
