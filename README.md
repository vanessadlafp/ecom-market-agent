# Market Analysis Agent

AI-powered e-commerce market intelligence built with **LangGraph + Groq + FastAPI + DuckDuckGo**.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        FastAPI                           │
│            POST /api/v1/analyze   GET /                  │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│              LangGraph Orchestrator                       │
│         (react_agent.py — setup/index/create/chat)       │
│                                                          │
│  AgentState carries TYPED results (not JSON strings):    │
│    scraper_result   SentimentResult   TrendResult        │
└──────────┬───────────────┬───────────────┬───────────────┘
           │               │               │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐
    │   Scraper   │ │  Sentiment  │ │   Trends   │
    │  (DDG/real) │ │ (Groq/real) │ │   (mock)   │
    │  +mock mode │ │             │ │            │
    └─────────────┘ └─────────────┘ └────────────┘
                         │
          ┌──────────────▼───────────────┐
          │      Report Generator        │  ← POST-GRAPH (not a tool node)
          │         (Groq/real)          │
          └──────────────────────────────┘
                         │
                    MarketReport
```

### Key design decisions

| Decision | Rationale |
|---|---|
| Scraper has live + mock mode | DDG runs real in prod; `USE_MOCK_SCRAPER=true` or `force_mock=True` for testing |
| Trends are pure mock | Keeps demo deterministic; swap in Google Trends API without changing interface |
| Report runs post-graph | Avoids passing JSON blobs between tool calls; LLM can't skip or mis-sequence it |
| AgentState holds typed objects | Structured access in tests; no JSON parsing in the orchestrator |

## Repo structure

```
market-agent/
├── config/
│   ├── agent_config.py     # Max iterations, tool descriptions
│   ├── llm_config.py       # Groq model, temperature
│   ├── search_config.py    # DDG + scraper settings
│   └── settings.py         # Runtime settings from .env
├── data/
│   ├── cache/              # Scrape cache (generated)
│   ├── documents/          # Future RAG source docs
│   └── samples/
│       ├── scraper_samples.json   # Mock pricing data
│       └── trends_samples.json    # Mock trend data
├── scripts/
│   └── health_check.py     # Checks Groq, DDG, FastAPI
├── src/
│   ├── agent/
│   │   └── react_agent.py  # setup / index / create_agent / chat
│   ├── api/
│   │   ├── routes.py       # FastAPI endpoints + browser UI
│   │   └── cache.py        # In-memory TTL cache
│   ├── tools/
│   │   ├── scraper.py      # DDG + httpx/BS4, mock fallback
│   │   ├── sentiment.py    # Groq LLM (real)
│   │   ├── trends.py       # Mock (deterministic)
│   │   └── report.py       # Groq LLM (real, post-graph)
│   └── utils/
│       ├── logger.py       # structlog
│       └── schemas.py      # Pydantic models
├── tests/
│   ├── unit/               # Per-tool tests (DDG + LLM mocked)
│   └── integration/        # Graph state + API route tests
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── main.py
├── Makefile
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env        # add GROQ_API_KEY
make dev                    # → http://localhost:8000
```

## Testing

```bash
make test                   # all tests (DDG + LLM mocked)
make test-unit              # tool-level only
make test-integration       # graph + API routes
make test-live              # real DDG calls (needs network)
make health                 # live health check
```

## Docker

```bash
make docker-build && make docker-up
make docker-logs
```

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Browser UI |
| `GET` | `/health` | Liveness probe |
| `POST` | `/api/v1/analyze` | Run market analysis |
| `GET` | `/docs` | Swagger UI |

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"product_query": "Sony WH-1000XM6"}'
```

To use mock scraper data (no DDG):
```bash
# In .env:
USE_MOCK_SCRAPER=true
```
