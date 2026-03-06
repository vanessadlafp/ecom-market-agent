# Market Analysis Agent

AI-powered e-commerce market intelligence built with **LangGraph + Groq + FastAPI + DuckDuckGo**.

## Setup and Basic Architecture

**LangGraph** was chosen over other options (e.g. CrewAI, Google ADK, or a native Python orchestration) for its explicit, graph-based control flow, typed state (a single `AgentState` TypedDict with no JSON handoffs), and minimal abstraction—nodes are plain Python functions, so tools are easy to test and mock. It fits FastAPI’s async stack and leaves room to add branching or subgraphs later without rewriting the pipeline.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        FastAPI                           │
│            POST /api/v1/analyze   GET /   GET /health    │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────┐
│              LangGraph Orchestrator                       │
│         (react_agent.py — setup/index/create/chat)        │
│                                                          │
│  AgentState: product_query, use_sample_data, language    │
│  Typed results: raw_data → SentimentResult, TrendResult  │
└──────────┬───────────────┬───────────────┬───────────────┘
           │               │               │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌─────▼──────┐
    │   Scraper   │ │  Sentiment  │ │   Trends   │
    │  (DDG/real) │ │ Groq fast  │ │ Groq fast  │
    │  +sample    │ │ (8b, 800tk) │ │ (8b, 800tk)│
    └─────────────┘ └─────────────┘ └────────────┘
                         │
          ┌──────────────▼───────────────┐
          │      Report Generator        │  ← POST-GRAPH (not a tool node)
          │  Groq main model (70b, 2k tk)│
          └──────────────────────────────┘
                         │
                    MarketReport
```

### Key design decisions

| Decision | Rationale |
|---|---|
| Scraper has live + sample mode | DDG runs real in prod; `USE_SAMPLE_DATA=true` in .env or `use_sample_data: true` in request for testing |
| Trends are pure mock | Keeps demo deterministic; swap in Google Trends API without changing interface |
| Report runs post-graph | Avoids passing JSON blobs between tool calls; LLM can't skip or mis-sequence it |
| AgentState holds typed objects | Structured access in tests; no JSON parsing in the orchestrator |
| Request-level language | `language: "en"` or `"fr"` for report output; optional in API and UI |
| Two-tier LLM usage | Sentiment and trends use **fast model** (e.g. `llama-3.1-8b-instant`, 800 tokens) for short JSON; **report** uses main model (`GROQ_MODEL`, 2048 tokens) to save quota and reduce rate limits |

## Repo structure

```
ecom-market-agent/
├── config/
│   ├── agent_config.py     # Max iterations, tool descriptions
│   ├── llm_config.py       # fast_model (sentiment/trends), model (report), max_tokens, report_max_tokens
│   ├── search_config.py    # DDG + scraper settings
│   └── settings.py         # Runtime settings from .env
├── data/
│   ├── cache/              # Scrape cache (generated)
│   ├── documents/          # Future RAG source docs
│   └── samples/
│       ├── scraper_samples.json   # Sample pricing/listings (used when use_sample_data=true)
│       ├── default.json           # Optional sample fixture
│       └── iphone_16_pro.json     # Optional sample fixture
├── scripts/
│   └── health_check.py     # Checks Groq, DDG, FastAPI
├── src/
│   ├── agent/
│   │   └── react_agent.py  # setup / index / create_agent / chat
│   ├── api/
│   │   ├── routes.py       # FastAPI endpoints + browser UI (Chart.js sentiment)
│   │   └── cache.py        # In-memory TTL cache
│   ├── tools/
│   │   ├── scraper.py      # DDG + httpx/BS4, sample fallback
│   │   ├── sentiment.py    # Groq fast model (short JSON)
│   │   ├── trends.py       # Groq fast model (short JSON)
│   │   └── report.py       # Groq main model (full report, post-graph)
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
├── pytest.ini              # asyncio_mode=auto, live marker
└── requirements.txt
```

## Demo

Demo of the report flow: running an analysis, toggling **Sample data**, and switching report **language** (e.g. Français).

![Market Analysis Agent demo](ecom-market-agent-demo.mp4)

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env        # add GROQ_API_KEY
make dev                    # → http://localhost:8000
```

The browser UI at `http://localhost:8000` supports a **Sample data** checkbox (uses `data/samples/` instead of live DDG) and a **Français** toggle for report language.

## Testing

```bash
make test                   # all tests (DDG + LLM mocked)
make test-unit              # tool-level only
make test-integration       # graph + API routes
make health                 # live health check (Groq, DDG, FastAPI)
```

**Alternative — run all tests from inside the container:**

```bash
# inside the container
pytest tests/ -v
```

(Use `make docker-shell` to get a shell in the agent container, then run the command above.)

Tests that hit real APIs are marked with `@pytest.mark.live`; run them with `pytest -m live` (requires network).

## Docker

```bash
make docker-build && make docker-up
make docker-logs            # follow agent logs
make docker-down            # stop containers
make docker-shell           # shell into agent container
```

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Browser UI (Chart.js sentiment, sample/language toggles) |
| `GET` | `/health` | Liveness probe (returns status + Groq model) |
| `POST` | `/api/v1/analyze` | Run market analysis |
| `GET` | `/docs` | Swagger UI |

**Request body** for `POST /api/v1/analyze`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `product_query` | string | required | Product to analyze (e.g. "Sony WH-1000XM6") |
| `use_sample_data` | boolean | `false` | Use `data/samples/` instead of live DuckDuckGo |
| `language` | string | `"en"` | Report language: `"en"` or `"fr"` |

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"product_query": "Sony WH-1000XM6"}'
```

With sample data and French report:

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"product_query": "iPhone 16 Pro", "use_sample_data": true, "language": "fr"}'
```

**Environment:** set `USE_SAMPLE_DATA=true` in `.env` to make the scraper use sample data by default (overridable per request).

### Optional env vars (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | (required) | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Main LLM for **report** only; sentiment and trends use the fast model in `config/llm_config.py` (`llama-3.1-8b-instant`) |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | Server bind |
| `API_RELOAD` | `false` | Enable uvicorn reload |
| `AGENT_MAX_ITERATIONS` | `10` | LangGraph recursion limit |
| `SCRAPER_MAX_RESULTS` / `SCRAPER_TIMEOUT` | `8` / `15` | DDG scraper limits |
| `CACHE_TTL_SECONDS` | `3600` | In-memory cache TTL for reports |
| `USE_SAMPLE_DATA` | `false` | Use sample data instead of live DDG |

**LLM token limits and fast model** are set in `config/llm_config.py`: `max_tokens` (sentiment/trends, default 800), `report_max_tokens` (report, default 2048), and `fast_model` (e.g. `llama-3.1-8b-instant`). Adjust there to reduce rate-limit risk or allow longer reports.

---

## 4. Data Architecture and Storage

| Concern | Approach | Storage recommendation |
|--------|----------|------------------------|
| **Analysis results** | Persist full `MarketReport` (executive summary, pricing/sentiment/trends text, recommendations) plus `request_id`, `product_query`, `status`, `duration_seconds`, `generated_at`. Optionally store `raw_data` / `raw_sentiment` / `raw_trends` for replay or debugging. | **PostgreSQL** (or similar RDBMS): one row per analysis with JSONB for report payload; indexes on `product_query`, `generated_at`, `status`. Enables history, search, and analytics. |
| **Query history** | Log every request: `request_id`, `product_query`, `use_sample_data`, `language`, `status`, `duration_seconds`, `created_at`, optional `user_id`/`tenant_id`. | Same **PostgreSQL**: `analysis_requests` (or append to a single table). Enables “recent analyses”, rate limiting per user, and quality dashboards. |
| **Cache (collected data)** | Cache key = `report:{product_query}:sample={bool}:lang={lang}`. Value = full `AnalysisResponse`. Today: in-memory TTL (`src/api/cache.py`); for multi-instance and durability: external cache. | **Redis**: same key/value shape, TTL per key. Survives restarts and is shared across replicas. Optional: cache raw scraper output separately (longer TTL) to reuse for sentiment/trends without re-scraping. |
| **Agent configuration** | Store model name, max iterations, timeouts, feature flags (e.g. “use sample data by default”). | **PostgreSQL** or **config service**: one row/setting per key, or use existing `.env` + a small `agent_config` table for overrides. Avoid storing secrets in DB; keep `GROQ_API_KEY` in env/secrets manager. |

**Recommended mix:** **PostgreSQL** for results and history (durable, queryable, backups); **Redis** for report and optional scraper-output cache (low latency, TTL); **env/secrets manager** for API keys and non-overridable config. Queues (e.g. **Celery + Redis** or **RabbitMQ**) become relevant when decoupling HTTP from execution (see §6).

---

## 5. Monitoring and Observability

| Goal | Approach |
|------|----------|
| **Trace agent execution** | Use **structured logging** (e.g. `structlog`, already in use) with a shared `request_id` (and optional `trace_id`) on every log line. In production, ship logs to **OpenTelemetry** or a vendor (Datadog, Grafana Loki, etc.) and correlate with traces. Optionally add **OpenTelemetry tracing** around the graph: one span per node (scrape, sentiment, trends) and one for `generate_report`, so you see latency per step. |
| **Performance metrics** | Instrument: **request count** and **latency** (e.g. p50/p95/p99) for `POST /api/v1/analyze`; **per-node duration** (scrape, sentiment, trends, report); **cache hit rate**; **Groq/DDG error rate** and **retries**. Expose a **Prometheus** `/metrics` endpoint (e.g. via `prometheus-fastapi-instrumentator`) and scrape with Prometheus or Grafana Agent. |
| **Alerting on malfunctions** | Define alerts for: **error rate** above threshold; **latency** above SLO (e.g. p95 &gt; 60s); **dependency failures** (Groq 429, DDG timeouts); **health check** `/health` failing. Use **Alertmanager** (with Prometheus) or your observability stack’s alerting. On-call runbooks should reference logs and traces keyed by `request_id`. |
| **Output quality** | Track **structured outputs** (e.g. report has non-empty sections, recommendation count). Optionally score quality with an **LLM-as-judge** (see §7) and store scores in DB; dashboard by `product_query` or time. Monitor **anomalies** (e.g. very short reports, repeated failures for the same query). |

**Key metrics to monitor:** request rate, latency (overall and per node), cache hit ratio, Groq token usage and rate-limit errors, DDG success/timeout rate, analysis success vs failed count, and—if implemented—quality score distribution.

---

## 6. Scaling and Optimization

| Challenge | Approach |
|-----------|----------|
| **Peak loads (100+ simultaneous analyses)** | **Horizontal scaling:** run multiple FastAPI replicas behind a load balancer; share nothing except Redis (cache) and PostgreSQL (results/history). **Queue-based decoupling:** put each analysis request on a queue (e.g. Celery, Redis Queue, or SQS); workers consume and call the existing `chat()` pipeline. Return `request_id` immediately with **202 Accepted**; clients poll a “status” endpoint or use webhooks. This caps in-flight work per worker and avoids overloading Groq/DDG. **Rate limiting:** per-user or per-API-key limits to avoid a single client exhausting capacity. |
| **Optimize LLM usage costs** | **Cache aggressively:** same `(product_query, language, use_sample_data)` → return cached report (Redis). **Smaller/faster model** for sentiment if acceptable (e.g. smaller Groq model or a dedicated sentiment model). **Prompt design:** shorter system prompts and fewer few-shot examples where possible; reuse one report prompt for both en/fr. **Token budgets:** cap max tokens for Groq calls; fallback to “insufficient data” instead of retrying with huge context. **Monitor token usage** per request and aggregate by tenant to spot outliers. |
| **Smart caching** | **Layered cache:** (1) **Full report cache** (current): key = query + options, TTL e.g. 1h. (2) **Scraper-output cache:** key = `scraper:{product_query}`, TTL longer (e.g. 6–24h); reuse RawData for sentiment/trends/report without re-scraping. Invalidate on demand or by TTL. **Cache warming:** optional background job for popular queries. **Cache-aside:** app checks cache before calling the graph; on miss, run graph then write to cache. |
| **Parallelize analysis tasks** | **Across requests:** multiple workers or replicas handle different requests in parallel (no change to single-request flow). **Within a request:** the current graph is sequential (scrape → sentiment → trends); sentiment and trends could run in parallel after scrape (e.g. LangGraph `Send` or two branches joining before report). Report stays post-graph. **Parallelize scrape:** if we ever support multiple sources, scrape them concurrently and merge into `RawData`. |

---

## 7. Continuous Improvement and A/B Testing

| Goal | Approach |
|------|----------|
| **Automatically evaluate analysis quality (LLM as Judge)** | For each completed report (or a sample), call a **judge LLM** with a fixed rubric: clarity, completeness, relevance of recommendations, factual consistency with inputs. Prompt: “Given this MarketReport and the raw inputs (listings, sentiment, trends), score 1–5 on criteria X, Y, Z.” Store scores in PostgreSQL (e.g. `quality_evaluations` table: `request_id`, `scores`, `model`, `evaluated_at`). Aggregate by time, model version, or prompt version to track quality over time. |
| **Compare prompt engineering strategies** | **A/B test prompts:** store multiple “prompt versions” (e.g. in DB or config); assign each request a `prompt_version` (random or by hash of `product_query`). Log `prompt_version` with the analysis. Compare outcomes (quality scores, user feedback, latency) by version. Use statistical significance (e.g. t-test or Bayesian) before rolling out a new default. **Feature flags** can toggle between prompt sets without code deploy. |
| **User feedback loop** | **Explicit feedback:** add a “thumbs up/down” or rating (1–5) to the API or UI; store in DB with `request_id` and optional comment. **Implicit signals:** track “same query run again soon” (user re-ran = possibly unsatisfied) or “no follow-up request” (neutral). Correlate feedback with `prompt_version`, model, and quality judge scores to tune prompts and models. |
| **Scale agent capabilities** | **New tools:** add nodes to the graph (e.g. “competitor comparison”, “inventory check”) with the same pattern: typed state, plain Python node, post-graph report step consumes new result. **RAG:** use `data/documents/` and the existing `index()` hook to inject retrieved docs into the report prompt. **Specialisation:** separate agents or subgraphs for “quick summary” vs “full report” (different recursion limits or tools); route by request parameter. **Model upgrades:** run A/B tests when switching Groq model; compare quality and cost. |
