"""Market Analysis Agent — LangGraph orchestrator.

Architecture
------------
The graph is responsible for DATA COLLECTION only.
Report generation happens outside the graph in the service layer.

Graph topology:

    START
      │
      ▼
  [scrape] ──► [sentiment] ──► [trends] ──► END

State carries typed result objects — not JSON strings or message blobs.
Each node receives the full state and writes exactly one field.

Why sequential (not parallel)?
  sentiment takes ScraperResult as input — it needs scraper to run first.
  trends is independent but kept sequential for simplicity; trivial to
  parallelise later with Send() if latency becomes a concern.

Public interface (mirrors reference repo react_agent.py):
  setup()        — validate config
  index()        — future RAG hook
  create_agent() — compile the graph
  chat()         — run graph + generate report, return MarketReport
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from config import agent_config
from config.settings import get_settings
from src.tools.report import generate_report
from src.tools.scraper import scrape_product
from src.tools.sentiment import analyze_sentiment
from src.tools.trends import analyze_trends
from src.utils.logger import get_logger
from src.utils.schemas import (
    AnalysisRequest,
    MarketReport,
    RawData,
    SentimentResult,
    TrendResult,
)

logger = get_logger(__name__)


# ── Typed state ───────────────────────────────────────────────────────────────
# scrape_node runs first and populates raw_data.
# sentiment_node and trends_node read from raw_data — they never call DDG.

class AgentState(TypedDict):
    product_query:    str
    use_sample_data:  bool
    raw_data:         RawData         | None   # written by scrape_node
    sentiment_result: SentimentResult | None   # written by sentiment_node
    trend_result:     TrendResult     | None   # written by trends_node


# ── Node functions ────────────────────────────────────────────────────────────

def scrape_node(state: AgentState) -> dict:
    """Collect all raw data — listings, review snippets, trend snippets."""
    logger.info("node.scrape", query=state["product_query"])
    raw = scrape_product(
        state["product_query"],
        use_sample_data=state["use_sample_data"],
    )
    return {"raw_data": raw}


def sentiment_node(state: AgentState) -> dict:
    """Analyse sentiment using raw_data.review_snippets from the scraper."""
    logger.info("node.sentiment", query=state["product_query"])
    result = analyze_sentiment(state["raw_data"])  # receives full RawData
    return {"sentiment_result": result}


def trends_node(state: AgentState) -> dict:
    """Analyse trends using raw_data.trend_snippets from the scraper."""
    logger.info("node.trends", query=state["product_query"])
    result = analyze_trends(state["raw_data"])  # receives full RawData
    return {"trend_result": result}


# ── Graph construction ────────────────────────────────────────────────────────

def create_agent():
    """Build and compile the LangGraph StateGraph.

    Topology: scrape → sentiment → trends → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("scrape",    scrape_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("trends",    trends_node)

    graph.add_edge(START,       "scrape")
    graph.add_edge("scrape",    "sentiment")
    graph.add_edge("sentiment", "trends")
    graph.add_edge("trends",    END)

    compiled = graph.compile()
    logger.info("agent.graph_compiled")
    return compiled


# ── Lifecycle (mirrors reference repo interface) ──────────────────────────────

def setup() -> None:
    """Validate configuration and warm-up check."""
    s = get_settings()
    if s.groq_api_key == "change-me":
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Copy .env.example → .env and add your key."
        )
    logger.info("agent.setup_ok", model=s.groq_model)


def index() -> None:
    """Placeholder for future vector-store indexing (data/documents/ → RAG)."""
    logger.info("agent.index", note="No vector store configured — skipping.")


# ── Public entry point ────────────────────────────────────────────────────────

async def chat(request: AnalysisRequest, graph=None) -> MarketReport:
    """Run the full pipeline and return a MarketReport.

    Steps:
      1. Run LangGraph graph  → collects ScraperResult, SentimentResult, TrendResult
      2. Call generate_report → Groq LLM synthesises the final report

    Args:
        request: Validated AnalysisRequest.
        graph:   Pre-compiled graph (compiled fresh if None).
    """
    if graph is None:
        graph = create_agent()

    log = logger.bind(query=request.product_query)
    log.info("agent.chat_start")

    initial_state: AgentState = {
        "product_query":    request.product_query,
        "use_sample_data":  request.use_sample_data,
        "raw_data":         None,
        "sentiment_result": None,
        "trend_result":     None,
    }

    final_state: AgentState = await graph.ainvoke(
        initial_state,
        config={"recursion_limit": agent_config.max_iterations},
    )

    # All three outputs must be present before report generation
    raw_data  = final_state["raw_data"]
    sentiment = final_state["sentiment_result"]
    trends    = final_state["trend_result"]

    if raw_data is None or sentiment is None or trends is None:
        missing = [k for k, v in {
            "raw_data": raw_data, "sentiment": sentiment, "trends": trends
        }.items() if v is None]
        raise RuntimeError(f"Graph completed but missing outputs: {missing}")

    # ── Post-graph step: report generation (outside the graph) ────────────────
    log.info("agent.generating_report")
    report = generate_report(
        product_query=request.product_query,
        raw_data=raw_data,
        sentiment=sentiment,
        trends=trends,
    )

    log.info("agent.chat_done", recommendations=len(report.strategic_recommendations))
    return report
