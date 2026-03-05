"""Report Generator — synthesises all tool outputs into a MarketReport.

Called POST-GRAPH by react_agent.chat() — never inside the graph itself.
Receives typed objects, builds a structured context, calls Groq LLM once.
"""

from __future__ import annotations

import json

from langchain_groq import ChatGroq

from config.settings import get_settings
from src.utils.logger import get_logger
from src.utils.schemas import MarketReport, RawData, SentimentResult, TrendResult

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a senior e-commerce market analyst writing a strategic intelligence report.
You will receive structured JSON data from three research tools.
Respond ONLY with valid JSON — no markdown, no preamble.

Required schema:
{
  "executive_summary":         "<3-4 sentence paragraph>",
  "pricing_analysis":          "<detailed paragraph about pricing landscape>",
  "sentiment_analysis":        "<detailed paragraph about customer sentiment>",
  "market_trends":             "<detailed paragraph about market direction>",
  "strategic_recommendations": ["<actionable recommendation>", ...]
}

Provide 4-6 recommendations ordered by priority. Be specific and data-driven."""


def _build_context(
    product_query: str,
    raw_data:  RawData | None,
    sentiment: SentimentResult | None,
    trends:    TrendResult | None,
) -> str:
    ctx: dict = {"product": product_query}

    if raw_data:
        ctx["pricing"] = {
            "avg_price":    raw_data.avg_price,
            "min_price":    raw_data.min_price,
            "max_price":    raw_data.max_price,
            "listing_count": len(raw_data.listings),
            "data_source":  raw_data.source,
            "sample_listings": [
                {"title": l.title, "price": l.price, "source": l.source}
                for l in raw_data.listings[:5]
            ],
        }

    if sentiment:
        ins = sentiment.insights
        ctx["sentiment"] = {
            "overall":   ins.sentiment,
            "score":     ins.score,
            "positives": ins.key_positives,
            "negatives": ins.key_negatives,
            "summary":   ins.summary,
        }

    if trends:
        ctx["trends"] = {
            "direction":   trends.trend_direction,
            "data_source": trends.source,
            "insights":    trends.key_insights,
            "data_points": [tp.model_dump() for tp in trends.trend_points],
        }

    return json.dumps(ctx, indent=2, default=str)


# ── Public interface ──────────────────────────────────────────────────────────

def generate_report(
    product_query: str,
    raw_data:  RawData | None = None,
    sentiment: SentimentResult | None = None,
    trends:    TrendResult | None = None,
) -> MarketReport:
    """Synthesise all tool outputs into a final MarketReport.

    Args:
        product_query: The product being analysed.
        raw_data:      RawData from scraper.py (pricing + provenance).
        sentiment:     SentimentResult from sentiment.py.
        trends:        TrendResult from trends.py.
    """
    log = logger.bind(tool="report", query=product_query)
    log.info("report.start")

    context = _build_context(product_query, raw_data, sentiment, trends)
    settings = get_settings()

    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.3,
    )
    response = llm.invoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Research data:\n{context}"},
    ])

    raw = (
        response.content.strip()
        .removeprefix("```json").removeprefix("```")
        .removesuffix("```").strip()
    )

    try:
        data = json.loads(raw)
        report = MarketReport(
            product_query=product_query,
            executive_summary=data["executive_summary"],
            pricing_analysis=data["pricing_analysis"],
            sentiment_analysis=data["sentiment_analysis"],
            market_trends=data["market_trends"],
            strategic_recommendations=data.get("strategic_recommendations", []),
            raw_data=raw_data,
            raw_sentiment=sentiment,
            raw_trends=trends,
        )
    except Exception as exc:
        log.error("report.parse_failed", error=str(exc))
        report = MarketReport(
            product_query=product_query,
            executive_summary="Report generation failed — raw outputs attached.",
            pricing_analysis="", sentiment_analysis="", market_trends="",
            raw_data=raw_data,
            raw_sentiment=sentiment,
            raw_trends=trends,
        )

    log.info("report.done", recommendations=len(report.strategic_recommendations))
    return report
