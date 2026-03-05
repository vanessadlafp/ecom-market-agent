"""Market Trend Analyzer Tool — consumes RawData.trend_snippets, runs Groq LLM.

Responsibility
--------------
Receives the RawData collected by scraper.py and analyses the trend_snippets
slice. No DDG calls here — data collection is fully handled upstream.

This replaces the previous mock-only approach: real snippets are collected by
the scraper and analysed here by the LLM, giving genuine trend signals.

If trend_snippets is empty, the LLM falls back to general market knowledge.
"""

from __future__ import annotations

import json

from langchain_groq import ChatGroq

from config import llm_config
from config.settings import get_settings
from src.utils.logger import get_logger
from src.utils.schemas import RawData, TrendDirection, TrendPoint, TrendResult

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a senior e-commerce market trend analyst.
Analyse the provided market/news article snippets and respond ONLY with valid JSON — no markdown, no preamble.

Required schema:
{
  "trend_direction": "rising" | "stable" | "declining",
  "trend_points": [
    {"period": "<e.g. Q1 2025>", "interest_score": <0-100>, "notes": "<optional>"},
    ...
  ],
  "key_insights": ["<insight>", ...]
}

Provide 3-5 trend_points and 3-5 key_insights grounded in the snippets."""


def _call_llm(product_query: str, snippets_text: str) -> dict:
    settings = get_settings()
    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=llm_config.temperature,
    )
    response = llm.invoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Product: {product_query}\n\nMarket articles:\n{snippets_text}"},
    ])
    raw = (
        response.content.strip()
        .removeprefix("```json").removeprefix("```")
        .removesuffix("```").strip()
    )
    return json.loads(raw)


# ── Public interface ──────────────────────────────────────────────────────────

def analyze_trends(raw_data: RawData) -> TrendResult:
    """Analyse market trends using trend snippets from RawData.

    Args:
        raw_data: The RawData collected by scraper.py.
                  Reads raw_data.trend_snippets.
    """
    log = logger.bind(tool="trends", query=raw_data.product_query)
    log.info("trends.start", snippets=len(raw_data.trend_snippets))

    snippets = raw_data.trend_snippets
    if snippets:
        snippets_text = "\n\n---\n\n".join(snippets[:6])
    else:
        log.warning("trends.no_snippets", note="Falling back to general LLM knowledge")
        snippets_text = (
            f"No trend snippets available. Use your general market knowledge "
            f"about demand and trends for: {raw_data.product_query}"
        )

    try:
        data = _call_llm(raw_data.product_query, snippets_text)
        result = TrendResult(
            product_query=raw_data.product_query,
            trend_direction=TrendDirection(data["trend_direction"]),
            trend_points=[
                TrendPoint(
                    period=tp["period"],
                    interest_score=float(tp["interest_score"]),
                    notes=tp.get("notes"),
                )
                for tp in data.get("trend_points", [])
            ],
            key_insights=data.get("key_insights", []),
            source=raw_data.source,
        )
    except Exception as exc:
        log.error("trends.llm_failed", error=str(exc))
        result = TrendResult(
            product_query=raw_data.product_query,
            trend_direction=TrendDirection.STABLE,
            key_insights=[f"Trend analysis failed: {exc}"],
            source=raw_data.source,
        )

    log.info("trends.done", direction=result.trend_direction, insights=len(result.key_insights))
    return result
