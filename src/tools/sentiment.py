"""Sentiment Analyzer Tool — consumes RawData.review_snippets, runs Groq LLM.

Responsibility
--------------
Receives the RawData collected by scraper.py and analyses the review_snippets
slice. No DDG calls here — data collection is fully handled upstream.

If review_snippets is empty (e.g. DDG returned nothing for reviews), the LLM
falls back to general product knowledge.
"""

from __future__ import annotations

import json

from langchain_groq import ChatGroq

from config import llm_config
from config.settings import get_settings
from src.utils.logger import get_logger
from src.utils.schemas import RawData, ReviewInsight, Sentiment, SentimentResult

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a market research expert specialising in customer sentiment analysis.
Analyse the provided review snippets and respond ONLY with valid JSON — no markdown, no preamble.

Required schema:
{
  "sentiment": "positive" | "neutral" | "negative",
  "score": <float 0.0–1.0>,
  "key_positives": ["<point>", ...],
  "key_negatives": ["<point>", ...],
  "summary": "<2-3 sentence paragraph>"
}

Be specific and grounded in the review content. Provide 3-5 items per list."""


def _call_llm(product_query: str, reviews_text: str) -> ReviewInsight:
    settings = get_settings()
    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=llm_config.temperature,
    )
    response = llm.invoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Product: {product_query}\n\nReviews:\n{reviews_text}"},
    ])
    raw = (
        response.content.strip()
        .removeprefix("```json").removeprefix("```")
        .removesuffix("```").strip()
    )
    data = json.loads(raw)
    return ReviewInsight(
        sentiment=Sentiment(data["sentiment"]),
        score=float(data["score"]),
        key_positives=data.get("key_positives", []),
        key_negatives=data.get("key_negatives", []),
        summary=data["summary"],
    )


# ── Public interface ──────────────────────────────────────────────────────────

def analyze_sentiment(raw_data: RawData) -> SentimentResult:
    """Analyse sentiment using review snippets from RawData.

    Args:
        raw_data: The RawData collected by scraper.py.
                  Reads raw_data.review_snippets.
    """
    log = logger.bind(tool="sentiment", query=raw_data.product_query)
    log.info("sentiment.start", snippets=len(raw_data.review_snippets))

    reviews = raw_data.review_snippets
    if reviews:
        reviews_text = "\n\n---\n\n".join(reviews[:6])
    else:
        log.warning("sentiment.no_reviews", note="Falling back to general LLM knowledge")
        reviews_text = (
            f"No review snippets available. Use your general knowledge about "
            f"customer sentiment for: {raw_data.product_query}"
        )

    try:
        insight = _call_llm(raw_data.product_query, reviews_text)
    except Exception as exc:
        log.error("sentiment.llm_failed", error=str(exc))
        insight = ReviewInsight(
            sentiment=Sentiment.NEUTRAL,
            score=0.5,
            summary=f"Sentiment analysis failed: {exc}",
        )

    result = SentimentResult(
        product_query=raw_data.product_query,
        insights=insight,
        sample_reviews=reviews[:3],
    )
    log.info("sentiment.done", sentiment=insight.sentiment, score=insight.score)
    return result
