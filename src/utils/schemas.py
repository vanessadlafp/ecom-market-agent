"""Shared Pydantic domain models.

Data flow
---------
scraper.py   →  RawData         (all collected text: listings, reviews, trend snippets)
sentiment.py →  SentimentResult (consumes RawData.review_snippets)
trends.py    →  TrendResult     (consumes RawData.trend_snippets)
report.py    →  MarketReport    (consumes all three — post-graph)

AgentState carries typed objects throughout — no JSON strings between nodes.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class AnalysisStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL  = "neutral"
    NEGATIVE = "negative"


class TrendDirection(str, Enum):
    RISING    = "rising"
    STABLE    = "stable"
    DECLINING = "declining"


# ── Scraper — raw collected data ──────────────────────────────────────────────

class ProductListing(BaseModel):
    """A single priced product listing from the web."""
    title:       str
    price:       float | None = None
    currency:    str = "USD"
    source:      str
    url:         str | None = None
    description: str | None = None


class RawData(BaseModel):
    """Everything the scraper collected in one place.

    Downstream tools pull exactly the slice they need:
      sentiment.py  → review_snippets
      trends.py     → trend_snippets
      report.py     → listings + price stats (via ScraperResult)
    """
    product_query:   str
    # Pricing
    listings:        list[ProductListing] = Field(default_factory=list)
    avg_price:       float | None = None
    min_price:       float | None = None
    max_price:       float | None = None
    # Review text for sentiment analysis
    review_snippets: list[str] = Field(default_factory=list)
    # Market/news text for trend analysis
    trend_snippets:  list[str] = Field(default_factory=list)
    # Provenance
    source:          str = "live"   # "live" | "sample"
    collected_at:    datetime = Field(default_factory=datetime.utcnow)


class ScraperResult(BaseModel):
    """Pricing summary derived from RawData — passed to the report generator."""
    product_query: str
    listings:      list[ProductListing] = Field(default_factory=list)
    avg_price:     float | None = None
    min_price:     float | None = None
    max_price:     float | None = None
    source:        str = "live"
    scraped_at:    datetime = Field(default_factory=datetime.utcnow)

    @classmethod
    def from_raw(cls, raw: RawData) -> "ScraperResult":
        return cls(
            product_query=raw.product_query,
            listings=raw.listings,
            avg_price=raw.avg_price,
            min_price=raw.min_price,
            max_price=raw.max_price,
            source=raw.source,
        )


# ── Sentiment ─────────────────────────────────────────────────────────────────

class ReviewInsight(BaseModel):
    sentiment:     Sentiment
    score:         float = Field(ge=0.0, le=1.0)
    key_positives: list[str] = Field(default_factory=list)
    key_negatives: list[str] = Field(default_factory=list)
    summary:       str


class SentimentResult(BaseModel):
    product_query:  str
    insights:       ReviewInsight
    sample_reviews: list[str] = Field(default_factory=list)
    analyzed_at:    datetime = Field(default_factory=datetime.utcnow)


# ── Trends ────────────────────────────────────────────────────────────────────

class TrendPoint(BaseModel):
    period:         str
    interest_score: float = Field(ge=0.0, le=100.0)
    notes:          str | None = None


class TrendResult(BaseModel):
    product_query:   str
    trend_direction: TrendDirection
    trend_points:    list[TrendPoint] = Field(default_factory=list)
    key_insights:    list[str]        = Field(default_factory=list)
    source:          str = "live"
    analyzed_at:     datetime = Field(default_factory=datetime.utcnow)


# ── Report (produced OUTSIDE the graph) ───────────────────────────────────────

class MarketReport(BaseModel):
    id:                        UUID = Field(default_factory=uuid4)
    product_query:             str
    executive_summary:         str
    pricing_analysis:          str
    sentiment_analysis:        str
    market_trends:             str
    strategic_recommendations: list[str] = Field(default_factory=list)
    # Full outputs attached for transparency / debugging
    raw_data:      RawData         | None = None
    raw_sentiment: SentimentResult | None = None
    raw_trends:    TrendResult     | None = None
    generated_at:  datetime = Field(default_factory=datetime.utcnow)


# ── API request / response ────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    product_query: str = Field(
        ..., min_length=2, max_length=200,
        examples=["iPhone 16 Pro", "Nike Air Max 2025"],
    )
    use_sample_data: bool = Field(
        default=False,
        description="When True the scraper reads from data/samples/ instead of live DDG.",
    )
    language: str = Field(
        default="en",
        description="Report language — 'en' for English, 'fr' for French.",
    )


class AnalysisResponse(BaseModel):
    request_id:       UUID = Field(default_factory=uuid4)
    status:           AnalysisStatus = AnalysisStatus.COMPLETED
    product_query:    str
    report:           MarketReport | None = None
    error:            str | None = None
    duration_seconds: float | None = None