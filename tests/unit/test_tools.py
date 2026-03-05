"""Unit tests for sentiment, trends, and report tools."""

from __future__ import annotations
import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.utils.schemas import (
    MarketReport, ProductListing, RawData,
    ReviewInsight, Sentiment, SentimentResult,
    TrendDirection, TrendPoint, TrendResult,
)


def _make_raw_data(
    product_query: str = "Test Product",
    review_snippets: list[str] | None = None,
    trend_snippets: list[str] | None = None,
) -> RawData:
    return RawData(
        product_query=product_query,
        listings=[ProductListing(title="Product A", price=99.0, source="amazon.com")],
        avg_price=99.0, min_price=99.0, max_price=99.0,
        review_snippets=review_snippets or ["Great product, highly recommend!"],
        trend_snippets=trend_snippets or ["Demand rising sharply in Q1 2025."],
        source="sample",
    )


# ── Sentiment ─────────────────────────────────────────────────────────────────

class TestSentiment:
    def _llm_returning(self, payload: dict):
        msg = MagicMock()
        msg.content = json.dumps(payload)
        llm = MagicMock()
        llm.invoke.return_value = msg
        return llm

    @patch("src.tools.sentiment.ChatGroq")
    def test_consumes_review_snippets_from_raw_data(self, mock_groq):
        """Sentiment tool must read raw_data.review_snippets — no DDG calls."""
        mock_groq.return_value = self._llm_returning({
            "sentiment": "positive", "score": 0.88,
            "key_positives": ["great battery", "nice design"],
            "key_negatives": ["expensive"],
            "summary": "Customers love it.",
        })
        raw = _make_raw_data(review_snippets=["Really great phone!", "Best camera ever."])

        from src.tools.sentiment import analyze_sentiment
        result = analyze_sentiment(raw)

        assert isinstance(result, SentimentResult)
        assert result.insights.sentiment == Sentiment.POSITIVE
        assert result.insights.score == pytest.approx(0.88)
        # Confirm the review snippets made it into sample_reviews
        assert result.sample_reviews == ["Really great phone!", "Best camera ever."]

    @patch("src.tools.sentiment.ChatGroq")
    def test_empty_reviews_falls_back_gracefully(self, mock_groq):
        """Empty review_snippets should still call LLM with fallback text."""
        mock_groq.return_value = self._llm_returning({
            "sentiment": "neutral", "score": 0.5,
            "key_positives": [], "key_negatives": [],
            "summary": "General knowledge only.",
        })
        raw = _make_raw_data(review_snippets=[])

        from src.tools.sentiment import analyze_sentiment
        result = analyze_sentiment(raw)
        assert result.insights.sentiment == Sentiment.NEUTRAL
        assert mock_groq.return_value.invoke.called

    @patch("src.tools.sentiment.ChatGroq")
    def test_bad_json_returns_neutral_fallback(self, mock_groq):
        bad = MagicMock()
        bad.content = "not json at all"
        mock_groq.return_value.invoke.return_value = bad

        from src.tools.sentiment import analyze_sentiment
        result = analyze_sentiment(_make_raw_data())
        assert result.insights.sentiment == Sentiment.NEUTRAL
        assert "failed" in result.insights.summary.lower()


# ── Trends ────────────────────────────────────────────────────────────────────

class TestTrends:
    def _llm_returning(self, payload: dict):
        msg = MagicMock()
        msg.content = json.dumps(payload)
        llm = MagicMock()
        llm.invoke.return_value = msg
        return llm

    @patch("src.tools.trends.ChatGroq")
    def test_consumes_trend_snippets_from_raw_data(self, mock_groq):
        """Trends tool must read raw_data.trend_snippets — no DDG calls."""
        mock_groq.return_value = self._llm_returning({
            "trend_direction": "rising",
            "trend_points": [{"period": "Q1 2025", "interest_score": 78, "notes": "Peak demand"}],
            "key_insights": ["Growing fast", "New entrants"],
        })
        raw = _make_raw_data(trend_snippets=["Demand surging in 2025.", "Market share growing."])

        from src.tools.trends import analyze_trends
        result = analyze_trends(raw)

        assert isinstance(result, TrendResult)
        assert result.trend_direction == TrendDirection.RISING
        assert len(result.trend_points) == 1
        assert result.source == "sample"

    @patch("src.tools.trends.ChatGroq")
    def test_empty_snippets_falls_back_gracefully(self, mock_groq):
        mock_groq.return_value = self._llm_returning({
            "trend_direction": "stable",
            "trend_points": [],
            "key_insights": ["No data available"],
        })
        raw = _make_raw_data(trend_snippets=[])

        from src.tools.trends import analyze_trends
        result = analyze_trends(raw)
        assert result.trend_direction == TrendDirection.STABLE
        assert mock_groq.return_value.invoke.called

    @patch("src.tools.trends.ChatGroq")
    def test_bad_json_returns_stable_fallback(self, mock_groq):
        bad = MagicMock()
        bad.content = "totally not json"
        mock_groq.return_value.invoke.return_value = bad

        from src.tools.trends import analyze_trends
        result = analyze_trends(_make_raw_data())
        assert result.trend_direction == TrendDirection.STABLE
        assert "failed" in result.key_insights[0].lower()


# ── Report ────────────────────────────────────────────────────────────────────

class TestReport:
    def _make_sentiment(self) -> SentimentResult:
        return SentimentResult(
            product_query="Test Product",
            insights=ReviewInsight(
                sentiment=Sentiment.POSITIVE, score=0.8,
                key_positives=["great value"],
                key_negatives=["slow shipping"],
                summary="Customers are satisfied.",
            ),
        )

    def _make_trends(self) -> TrendResult:
        return TrendResult(
            product_query="Test Product",
            trend_direction=TrendDirection.RISING,
            trend_points=[TrendPoint(period="Q1 2025", interest_score=75)],
            key_insights=["Growing demand"],
        )

    @patch("src.tools.report.ChatGroq")
    def test_receives_raw_data_not_scraper_result(self, mock_groq):
        """Report generator must accept RawData, not a ScraperResult."""
        mock_msg = MagicMock()
        mock_msg.content = json.dumps({
            "executive_summary": "Strong market.",
            "pricing_analysis": "Mid-range pricing.",
            "sentiment_analysis": "Positive reviews.",
            "market_trends": "Rising demand.",
            "strategic_recommendations": ["Invest now", "Expand channels"],
        })
        mock_groq.return_value.invoke.return_value = mock_msg

        from src.tools.report import generate_report
        raw = _make_raw_data()
        report = generate_report("Test Product", raw_data=raw,
                                 sentiment=self._make_sentiment(),
                                 trends=self._make_trends())

        assert isinstance(report, MarketReport)
        assert report.raw_data is raw          # typed RawData attached
        assert report.raw_data.review_snippets  # snippets preserved
        assert len(report.strategic_recommendations) == 2

    @patch("src.tools.report.ChatGroq")
    def test_works_with_all_none_inputs(self, mock_groq):
        mock_msg = MagicMock()
        mock_msg.content = json.dumps({
            "executive_summary": "Partial.",
            "pricing_analysis": "", "sentiment_analysis": "", "market_trends": "",
            "strategic_recommendations": ["Gather more data"],
        })
        mock_groq.return_value.invoke.return_value = mock_msg

        from src.tools.report import generate_report
        report = generate_report("Sparse Product")
        assert isinstance(report, MarketReport)
        assert report.raw_data is None
