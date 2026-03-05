"""Integration tests — agent graph state flow + FastAPI routes."""

from __future__ import annotations
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.utils.schemas import (
    AnalysisStatus, MarketReport, ProductListing, RawData,
    ReviewInsight, Sentiment, SentimentResult,
    TrendDirection, TrendPoint, TrendResult,
)


def _sample_raw_data() -> RawData:
    return RawData(
        product_query="Test Product",
        listings=[ProductListing(title="A", price=99.0, source="amazon.com")],
        avg_price=99.0, min_price=99.0, max_price=99.0,
        review_snippets=["Great product!"],
        trend_snippets=["Demand growing in 2025."],
        source="sample",
    )


def _sample_report() -> MarketReport:
    return MarketReport(
        product_query="Test Product",
        executive_summary="Strong market position.",
        pricing_analysis="Mid-range pricing at $99.",
        sentiment_analysis="Overwhelmingly positive.",
        market_trends="Rising demand in Q1 2025.",
        strategic_recommendations=["Expand DTC", "Invest in brand"],
        raw_data=_sample_raw_data(),
    )


@pytest.fixture
def client():
    with patch("src.api.routes.setup"), patch("src.api.routes.create_agent", return_value=None):
        from src.api.routes import app
        from src.api.cache import get_cache
        get_cache().clear()
        with TestClient(app) as c:
            yield c


# ── Agent state flow ──────────────────────────────────────────────────────────

class TestAgentStateFlow:
    """Verify the data separation contract:
    - scrape_node  populates raw_data
    - sentiment/trends nodes receive raw_data (not separate DDG calls)
    - generate_report receives typed objects post-graph
    """

    @pytest.mark.asyncio
    async def test_generate_report_receives_raw_data_object(self):
        from src.agent.react_agent import chat
        from src.utils.schemas import AnalysisRequest

        raw = _sample_raw_data()
        sentiment = SentimentResult(
            product_query="Test",
            insights=ReviewInsight(
                sentiment=Sentiment.POSITIVE, score=0.9,
                key_positives=["good"], key_negatives=[],
                summary="Great.",
            ),
        )
        trends = TrendResult(
            product_query="Test",
            trend_direction=TrendDirection.RISING,
            trend_points=[TrendPoint(period="Q1 2025", interest_score=80)],
            key_insights=["Growing"],
        )

        final_state = {
            "product_query": "Test", "use_sample_data": False,
            "raw_data": raw,
            "sentiment_result": sentiment,
            "trend_result": trends,
        }
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = final_state

        with patch("src.agent.react_agent.generate_report", return_value=_sample_report()) as mock_gen:
            await chat(AnalysisRequest(product_query="Test"), graph=mock_graph)

            kwargs = mock_gen.call_args.kwargs
            # Must receive typed RawData, not a ScraperResult or string
            assert isinstance(kwargs["raw_data"], RawData)
            assert isinstance(kwargs["sentiment"], SentimentResult)
            assert isinstance(kwargs["trends"], TrendResult)
            # RawData must carry both snippet slices
            assert kwargs["raw_data"].review_snippets == ["Great product!"]
            assert kwargs["raw_data"].trend_snippets == ["Demand growing in 2025."]

    @pytest.mark.asyncio
    async def test_missing_state_raises_runtime_error(self):
        from src.agent.react_agent import chat
        from src.utils.schemas import AnalysisRequest

        final_state = {
            "product_query": "Test", "use_sample_data": False,
            "raw_data": None,          # scrape_node failed
            "sentiment_result": None,
            "trend_result": None,
        }
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = final_state

        with pytest.raises(RuntimeError, match="missing outputs"):
            await chat(AnalysisRequest(product_query="Test"), graph=mock_graph)


# ── API routes ────────────────────────────────────────────────────────────────

class TestRoutes:
    @patch("src.api.routes.chat", new_callable=AsyncMock)
    def test_successful_analysis(self, mock_chat, client):
        mock_chat.return_value = _sample_report()
        r = client.post("/api/v1/analyze", json={"product_query": "Test Product"})
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "completed"
        assert d["report"]["executive_summary"] == "Strong market position."
        assert len(d["report"]["strategic_recommendations"]) == 2

    @patch("src.api.routes.chat", new_callable=AsyncMock)
    def test_error_returns_failed_status(self, mock_chat, client):
        mock_chat.side_effect = RuntimeError("Groq unavailable")
        r = client.post("/api/v1/analyze", json={"product_query": "Bad Query"})
        assert r.status_code == 200
        assert r.json()["status"] == "failed"
        assert "Groq unavailable" in r.json()["error"]

    @patch("src.api.routes.chat", new_callable=AsyncMock)
    def test_cache_prevents_double_call(self, mock_chat, client):
        mock_chat.return_value = _sample_report()
        client.post("/api/v1/analyze", json={"product_query": "Cached Product"})
        client.post("/api/v1/analyze", json={"product_query": "Cached Product"})
        assert mock_chat.call_count == 1

    @patch("src.api.routes.chat", new_callable=AsyncMock)
    def test_sample_data_flag_passed_through(self, mock_chat, client):
        mock_chat.return_value = _sample_report()
        client.post("/api/v1/analyze", json={"product_query": "Test", "use_sample_data": True})
        req = mock_chat.call_args.args[0]
        assert req.use_sample_data is True

    def test_short_query_rejected(self, client):
        r = client.post("/api/v1/analyze", json={"product_query": "x"})
        assert r.status_code == 422

    def test_health_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_ui_returns_html(self, client):
        r = client.get("/")
        assert "Market Analysis Agent" in r.text
