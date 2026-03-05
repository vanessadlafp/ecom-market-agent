"""Integration tests — agent graph and FastAPI routes."""

from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.utils.schemas import (
    AnalysisStatus, MarketReport, ScraperResult, SentimentResult,
    TrendResult, TrendDirection, ReviewInsight, Sentiment, ProductListing,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample_report() -> MarketReport:
    return MarketReport(
        product_query="Test Product",
        executive_summary="Strong market position.",
        pricing_analysis="Prices range $949–$1,099.",
        sentiment_analysis="Customers rate it positively.",
        market_trends="Rising demand over 4 quarters.",
        strategic_recommendations=["Rec 1", "Rec 2", "Rec 3", "Rec 4", "Rec 5"],
    )


# ── Agent graph ───────────────────────────────────────────────────────────────

class TestAgentGraph:
    def test_create_agent_compiles(self):
        from src.agent.react_agent import create_agent
        graph = create_agent()
        assert graph is not None

    def test_setup_raises_without_api_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "change-me")
        # Clear the lru_cache so env var is re-read
        from config.settings import get_settings
        get_settings.cache_clear()
        from src.agent.react_agent import setup
        with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
            setup()
        get_settings.cache_clear()

    @pytest.mark.asyncio
    @patch("src.agent.react_agent.generate_report")
    async def test_chat_calls_report_after_graph(self, mock_report):
        """Report generator must be called with typed results, not JSON strings."""
        mock_report.return_value = _sample_report()

        from src.agent.react_agent import create_agent, chat
        from src.utils.schemas import AnalysisRequest

        graph = create_agent()

        # Patch the three node functions so graph runs instantly
        with (
            patch("src.agent.react_agent.scrape_product") as ms,
            patch("src.agent.react_agent.analyze_sentiment") as mse,
            patch("src.agent.react_agent.analyze_trends") as mt,
        ):
            ms.return_value  = ScraperResult(product_query="T", source="sample")
            mse.return_value = SentimentResult(
                product_query="T",
                insights=ReviewInsight(sentiment=Sentiment.POSITIVE, score=0.8, summary="Good."),
            )
            mt.return_value = TrendResult(
                product_query="T", trend_direction=TrendDirection.RISING, key_insights=["up"],
            )

            report = await chat(AnalysisRequest(product_query="Test Product"), graph=graph)

        assert isinstance(report, MarketReport)
        # generate_report must have received typed objects, not strings
        call_kwargs = mock_report.call_args.kwargs
        assert isinstance(call_kwargs["scraper"],   ScraperResult)
        assert isinstance(call_kwargs["sentiment"], SentimentResult)
        assert isinstance(call_kwargs["trends"],    TrendResult)


# ── API routes ────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    with (
        patch("src.api.routes.setup"),
        patch("src.api.routes.create_agent", return_value=None),
    ):
        from src.api.routes import app
        from src.api.cache import get_cache
        get_cache().clear()
        from fastapi.testclient import TestClient
        with TestClient(app) as c:
            yield c


class TestRoutes:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_ui_renders(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "Market Analysis Agent" in r.text

    @patch("src.api.routes.chat", new_callable=AsyncMock)
    def test_successful_analysis(self, mock_chat, client):
        mock_chat.return_value = _sample_report()
        r = client.post("/api/v1/analyze", json={"product_query": "iPhone 16 Pro"})
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "completed"
        assert len(d["report"]["strategic_recommendations"]) == 5

    @patch("src.api.routes.chat", new_callable=AsyncMock)
    def test_sample_data_flag_forwarded(self, mock_chat, client):
        mock_chat.return_value = _sample_report()
        client.post("/api/v1/analyze", json={
            "product_query": "Test", "use_sample_data": True
        })
        req = mock_chat.call_args.args[0]
        assert req.use_sample_data is True

    @patch("src.api.routes.chat", new_callable=AsyncMock)
    def test_cache_prevents_double_call(self, mock_chat, client):
        mock_chat.return_value = _sample_report()
        client.post("/api/v1/analyze", json={"product_query": "Cached Product"})
        client.post("/api/v1/analyze", json={"product_query": "Cached Product"})
        assert mock_chat.call_count == 1

    @patch("src.api.routes.chat", new_callable=AsyncMock)
    def test_failed_analysis(self, mock_chat, client):
        mock_chat.side_effect = RuntimeError("Groq unavailable")
        r = client.post("/api/v1/analyze", json={"product_query": "Broken"})
        d = r.json()
        assert d["status"] == "failed"
        assert "Groq unavailable" in d["error"]

    def test_short_query_rejected(self, client):
        r = client.post("/api/v1/analyze", json={"product_query": "x"})
        assert r.status_code == 422
