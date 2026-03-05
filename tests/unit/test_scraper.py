"""Unit tests for src/tools/scraper.py."""

from __future__ import annotations
from unittest.mock import patch
import pytest

from src.tools.scraper import _parse_price, _domain, scrape_product
from src.utils.schemas import RawData


class TestParsePrice:
    def test_dollar(self):          assert _parse_price("Buy for $129.99") == 129.99
    def test_euro(self):            assert _parse_price("Price: €99") == 99.0
    def test_usd_suffix(self):      assert _parse_price("249.00 USD") == 249.0
    def test_comma_thousands(self): assert _parse_price("$1,299.00") == 1299.0
    def test_no_price(self):        assert _parse_price("No price here") is None


class TestDomain:
    def test_www(self):     assert _domain("https://www.amazon.com/dp/B0123") == "amazon.com"
    def test_no_www(self):  assert _domain("https://ebay.com/item/1") == "ebay.com"
    def test_invalid(self): assert _domain("not-a-url") == "unknown"


class TestScrapeProductInjection:
    """Direct mock_data injection — no DDG, no file I/O."""

    def test_returns_raw_data(self):
        data = {
            "listings": [
                {"title": "A", "price": 100.0, "source": "amazon.com"},
                {"title": "B", "price": 200.0, "source": "ebay.com"},
            ],
            "review_snippets": ["Great product!", "Works well."],
            "trend_snippets":  ["Demand rising in 2025.", "Market growing fast."],
        }
        result = scrape_product("Test Product", mock_data=data)

        assert isinstance(result, RawData)
        assert result.avg_price == 150.0
        assert result.min_price == 100.0
        assert result.max_price == 200.0
        assert len(result.listings) == 2
        assert len(result.review_snippets) == 2
        assert len(result.trend_snippets) == 2
        assert result.source == "sample"

    def test_all_three_slices_present(self):
        """Downstream tools depend on all three slices being in the same object."""
        data = {
            "listings":        [{"title": "X", "price": 50.0, "source": "site.com"}],
            "review_snippets": ["Review text here"],
            "trend_snippets":  ["Trend text here"],
        }
        result = scrape_product("Slice Test", mock_data=data)
        assert result.listings[0].title == "X"
        assert result.review_snippets[0] == "Review text here"
        assert result.trend_snippets[0] == "Trend text here"

    def test_empty_slices(self):
        result = scrape_product("Empty", mock_data={"listings": [], "review_snippets": [], "trend_snippets": []})
        assert result.avg_price is None
        assert result.review_snippets == []
        assert result.trend_snippets == []

    def test_no_prices(self):
        data = {"listings": [{"title": "P", "price": None, "source": "site.com"}]}
        result = scrape_product("No Price", mock_data=data)
        assert result.avg_price is None


class TestScrapeProductSampleFile:
    """File-based sample data."""

    def test_known_product_has_all_slices(self):
        result = scrape_product("iPhone 16 Pro", use_sample_data=True)
        assert isinstance(result, RawData)
        assert len(result.listings) > 0
        assert len(result.review_snippets) > 0
        assert len(result.trend_snippets) > 0
        assert result.source == "sample"

    def test_unknown_product_returns_empty_raw_data(self):
        result = scrape_product("xyzzy-nonexistent-9999", use_sample_data=True)
        assert isinstance(result, RawData)
        assert result.source == "sample"


class TestScrapeProductLive:
    @pytest.mark.live
    @patch("src.tools.scraper.DDGS")
    def test_live_runs_three_queries(self, mock_ddgs):
        mock_ddgs.return_value.__enter__.return_value.text.return_value = [
            {"title": "iPhone $999", "href": "https://amazon.com/i", "body": "Price $999.00"},
        ]
        result = scrape_product("iPhone 16 Pro")
        assert isinstance(result, RawData)
        assert result.source == "live"
        # Three queries fired: listings + reviews + trends
        assert mock_ddgs.return_value.__enter__.return_value.text.call_count == 3
