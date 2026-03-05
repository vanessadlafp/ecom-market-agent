"""Web Scraper Tool — single data-collection pass, three query types.

Responsibility
--------------
This is the only tool that touches external data sources (DDG / sample files).
It runs three focused queries and returns a RawData object containing:

  listings        → pricing data consumed by report.py
  review_snippets → text consumed by sentiment.py
  trend_snippets  → text consumed by trends.py

Downstream analysis tools receive this RawData and never call DDG themselves.

Two modes
---------
LIVE   — real DDG queries (production)
SAMPLE — loads from data/samples/scraper_samples.json (USE_MOCK_SCRAPER=true
         in .env, or pass use_sample_data=True / inject mock_data in tests)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from config import search_config
from config.settings import get_settings
from src.utils.logger import get_logger
from src.utils.schemas import ProductListing, RawData

logger = get_logger(__name__)

_SAMPLES_PATH = Path(__file__).parents[2] / "data" / "samples" / "scraper_samples.json"

_PRICE_RE = re.compile(
    r"(?:[$€£¥])\s*[\d,]+(?:\.\d{1,2})?|[\d,]+(?:\.\d{1,2})?\s*(?:USD|EUR|GBP|CAD|AUD)",
    re.IGNORECASE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_price(text: str) -> float | None:
    m = _PRICE_RE.search(text)
    if not m:
        return None
    try:
        return float(re.sub(r"[^\d.]", "", m.group()))
    except ValueError:
        return None


def _domain(url: str) -> str:
    m = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return m.group(1) if m else "unknown"


def _fetch_text(url: str) -> str:
    try:
        with httpx.Client(
            timeout=search_config.request_timeout,
            follow_redirects=search_config.follow_redirects,
        ) as client:
            r = client.get(url, headers={"User-Agent": search_config.user_agent})
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            return soup.get_text(" ", strip=True)
    except Exception as exc:
        logger.warning("scraper.fetch_failed", url=url, error=str(exc))
        return ""


def _ddg_query(query: str, max_results: int) -> list[dict]:
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as exc:
        logger.warning("scraper.ddg_failed", query=query, error=str(exc))
        return []


def _compute_stats(listings: list[ProductListing]) -> dict:
    prices = [l.price for l in listings if l.price is not None]
    return {
        "avg_price": round(sum(prices) / len(prices), 2) if prices else None,
        "min_price": min(prices) if prices else None,
        "max_price": max(prices) if prices else None,
    }


# ── Live collection — three focused DDG queries ───────────────────────────────

def _collect_listings(query: str, max_results: int) -> list[ProductListing]:
    """Query 1 — pricing / product listings."""
    raw = _ddg_query(
        f"{query} buy price {search_config.pricing_site_filter}",
        max_results,
    )
    listings = []
    for r in raw:
        url = r.get("href") or r.get("url", "")
        snippet = r.get("body", "")
        price = _parse_price(snippet)
        if price is None and url:
            price = _parse_price(_fetch_text(url))
        listings.append(ProductListing(
            title=r.get("title", "Unknown"),
            price=price,
            source=_domain(url),
            url=url,
            description=snippet[:300] or None,
        ))
    return listings


def _collect_reviews(query: str, max_results: int) -> list[str]:
    """Query 2 — customer review snippets for sentiment analysis."""
    raw = _ddg_query(f"{query} customer reviews pros cons", max_results)
    return [r["body"] for r in raw if r.get("body")]


def _collect_trend_snippets(query: str, max_results: int) -> list[str]:
    """Query 3 — market/news snippets for trend analysis."""
    raw = _ddg_query(
        f"{query} market trend 2024 2025 popularity demand growth",
        max_results,
    )
    return [r["body"] for r in raw if r.get("body")]


def _scrape_live(product_query: str) -> RawData:
    log = logger.bind(tool="scraper", mode="live", query=product_query)
    log.info("scraper.start")

    max_r = get_settings().scraper_max_results

    listings        = _collect_listings(product_query, max_r)
    review_snippets = _collect_reviews(product_query, max_r)
    trend_snippets  = _collect_trend_snippets(product_query, max_r)

    stats = _compute_stats(listings)
    log.info(
        "scraper.done",
        listings=len(listings),
        reviews=len(review_snippets),
        trends=len(trend_snippets),
        avg_price=stats["avg_price"],
    )
    return RawData(
        product_query=product_query,
        listings=listings,
        review_snippets=review_snippets,
        trend_snippets=trend_snippets,
        source="live",
        **stats,
    )


# ── Sample / mock collection ──────────────────────────────────────────────────

def _load_sample(product_query: str) -> RawData:
    """Load deterministic sample data from JSON files."""
    log = logger.bind(tool="scraper", mode="sample", query=product_query)

    try:
        samples = json.loads(_SAMPLES_PATH.read_text())
        key = product_query.lower().strip()
        data = samples["products"].get(key) or next(
            (v for k, v in samples["products"].items() if k in key or key in k),
            None,
        )
    except Exception as exc:
        log.warning("scraper.sample_load_failed", error=str(exc))
        data = None

    if data is None:
        log.warning("scraper.sample_no_match", query=product_query)
        return RawData(product_query=product_query, source="sample")

    listings = [ProductListing(**item) for item in data.get("listings", [])]
    stats = _compute_stats(listings)

    log.info("scraper.done", listings=len(listings))
    return RawData(
        product_query=product_query,
        listings=listings,
        review_snippets=data.get("review_snippets", []),
        trend_snippets=data.get("trend_snippets", []),
        source="sample",
        **stats,
    )


# ── Public interface ──────────────────────────────────────────────────────────

def scrape_product(
    product_query: str,
    *,
    use_sample_data: bool = False,
    mock_data: dict | None = None,
) -> RawData:
    """Collect all raw data for a product query.

    Returns a RawData containing listings, review_snippets, and trend_snippets.
    Downstream tools (sentiment, trends) consume the relevant slices.

    Args:
        product_query:   What to search for.
        use_sample_data: Load from data/samples/ instead of live DDG.
        mock_data:       Inject a full RawData-compatible dict (unit tests).
                         Implies use_sample_data.
    """
    # Direct injection for unit tests
    if mock_data is not None:
        listings = [ProductListing(**item) for item in mock_data.get("listings", [])]
        stats = _compute_stats(listings)
        return RawData(
            product_query=product_query,
            listings=listings,
            review_snippets=mock_data.get("review_snippets", []),
            trend_snippets=mock_data.get("trend_snippets", []),
            source="sample",
            **stats,
        )
        
    if use_sample_data or get_settings().use_sample_data:
        return _load_sample(product_query)

    return _scrape_live(product_query)
