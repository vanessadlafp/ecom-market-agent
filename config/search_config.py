"""Web search and scraper configuration (analogous to opensearch_config.py)."""

from dataclasses import dataclass, field


@dataclass
class SearchConfig:
    ddg_max_results: int = 4 
    ddg_region: str = "wt-wt"
    ddg_safesearch: str = "moderate"

    request_timeout: int = 15
    user_agent: str = "market-agent/0.1 (research bot)"
    follow_redirects: bool = True

    pricing_sites: list[str] = field(default_factory=lambda: [
        "amazon.com", "ebay.com", "walmart.com", "bestbuy.com", "target.com",
    ])

    cache_ttl: int = 3600

    @property
    def pricing_site_filter(self) -> str:
        return " OR ".join(f"site:{s}" for s in self.pricing_sites)


search_config = SearchConfig()
