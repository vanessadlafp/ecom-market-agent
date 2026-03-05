"""Agent settings — top-k, tool descriptions, max iterations."""

from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    max_iterations: int = 10
    temperature: float = 0.2

    tool_descriptions: dict[str, str] = field(default_factory=lambda: {
        "scrape_product": (
            "Searches the web via DuckDuckGo and scrapes e-commerce pages to collect "
            "pricing data and product listings. Accepts a product query string."
        ),
        "analyze_sentiment": (
            "Uses an LLM to score customer sentiment from review text. "
            "Returns sentiment label, score, key positives/negatives, and a summary."
        ),
        "analyze_trends": (
            "Returns market trend direction and data points for a product. "
            "Provides rising/stable/declining signal with period-level interest scores."
        ),
    })


agent_config = AgentConfig()
