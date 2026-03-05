"""Runtime settings — loaded from environment / .env file."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    groq_api_key: str = "change-me"
    groq_model: str = "llama-3.3-70b-versatile"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False

    # Agent
    agent_max_iterations: int = 4
    scraper_max_results: int = 4
    scraper_timeout: int = 15
    cache_ttl_seconds: int = 3600

    # Feature flags
    use_sample_data: bool = False   # True → scraper uses data/samples/ instead of live DDG
    report_language: str = "en"     # "en" | "fr" — language for LLM report output


@lru_cache
def get_settings() -> Settings:
    return Settings()