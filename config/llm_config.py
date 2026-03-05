"""LLM and embedding model settings."""

from dataclasses import dataclass


@dataclass
class LLMConfig:
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    api_key_env: str = "GROQ_API_KEY"
    fast_model: str = "llama-3.1-8b-instant"  # used for sentiment + trends (short JSON)
    temperature: float = 0.2
    max_tokens: int = 800  # for sentiment/trends; reduces Groq quota usage
    report_max_tokens: int = 2048  # report needs longer output; uses main model
    # Embedding placeholder for future RAG over data/documents/
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384


llm_config = LLMConfig()
