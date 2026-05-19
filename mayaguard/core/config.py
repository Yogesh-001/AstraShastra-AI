"""
MayaGuard — Central configuration loaded from environment / .env file.
All modules import from here; nothing reads os.environ directly.
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral:7b-instruct-q4_K_M"
    ollama_embed_model: str = "nomic-embed-text"

    vllm_base_url: str = "http://localhost:8000"
    vllm_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    vllm_enabled: bool = False

    # ── Vector DB ─────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "mayaguard_core"
    qdrant_api_key: str = ""

    # ── Thresholds ────────────────────────────────────────────────
    hallucination_risk_threshold: float = Field(0.6, ge=0.0, le=1.0)
    entropy_high_threshold: float = 3.5
    entropy_medium_threshold: float = 2.5
    faithfulness_min_score: float = Field(0.7, ge=0.0, le=1.0)

    # ── API ───────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_workers: int = 1
    log_level: str = "INFO"

    # ── Observability ─────────────────────────────────────────────
    prometheus_enabled: bool = True
    prometheus_port: int = 9090


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings object (cached after first call)."""
    return Settings()
