from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM configuration (Ollama - default backend)
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "mistral:7b-instruct-q4_K_M"
    ollama_embed_model: str = "nomic-embed-text"

    # vLLM configuration (production GPU serving)
    vllm_enabled: bool = False
    vllm_base_url: str = "http://127.0.0.1:8000"
    vllm_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    vllm_max_tokens: int = 512
    vllm_temperature: float = 0.3

    # Vector Database
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "mayaguard_core"
    qdrant_api_key: str = ""

    # Thresholds and safety parameters
    hallucination_risk_threshold: float = Field(0.6, ge=0.0, le=1.0)
    entropy_high_threshold: float = 3.5
    entropy_medium_threshold: float = 2.5
    faithfulness_min_score: float = Field(0.7, ge=0.0, le=1.0)

    # Serving API configurations
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_workers: int = 1
    log_level: str = "INFO"

    # Observability
    prometheus_enabled: bool = True
    prometheus_port: int = 9090

    # Fine-tuned models
    finetuned_verifier_path: str = ""
    finetuned_verifier_enabled: bool = False
    qlora_adapter_path: str = ""
    qlora_adapter_enabled: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
