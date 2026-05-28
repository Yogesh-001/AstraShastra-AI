from __future__ import annotations

from core.config import get_settings
from core.logging import get_logger

logger = get_logger(__name__)
_settings = get_settings()

# Lazy singleton - one client shared across the application
_client = None


async def get_inference_client():
    """
    Return the active inference client based on configuration.

    When ``VLLM_ENABLED=true`` the factory will attempt to connect to the
    vLLM OpenAI-compatible server and run a health-check.  If the check
    fails, it transparently falls back to Ollama so the pipeline never
    crashes due to a misconfigured serving backend.
    """
    global _client
    if _client is not None:
        return _client

    if _settings.vllm_enabled:
        from core.inference.vllm_client import VLLMClient

        vllm = VLLMClient(
            base_url=_settings.vllm_base_url,
            model=_settings.vllm_model,
            max_tokens=_settings.vllm_max_tokens,
            temperature=_settings.vllm_temperature,
        )
        if await vllm.health_check():
            logger.info("inference.backend_selected", backend="vllm", url=_settings.vllm_base_url)
            _client = vllm
            return _client
        logger.warning("inference.vllm_unhealthy_fallback_ollama")

    from core.inference.ollama_client import OllamaClient

    _client = OllamaClient(
        base_url=_settings.ollama_base_url,
        model=_settings.ollama_model,
    )
    logger.info("inference.backend_selected", backend="ollama", url=_settings.ollama_base_url)
    return _client


def reset_client() -> None:
    """Reset the cached client (useful for testing)."""
    global _client
    _client = None
