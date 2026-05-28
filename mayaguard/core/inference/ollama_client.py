from __future__ import annotations

import httpx

from core.logging import get_logger

logger = get_logger(__name__)


class OllamaClient:
    """
    Inference client wrapping the Ollama ``/api/generate`` REST endpoint.

    This is the default backend used when vLLM is not enabled or unreachable.
    """

    def __init__(self, base_url: str, model: str, timeout: float = 60) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    @property
    def backend_name(self) -> str:
        return "ollama"

    async def generate(self, prompt: str, *, model: str | None = None) -> str:
        """Send a prompt to Ollama and return the generated text."""
        target_model = model or self._model
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": target_model,
                        "prompt": prompt,
                        "stream": False,
                    },
                )
                resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as exc:
            logger.warning("ollama_client.generate_failed", error=str(exc))
            raise

    async def health_check(self) -> bool:
        """Return True if Ollama is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False
