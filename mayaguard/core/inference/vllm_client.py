from __future__ import annotations

import httpx

from core.logging import get_logger

logger = get_logger(__name__)


class VLLMClient:
    """
    Inference client for vLLM's OpenAI-compatible ``/v1/completions`` API.

    vLLM provides significantly higher throughput than Ollama for production
    workloads through continuous batching, PagedAttention, and optional
    LoRA adapter serving.  This client targets the OpenAI-compatible
    endpoints that vLLM exposes by default.

    Usage::

        client = VLLMClient(base_url="http://localhost:8000", model="mistralai/Mistral-7B-Instruct-v0.2")
        text = await client.generate("Explain RAG in 3 sentences.")
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        timeout: float = 90,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout

    @property
    def backend_name(self) -> str:
        return "vllm"

    async def generate(self, prompt: str, *, model: str | None = None) -> str:
        """
        Send a prompt to vLLM and return the generated text.

        Uses the ``/v1/completions`` endpoint for raw prompt-based generation.
        """
        target_model = model or self._model
        payload = {
            "model": target_model,
            "prompt": prompt,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/v1/completions",
                    json=payload,
                )
                resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("text", "").strip()
            return ""
        except Exception as exc:
            logger.warning("vllm_client.generate_failed", error=str(exc))
            raise

    async def generate_chat(
        self, messages: list[dict[str, str]], *, model: str | None = None
    ) -> str:
        """
        Send a chat completion request to vLLM's ``/v1/chat/completions``.

        Useful for instruction-tuned models that expect chat-style formatting.
        """
        target_model = model or self._model
        payload = {
            "model": target_model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                )
                resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
            return ""
        except Exception as exc:
            logger.warning("vllm_client.chat_failed", error=str(exc))
            raise

    async def health_check(self) -> bool:
        """Return True if the vLLM server is reachable and healthy."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self._base_url}/health")
                return resp.status_code == 200
        except Exception:
            return False
