"""
Domain adapter registry.
Caches and lazily initializes pipeline instances mapped to specific adapter profiles.
"""

from __future__ import annotations

from typing import Optional

from adapters.default import DefaultAdapter
from adapters.medical import MedicalAdapter
from adapters.legal import LegalAdapter
from adapters.devops import DevopsAdapter
from core.config import get_settings
from core.logging import get_logger
from core.retrieval.retriever import Retriever
from serving.pipeline import MayaGuardPipeline

logger = get_logger(__name__)
_settings = get_settings()

_pipelines: dict[str, MayaGuardPipeline] = {}
_adapters = {
    "default": DefaultAdapter(),
    "medical": MedicalAdapter(),
    "legal": LegalAdapter(),
    "devops": DevopsAdapter(),
}


def register(name: str, adapter) -> None:  # type: ignore[no-untyped-def]
    """Register a domain adapter under *name*."""
    _adapters[name] = adapter
    # clear cached pipeline so it rebuilds with new adapter
    _pipelines.pop(name, None)


def _get_grounding_checker():
    """
    Return the appropriate grounding checker based on configuration.

    If a fine-tuned verifier adapter is configured and enabled, use the fast
    DeBERTa classifier. Otherwise, fall back to the default LLM-based checker.
    """
    if _settings.finetuned_verifier_enabled and _settings.finetuned_verifier_path:
        try:
            from core.verification.finetuned_verifier import FineTunedGroundingChecker
            checker = FineTunedGroundingChecker(_settings.finetuned_verifier_path)
            logger.info("registry.using_finetuned_verifier", path=_settings.finetuned_verifier_path)
            return checker
        except Exception as exc:
            logger.warning("registry.finetuned_verifier_failed_fallback", error=str(exc))

    from core.verification.verifier import GroundingChecker
    return GroundingChecker()


async def get_pipeline(name: str = "default") -> Optional[MayaGuardPipeline]:
    """Return (and lazily create) the pipeline for the named adapter."""
    if name not in _adapters:
        return None

    if name not in _pipelines:
        adapter = _adapters[name]
        retriever_cfg = adapter.get_retriever_config()
        retriever = await Retriever.create(**retriever_cfg)
        grounding_checker = _get_grounding_checker()
        _pipelines[name] = MayaGuardPipeline(
            retriever=retriever,
            adapter=adapter,
            grounding_checker=grounding_checker,
        )

    return _pipelines[name]
