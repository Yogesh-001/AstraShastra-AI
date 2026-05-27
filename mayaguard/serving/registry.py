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
from core.retrieval.retriever import Retriever
from serving.pipeline import MayaGuardPipeline

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


async def get_pipeline(name: str = "default") -> Optional[MayaGuardPipeline]:
    """Return (and lazily create) the pipeline for the named adapter."""
    if name not in _adapters:
        return None

    if name not in _pipelines:
        adapter = _adapters[name]
        retriever_cfg = adapter.get_retriever_config()
        retriever = await Retriever.create(**retriever_cfg)
        _pipelines[name] = MayaGuardPipeline(retriever=retriever, adapter=adapter)

    return _pipelines[name]
