"""
serving/app.py — FastAPI application.

Endpoints:
  POST /api/v1/query        → run full MayaGuard pipeline
  GET  /api/v1/health       → liveness probe
  GET  /api/v1/metrics      → Prometheus text exposition (if enabled)
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from core.config import get_settings
from core.logging import get_logger, setup_logging
from core.models import QueryRequest, MayaGuardResponse
from serving.registry import get_pipeline

logger = get_logger(__name__)
_settings = get_settings()

# ── Prometheus metrics ────────────────────────────────────────────────────────

_REQUEST_TOTAL = Counter(
    "mayaguard_requests_total", "Total query requests", ["adapter", "action"]
)
_LATENCY = Histogram(
    "mayaguard_request_latency_ms", "Request latency in milliseconds", ["adapter"]
)
_HALLUCINATION_SCORE = Histogram(
    "mayaguard_hallucination_score", "Hallucination risk score distribution", ["adapter"]
)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    logger.info("mayaguard.startup", model=_settings.ollama_model)
    yield
    logger.info("mayaguard.shutdown")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="MayaGuard API",
        description="Modular hallucination-aware AI framework",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────

    @app.get("/api/v1/health")
    async def health() -> dict:
        return {"status": "ok", "model": _settings.ollama_model}

    @app.post("/api/v1/query", response_model=MayaGuardResponse)
    async def query(request: QueryRequest, http_request: Request) -> MayaGuardResponse:
        pipeline = await get_pipeline(request.adapter)
        if pipeline is None:
            raise HTTPException(status_code=400, detail=f"Unknown adapter: {request.adapter}")

        t0 = time.perf_counter()
        try:
            result = await pipeline.run(request)
        except Exception as exc:
            logger.error("api.query_failed", error=str(exc))
            raise HTTPException(status_code=500, detail="Pipeline error") from exc

        latency = (time.perf_counter() - t0) * 1000
        _REQUEST_TOTAL.labels(
            adapter=result.adapter_used, action=result.action_taken.value
        ).inc()
        _LATENCY.labels(adapter=result.adapter_used).observe(latency)
        _HALLUCINATION_SCORE.labels(adapter=result.adapter_used).observe(
            result.hallucination_report.risk_score
        )
        return result

    @app.get("/api/v1/metrics")
    async def metrics() -> Response:
        if not _settings.prometheus_enabled:
            raise HTTPException(status_code=404, detail="Metrics disabled")
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
