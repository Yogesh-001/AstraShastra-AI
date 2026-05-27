from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# Core Enums

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ControllerAction(str, Enum):
    PASS_THROUGH = "pass_through"
    ADD_DISCLAIMER = "add_disclaimer"
    REWRITE = "rewrite"
    REFUSE = "refuse"


# Retrieval models

class Document(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    source: str
    score: float = Field(0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    documents: list[Document]
    total_found: int


# Claim structures

class Claim(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    span_start: int | None = None
    span_end: int | None = None


class ClaimVerdict(BaseModel):
    claim: Claim
    supported: bool
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_sources: list[str] = Field(default_factory=list)
    explanation: str = ""


# Hallucination assessment report

class EntropySpan(BaseModel):
    text: str
    entropy: float
    perplexity: float | None = None
    risk_level: RiskLevel


class HallucinationReport(BaseModel):
    response_id: str = Field(default_factory=lambda: str(uuid4()))
    overall_risk: RiskLevel
    risk_score: float = Field(ge=0.0, le=1.0)

    # Component scores
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    self_reflection_confidence: float = Field(ge=0.0, le=1.0)
    entropy_score: float = Field(ge=0.0, le=1.0)

    # Detailed findings
    claim_verdicts: list[ClaimVerdict] = Field(default_factory=list)
    high_entropy_spans: list[EntropySpan] = Field(default_factory=list)
    self_critique: str = ""

    # Source traceability
    retrieved_documents: list[Document] = Field(default_factory=list)


# Evaluation models

class EvaluationSample(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    reference_answer: str
    generated_answer: str
    is_hallucinated: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationMetrics(BaseModel):
    hallucination_rate: float
    faithfulness_mean: float
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    total_samples: int


# Endpoint request / response structures

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    context_documents: list[str] = Field(default_factory=list)
    adapter: str = "default"
    options: dict[str, Any] = Field(default_factory=dict)


class MayaGuardResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    raw_answer: str
    safe_answer: str
    action_taken: ControllerAction
    hallucination_report: HallucinationReport
    latency_ms: float
    adapter_used: str = "default"
