"""
Hallucination detection engine.
Aggregates signals from retrieval grounding, self-reflection, and token entropy.
"""

from __future__ import annotations

import math
from typing import Sequence

from core.config import get_settings
from core.logging import get_logger
from core.models import (
    ClaimVerdict,
    Document,
    EntropySpan,
    HallucinationReport,
    RiskLevel,
)

logger = get_logger(__name__)
_settings = get_settings()


# Risk classifier

def score_to_risk(score: float, threshold: float | None = None) -> RiskLevel:
    """Classify a risk score into a RiskLevel using configurable thresholds."""
    base = threshold if threshold is not None else _settings.hallucination_risk_threshold
    if score >= base + 0.15:
        return RiskLevel.CRITICAL
    if score >= base:
        return RiskLevel.HIGH
    if score >= base - 0.25:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


# Faithfulness scorer

def compute_faithfulness(verdicts: list[ClaimVerdict]) -> float:
    """
    Fraction of claims that are supported, weighted by confidence.
    Returns 1.0 (fully faithful) when there are no claims to check.
    """
    if not verdicts:
        return 1.0
    total_weight = sum(v.confidence for v in verdicts)
    if total_weight == 0:
        return 0.5
    supported_weight = sum(v.confidence for v in verdicts if v.supported)
    return supported_weight / total_weight


# Entropy scorer

def token_entropy(logprobs: Sequence[float]) -> float:
    """
    Shannon entropy over a probability distribution implied by log-probabilities.
    A higher value means higher uncertainty.
    """
    if not logprobs:
        return 0.0
    probs = [math.exp(lp) for lp in logprobs]
    total = sum(probs)
    normalised = [p / total for p in probs]
    return -sum(p * math.log2(p + 1e-12) for p in normalised)


def compute_entropy_spans(
    tokens: list[str], logprobs: list[float], window: int = 5
) -> list[EntropySpan]:
    """
    Slide a window over (token, logprob) pairs and emit spans whose
    entropy exceeds the medium threshold.
    """
    spans: list[EntropySpan] = []
    for i in range(0, len(tokens) - window + 1, window):
        chunk_tokens = tokens[i : i + window]
        chunk_lp = logprobs[i : i + window]
        entropy = token_entropy(chunk_lp)
        if entropy < _settings.entropy_medium_threshold:
            continue
        risk = (
            RiskLevel.HIGH
            if entropy >= _settings.entropy_high_threshold
            else RiskLevel.MEDIUM
        )
        spans.append(
            EntropySpan(
                text=" ".join(chunk_tokens),
                entropy=round(entropy, 3),
                perplexity=round(math.exp(-sum(chunk_lp) / len(chunk_lp)), 2),
                risk_level=risk,
            )
        )
    return spans


# Main engine

class HallucinationDetector:
    """
    Combines retrieval grounding, self-reflection, and entropy into a
    single HallucinationReport.

    Weights (configurable):
        faithfulness:      40%
        self-reflection:   40%   (inverted - low confidence = high risk)
        entropy:           20%
    """

    W_FAITH = 0.40
    W_REFLECT = 0.40
    W_ENTROPY = 0.20

    def build_report(
        self,
        *,
        response_id: str,
        claim_verdicts: list[ClaimVerdict],
        retrieved_documents: list[Document],
        self_reflection_confidence: float,
        self_critique: str = "",
        entropy_spans: list[EntropySpan] | None = None,
        risk_threshold: float | None = None,
    ) -> HallucinationReport:
        spans = entropy_spans or []

        # Component scores (all expressed as "risk contribution", 0=good 1=bad)
        faithfulness = compute_faithfulness(claim_verdicts)
        faith_risk = 1.0 - faithfulness

        reflect_risk = 1.0 - self_reflection_confidence

        if spans:
            max_entropy = max(s.entropy for s in spans)
            norm_entropy = min(1.0, max_entropy / 6.0)  # 6 bits ≈ max realistic entropy
        else:
            norm_entropy = 0.0

        risk_score = (
            self.W_FAITH * faith_risk
            + self.W_REFLECT * reflect_risk
            + self.W_ENTROPY * norm_entropy
        )

        # Floor penalty: any unsupported claim guarantees a minimum risk
        if claim_verdicts:
            unsupported = sum(1 for v in claim_verdicts if not v.supported)
            if unsupported > 0:
                floor = unsupported / len(claim_verdicts)
                risk_score = max(risk_score, floor)

        risk_score = round(min(1.0, max(0.0, risk_score)), 4)

        report = HallucinationReport(
            response_id=response_id,
            overall_risk=score_to_risk(risk_score, threshold=risk_threshold),
            risk_score=risk_score,
            faithfulness_score=round(faithfulness, 4),
            self_reflection_confidence=round(self_reflection_confidence, 4),
            entropy_score=round(norm_entropy, 4),
            claim_verdicts=claim_verdicts,
            high_entropy_spans=spans,
            self_critique=self_critique,
            retrieved_documents=retrieved_documents,
        )

        logger.info(
            "hallucination.report",
            response_id=response_id,
            risk=report.overall_risk,
            score=risk_score,
            faithfulness=faithfulness,
            reflection_conf=self_reflection_confidence,
        )
        return report
