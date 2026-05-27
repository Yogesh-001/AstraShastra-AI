"""
core/evaluation - Evaluation framework.

Runs a labelled dataset through the full MayaGuard pipeline and
computes aggregate metrics: hallucination rate, faithfulness, precision,
recall, F1.

This is intentionally built early so every other module can be validated
against ground truth from the start.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Callable, Awaitable

from core.logging import get_logger
from core.models import EvaluationMetrics, EvaluationSample, HallucinationReport, RiskLevel

logger = get_logger(__name__)


# ── Type alias ────────────────────────────────────────────────────────────────

PipelineFn = Callable[
    [EvaluationSample], Awaitable[tuple[HallucinationReport, str]]
]
"""
A coroutine that accepts one EvaluationSample and returns
(HallucinationReport, safe_answer).
"""


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs a batch of EvaluationSamples through a pipeline function
    and aggregates the results.
    """

    HIGH_RISK_LEVELS = {RiskLevel.HIGH, RiskLevel.CRITICAL}

    def __init__(self, pipeline: PipelineFn, concurrency: int = 4):
        self._pipeline = pipeline
        self._sem = asyncio.Semaphore(concurrency)

    async def run(self, samples: list[EvaluationSample]) -> EvaluationMetrics:
        results = await asyncio.gather(
            *[self._run_one(s) for s in samples], return_exceptions=False
        )
        return self._aggregate(samples, list(results))

    async def _run_one(
        self, sample: EvaluationSample
    ) -> tuple[HallucinationReport, str]:
        async with self._sem:
            return await self._pipeline(sample)

    def _aggregate(
        self,
        samples: list[EvaluationSample],
        results: list[tuple[HallucinationReport, str]],
    ) -> EvaluationMetrics:
        total = len(samples)
        high_risk_count = 0
        faithfulness_sum = 0.0

        tp = fp = tn = fn = 0

        for sample, (report, _) in zip(samples, results):
            is_high_risk = report.overall_risk in self.HIGH_RISK_LEVELS
            if is_high_risk:
                high_risk_count += 1

            faithfulness_sum += report.faithfulness_score

            if sample.is_hallucinated is not None:
                if sample.is_hallucinated and is_high_risk:
                    tp += 1
                elif not sample.is_hallucinated and not is_high_risk:
                    tn += 1
                elif not sample.is_hallucinated and is_high_risk:
                    fp += 1
                else:
                    fn += 1

        labelled = tp + fp + tn + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1: float | None = None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)

        metrics = EvaluationMetrics(
            hallucination_rate=round(high_risk_count / total, 4) if total else 0.0,
            faithfulness_mean=round(faithfulness_sum / total, 4) if total else 0.0,
            precision=round(precision, 4) if precision is not None else None,
            recall=round(recall, 4) if recall is not None else None,
            f1=round(f1, 4) if f1 is not None else None,
            total_samples=total,
        )

        logger.info(
            "evaluation.complete",
            total=total,
            labelled=labelled,
            hallucination_rate=metrics.hallucination_rate,
            faithfulness=metrics.faithfulness_mean,
            f1=metrics.f1,
        )
        return metrics


# ── Dataset utilities ─────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[EvaluationSample]:
    """
    Load an evaluation dataset from a JSONL file.

    Each line must be a JSON object with at least 'query', 'reference_answer',
    and 'generated_answer' keys.  'is_hallucinated' is optional.
    """
    samples: list[EvaluationSample] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            samples.append(EvaluationSample(**data))
    logger.info("evaluation.dataset_loaded", path=str(path), count=len(samples))
    return samples


def save_metrics(metrics: EvaluationMetrics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics.model_dump(), indent=2))
    logger.info("evaluation.metrics_saved", path=str(path))
