"""
Fine-tuned claim verification using a DeBERTa classifier with LoRA adapter.

Replaces the LLM-based GroundingChecker with a fast, specialized NLI classifier
that was fine-tuned on MayaGuard seed data.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from core.config import get_settings
from core.logging import get_logger
from core.models import Claim, ClaimVerdict, Document

logger = get_logger(__name__)
_settings = get_settings()

# Lazy-loaded model (initialized on first use)
_model = None
_tokenizer = None


def _load_model(adapter_path: str):
    """Load the base DeBERTa model with LoRA adapter weights."""
    global _model, _tokenizer

    if _model is not None:
        return

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError(
            "Fine-tuned verifier requires: pip install peft transformers torch. "
            "Install with: pip install -e '.[finetuning]'"
        ) from exc

    base_model_name = "microsoft/deberta-v3-base"
    logger.info("finetuned_verifier.loading", base=base_model_name, adapter=adapter_path)

    _tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=2
    )

    if Path(adapter_path).exists():
        _model = PeftModel.from_pretrained(base_model, adapter_path)
        logger.info("finetuned_verifier.adapter_loaded", path=adapter_path)
    else:
        # If adapter path doesn't exist locally, try loading from HuggingFace Hub
        _model = PeftModel.from_pretrained(base_model, adapter_path)
        logger.info("finetuned_verifier.hub_adapter_loaded", hub_id=adapter_path)

    _model.eval()
    logger.info("finetuned_verifier.ready")


class FineTunedGroundingChecker:
    """
    Fast claim verification using a fine-tuned DeBERTa + LoRA adapter.

    This classifier runs entirely locally (no LLM API calls needed) and is
    significantly faster than the LLM-based GroundingChecker - typically
    ~50ms per claim vs ~2-5 seconds per claim with an LLM.

    The model expects NLI-style inputs: (claim, context_document) → SUPPORTED/UNSUPPORTED
    """

    LABEL_MAP = {0: "UNSUPPORTED", 1: "SUPPORTED"}

    def __init__(self, adapter_path: str | None = None) -> None:
        path = adapter_path or _settings.finetuned_verifier_path
        if not path:
            raise ValueError(
                "No adapter path configured. Set FINETUNED_VERIFIER_PATH in .env "
                "or pass adapter_path to __init__."
            )
        _load_model(path)

    async def verify(
        self, claims: list[Claim], documents: list[Document]
    ) -> list[ClaimVerdict]:
        """
        Verify all claims against the retrieved documents using batch inference.

        Each claim is paired with the concatenated document context and
        classified as SUPPORTED or UNSUPPORTED.
        """
        context = "\n\n".join(
            f"[{i+1}] ({d.source}): {d.content}"
            for i, d in enumerate(documents)
        )

        verdicts = []
        for claim in claims:
            verdict = await self._classify_claim(claim, context, documents)
            verdicts.append(verdict)
        return verdicts

    async def _classify_claim(
        self, claim: Claim, context: str, documents: list[Document]
    ) -> ClaimVerdict:
        """Run the fine-tuned classifier on a single (claim, context) pair."""
        import torch

        # Truncate context to model max length
        truncated_context = context[:2048]

        # Tokenize
        inputs = _tokenizer(
            claim.text,
            truncated_context,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Run inference in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        logits = await loop.run_in_executor(None, self._forward, inputs)

        probs = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item()

        supported = predicted_class == 1  # 1 = SUPPORTED
        label = self.LABEL_MAP[predicted_class]
        sources = [d.source for d in documents if d.score > 0.5]

        return ClaimVerdict(
            claim=claim,
            supported=supported,
            confidence=round(confidence, 4),
            supporting_sources=sources if supported else [],
            explanation=f"Fine-tuned verifier: {label} (confidence: {confidence:.2%})",
        )

    @staticmethod
    def _forward(inputs):
        """Synchronous forward pass (runs in executor)."""
        import torch

        with torch.no_grad():
            outputs = _model(**inputs)
        return outputs.logits
