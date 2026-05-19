"""
core/verification — Claim extraction and grounding verification.

Two responsibilities:
  1. ClaimExtractor   — parse an LLM response into atomic factual claims
  2. GroundingChecker — verify each claim against retrieved documents
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from core.config import get_settings
from core.logging import get_logger
from core.models import Claim, ClaimVerdict, Document

logger = get_logger(__name__)
_settings = get_settings()

# ── Prompt templates ──────────────────────────────────────────────────────────

_EXTRACT_PROMPT = """\
You are a claim extraction assistant. Given the following text, extract all \
distinct factual claims as a numbered list. Each claim must be a single, \
self-contained sentence.

Text:
{text}

Output ONLY the numbered list. Example:
1. Claim one.
2. Claim two.
"""

_VERIFY_PROMPT = """\
You are a fact-checking assistant. Determine whether the following claim is \
supported by the provided context documents.

Claim:
{claim}

Context:
{context}

Respond with EXACTLY one of:
SUPPORTED   — the context clearly supports the claim
UNSUPPORTED — the context contradicts or does not mention the claim

Then on the next line write a one-sentence explanation.
"""


# ── Claim Extractor ───────────────────────────────────────────────────────────

class ClaimExtractor:
    """
    Uses an LLM to break a response into atomic factual claims.

    Fallback: simple sentence-splitting when the LLM is unavailable.
    """

    def __init__(self, ollama_url: str | None = None, model: str | None = None):
        self._url = (ollama_url or _settings.ollama_base_url) + "/api/generate"
        self._model = model or _settings.ollama_model

    async def extract(self, text: str) -> list[Claim]:
        try:
            claims = await self._llm_extract(text)
        except Exception as exc:
            logger.warning("claim_extractor.fallback", reason=str(exc))
            claims = self._sentence_split(text)
        logger.debug("claim_extractor.done", count=len(claims))
        return claims

    async def _llm_extract(self, text: str) -> list[Claim]:
        prompt = _EXTRACT_PROMPT.format(text=text)
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self._url,
                json={"model": self._model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
        raw = resp.json().get("response", "")
        return self._parse_numbered_list(raw)

    @staticmethod
    def _parse_numbered_list(text: str) -> list[Claim]:
        claims = []
        for line in text.splitlines():
            line = line.strip()
            match = re.match(r"^\d+\.\s+(.+)", line)
            if match:
                claims.append(Claim(text=match.group(1).strip()))
        return claims

    @staticmethod
    def _sentence_split(text: str) -> list[Claim]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [Claim(text=s) for s in sentences if len(s) > 10]


# ── Grounding Checker ─────────────────────────────────────────────────────────

class GroundingChecker:
    """
    Verifies each claim against a set of retrieved documents using an LLM.
    """

    def __init__(self, ollama_url: str | None = None, model: str | None = None):
        self._url = (ollama_url or _settings.ollama_base_url) + "/api/generate"
        self._model = model or _settings.ollama_model

    async def verify(
        self, claims: list[Claim], documents: list[Document]
    ) -> list[ClaimVerdict]:
        context = "\n\n".join(
            f"[{i+1}] (source: {d.source})\n{d.content}"
            for i, d in enumerate(documents)
        )
        verdicts = []
        for claim in claims:
            verdict = await self._check_one(claim, context, documents)
            verdicts.append(verdict)
        return verdicts

    async def _check_one(
        self, claim: Claim, context: str, documents: list[Document]
    ) -> ClaimVerdict:
        prompt = _VERIFY_PROMPT.format(claim=claim.text, context=context[:4000])
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    self._url,
                    json={"model": self._model, "prompt": prompt, "stream": False},
                )
                resp.raise_for_status()
            raw: str = resp.json().get("response", "").strip()
            lines = raw.splitlines()
            verdict_word = lines[0].strip().upper() if lines else "UNSUPPORTED"
            explanation = lines[1].strip() if len(lines) > 1 else ""
            supported = "SUPPORTED" in verdict_word
            confidence = 0.85 if supported else 0.15
            sources = [d.source for d in documents if d.score > 0.75]
        except Exception as exc:
            logger.warning("grounding_checker.error", claim=claim.text[:60], error=str(exc))
            supported, confidence, explanation, sources = False, 0.5, "Verification failed.", []

        return ClaimVerdict(
            claim=claim,
            supported=supported,
            confidence=confidence,
            supporting_sources=sources,
            explanation=explanation,
        )
