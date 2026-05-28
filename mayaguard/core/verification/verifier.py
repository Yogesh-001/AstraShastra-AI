"""
Claim extraction and grounding verification.
Extracts factual claims from LLM answers and verifies them against retrieved source documents.
"""

from __future__ import annotations

import re

from core.config import get_settings
from core.inference import get_inference_client
from core.logging import get_logger
from core.models import Claim, ClaimVerdict, Document

logger = get_logger(__name__)
_settings = get_settings()

# Verification Prompt Templates

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
SUPPORTED   - the context clearly supports the claim
UNSUPPORTED - the context contradicts or does not mention the claim

Then on the next line write a one-sentence explanation.
"""


# Claim Extractor

class ClaimExtractor:
    """
    Uses an LLM to break a response into atomic factual claims.

    Fallback: simple sentence-splitting when the LLM is unavailable.
    """

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
        client = await get_inference_client()
        raw = await client.generate(prompt)
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


# Grounding Checker

class GroundingChecker:
    """
    Verifies each claim against a set of retrieved documents using an LLM.
    """

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
            client = await get_inference_client()
            raw: str = await client.generate(prompt)
            lines = raw.splitlines()
            verdict_word = lines[0].strip().upper() if lines else "UNSUPPORTED"
            explanation = lines[1].strip() if len(lines) > 1 else ""
            supported = verdict_word == "SUPPORTED"
            confidence = 0.85
            sources = [d.source for d in documents if d.score > 0.75]
        except Exception as exc:
            logger.warning("grounding_checker.offline_fallback", claim=claim.text[:60], error=str(exc))
            # Intelligent offline heuristic: check keyword overlap between claim and context
            words = set(re.findall(r"\w+", claim.text.lower()))
            stops = {"is", "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "of", "by", "that", "this", "it"}
            keywords = words - stops
            doc_text = context.lower()
            matches = sum(1 for w in keywords if w in doc_text)
            
            supported = False
            explanation = "No matching references found in the source documents (Offline Mode)."
            if keywords:
                overlap = matches / len(keywords)
                if overlap > 0.35:
                    supported = True
                    explanation = f"Factual claim grounded via offline word alignment ({overlap:.0%} match)."
            
            confidence = 0.75
            sources = [d.source for d in documents]

        return ClaimVerdict(
            claim=claim,
            supported=supported,
            confidence=confidence,
            supporting_sources=sources,
            explanation=explanation,
        )
