"""
Interactive offline demonstration of the MayaGuard pipeline.

Runs the full orchestration pipeline (retrieval -> generation -> extraction ->
grounding -> reflection -> scoring -> control) under standard mock configs,
providing a step-by-step console visualization without needing Qdrant/Ollama active.

Usage:
    python demo_run.py
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from core.models import Document, RetrievalResult, QueryRequest, RiskLevel, ControllerAction
from serving.registry import get_pipeline, _pipelines


def setup_mock_harness(pipeline):
    """Mocks out the database retrieval and remote LLM HTTP calls."""
    # 1. Mock semantic retrieval matching
    mock_docs = [
        Document(
            content="Metformin is the first-line pharmacotherapy for type 2 diabetes mellitus (T2DM). It lowers hepatic glucose production and improves insulin sensitivity.",
            source="PubMed:PMC7654321",
            score=0.92
        )
    ]
    pipeline._retriever.retrieve = AsyncMock(return_value=RetrievalResult(
        query="What is Metformin used for?",
        documents=mock_docs,
        total_found=1
    ))

    # 2. Mock Ollama responses
    class MockResponse:
        def __init__(self, json_data):
            self._json = json_data
        def raise_for_status(self):
            pass
        def json(self):
            return self._json

    async def mock_post(url, **kwargs):
        json_body = kwargs.get("json", {})
        prompt = json_body.get("prompt", "")
        
        if "Answer:" in prompt and "Question:" in prompt:
            # Stage 3: Raw Generation
            return MockResponse({"response": "Metformin is the first-line treatment for type 2 diabetes. It has been investigated for PCOS and cancer prevention. Metformin is also a cure for cancer."})
        elif "numbered list" in prompt:
            # Stage 4a: Claim Extraction
            return MockResponse({"response": "1. Metformin is the first-line treatment for type 2 diabetes.\n2. It has been investigated for PCOS and cancer prevention.\n3. Metformin is also a cure for cancer."})
        elif "Determine whether the following claim is supported" in prompt:
            # Stage 4b: Grounding check
            if "first-line treatment" in prompt:
                return MockResponse({"response": "SUPPORTED\nThe literature clearly states Metformin is the first-line pharmacotherapy for T2DM."})
            elif "investigated for PCOS" in prompt:
                return MockResponse({"response": "SUPPORTED\nReference sources mention investigations into PCOS and oncology prevention."})
            else:
                return MockResponse({"response": "UNSUPPORTED\nNo evidence support curing cancer in any retrieved sources."})
        elif "critical fact-checking assistant" in prompt:
            # Stage 4c: Self-reflection
            return MockResponse({"response": "CONFIDENCE: 0.20\nCRITIQUE: The response claims Metformin is a cure for cancer, which is highly misleading and unsupported."})
        elif "cautious and accurate" in prompt:
            # Stage 6: Real-time rewrite (triggered by High Risk score)
            return MockResponse({"response": "Metformin is widely recognized as the first-line treatment for type 2 diabetes (Source: PubMed:PMC7654321). While being investigated for secondary oncology prevention, claims that Metformin cures cancer are not supported by clinical evidence."})
        
        return MockResponse({"response": "Default mock response."})

    return patch("httpx.AsyncClient.post", side_effect=mock_post)


async def main():
    print("=" * 70)
    print("[DEMO] MayaGuard - Interactive Pipeline Demonstration")
    print("=" * 70)
    print("[INFO] Mocking external databases and models (Offline Mode)...")

    # Clear pipelines cache
    _pipelines.clear()
    
    # Pre-mock creation phase dependencies to avoid import failures
    mock_transformer = MagicMock()
    mock_transformer.get_sentence_embedding_dimension.return_value = 384
    mock_transformer.get_embedding_dimension.return_value = 384
    mock_transformer.encode.return_value = MagicMock(tolist=lambda: [0.0]*384)

    with patch("core.retrieval.retriever.SentenceTransformer", return_value=mock_transformer), \
         patch("core.retrieval.retriever.AsyncQdrantClient", return_value=MagicMock()), \
         patch("core.retrieval.retriever.Retriever._ensure_collection", new_callable=AsyncMock):
        
        pipeline = await get_pipeline("medical")

    # Set up our test harness
    mock_harness = setup_mock_harness(pipeline)
    
    query = "What is Metformin used for?"
    print(f"\n[QUERY] Input: \"{query}\"")
    print(f"[CONFIG] Active Adapter: 'medical' (Strict threshold: 0.45, Mandatory disclaimers)")
    
    with mock_harness:
        print("\n--- Running Orchestrator Pipeline ---")
        print("[STAGE 1] Preprocessing query...")
        
        # We manually step into pipeline stages for visualization:
        processed_query = pipeline._adapter.preprocess_query(query)
        
        print("[STAGE 2] Fetching semantic documents from Qdrant...")
        retrieval = await pipeline._retriever.retrieve(processed_query, top_k=5)
        docs = pipeline._adapter.postprocess_documents(retrieval.documents)
        print(f"  -> Retrieved {len(docs)} document from collection 'mayaguard_medical':")
        for i, d in enumerate(docs, 1):
            print(f"     [{i}] Source: {d.source} | Score: {d.score}")
            print(f"         Content: \"{d.content[:110]}...\"")

        print("\n[STAGE 3] Running inference model (Ollama) to generate raw answer...")
        context = "\n\n".join(f"[{i+1}] {d.content}" for i, d in enumerate(docs))
        raw_answer = await pipeline._generate(
            system=pipeline._adapter.get_prompt_template().system,
            query=processed_query,
            context=context
        )
        print(f"  -> Raw LLM Answer generated:")
        print(f"     \"{raw_answer}\"")

        print("\n[STAGE 4] Executing Factual Verifications...")
        print("  -> Claim Extraction: Parsing answer into atomic assertions...")
        claims = await pipeline._extractor.extract(raw_answer)
        for i, c in enumerate(claims, 1):
            print(f"     Factual Claim {i}: \"{c.text}\"")

        print("  -> Grounding Checker: Fact-checking each claim against Qdrant sources...")
        verdicts = await pipeline._grounding.verify(claims, docs)
        for i, v in enumerate(verdicts, 1):
            icon = "[SUPPORTED]" if v.supported else "[UNSUPPORTED]"
            print(f"     Claim {i}: {icon} - {v.explanation}")

        print("  -> Self-Reflection Agent: Orchestrating secondary critique pass...")
        reflect_conf, critique = await pipeline._reflect.reflect(processed_query, raw_answer)
        print(f"     Self-Confidence Score: {reflect_conf:.2f}/1.0")
        print(f"     Self-Critique: \"{critique}\"")

        print("\n[STAGE 5] Calculating Hallucination Risk Score...")
        report = pipeline._detector.build_report(
            response_id="demo-req-1",
            claim_verdicts=verdicts,
            retrieved_documents=docs,
            self_reflection_confidence=reflect_conf,
            self_critique=critique,
            risk_threshold=pipeline._adapter.get_safety_policy().risk_threshold_override
        )
        print(f"  -> Aggregated Risk Score: {report.risk_score:.4f}/1.0")
        print(f"  -> Safety Action Level:  {report.overall_risk.value.upper()} (Medical Threshold Limit: 0.45)")

        print("\n[STAGE 6] Applying Response Controller Policy...")
        policy = pipeline._adapter.get_safety_policy()
        citation_block = pipeline._adapter.format_citations(docs)
        extra = (policy.disclaimer_text or "") + citation_block
        safe_answer, action = await pipeline._controller.control(
            raw_answer, report, extra_disclaimer=extra
        )
        print(f"  -> Applied Action: **{action.value.upper()}**")

        print("\n" + "=" * 70)
        print("[DEMO] Final Safe Answer Delivered to Client:")
        print("=" * 70)
        print(safe_answer)
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
