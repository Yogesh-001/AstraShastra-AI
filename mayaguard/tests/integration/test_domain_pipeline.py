import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.models import Document, RetrievalResult, QueryRequest, ControllerAction, RiskLevel
from core.retrieval.retriever import Retriever
from serving.pipeline import MayaGuardPipeline
from serving.registry import get_pipeline


@pytest.fixture(autouse=True)
def mock_retriever_dependencies():
    """
    Autouse fixture that completely mocks all external dependencies for Retriever.
    Prevents any network calls (Hugging Face hub downloads) or vector DB connections (Qdrant).
    """
    from serving.registry import _pipelines
    _pipelines.clear()  # Ensure clean pipeline initialization under mocks

    mock_transformer = MagicMock()
    mock_transformer.get_sentence_embedding_dimension.return_value = 384
    mock_transformer.get_embedding_dimension.return_value = 384
    mock_transformer.encode.return_value = MagicMock(tolist=lambda: [0.0] * 384)

    with patch("core.retrieval.retriever.SentenceTransformer", return_value=mock_transformer), \
         patch("core.retrieval.retriever.AsyncQdrantClient", return_value=MagicMock()), \
         patch("core.retrieval.retriever.Retriever._ensure_collection", new_callable=AsyncMock):
        yield


@pytest.mark.asyncio
async def test_medical_pipeline_integration():
    """
    Test the full medical pipeline end-to-end under simulated retrieval and LLM responses.
    Verifies adapter prompt formatting, stricter threshold override (0.45),
    unsupported claim floor penalty, high-risk controller rewrite action,
    and clinical disclaimers.
    """
    # 1. Retrieve the medical pipeline
    pipeline = await get_pipeline("medical")
    assert pipeline is not None
    assert pipeline._adapter.name == "medical"

    # 2. Setup mock data
    mock_query = "What is Metformin used for?"
    mock_docs = [
        Document(
            content="Metformin is the first-line pharmacotherapy for type 2 diabetes mellitus (T2DM). It lowers hepatic glucose production and improves insulin sensitivity.",
            source="PubMed:PMC7654321",
            score=0.92
        )
    ]
    
    # Mock Qdrant retrieval output
    pipeline._retriever.retrieve = AsyncMock(return_value=RetrievalResult(
        query=mock_query,
        documents=mock_docs,
        total_found=1
    ))

    # Mock Ollama HTTP responses
    class MockResponse:
        def __init__(self, json_data, status_code=200):
            self._json = json_data
            self.status_code = status_code

        def raise_for_status(self):
            pass

        def json(self):
            return self._json

    async def mock_post(url, **kwargs):
        json_body = kwargs.get("json", {})
        prompt = json_body.get("prompt", "")
        
        # Identify prompt context and return simulated LLM outputs
        if "Clinical Answer:" in prompt:
            # Main generation step
            return MockResponse({"response": "Metformin is the first-line treatment for type 2 diabetes. It can also cure cancer."})
        elif "numbered list" in prompt:
            # Claim extraction step
            return MockResponse({"response": "1. Metformin is the first-line treatment for type 2 diabetes.\n2. Metformin can also cure cancer."})
        elif "Determine whether the following claim is supported" in prompt:
            # Grounding check step
            if "first-line treatment" in prompt:
                return MockResponse({"response": "SUPPORTED\nThe document states Metformin is the first-line treatment for T2DM."})
            else:
                return MockResponse({"response": "UNSUPPORTED\nNo mention of curing cancer in the text."})
        elif "critical fact-checking assistant" in prompt:
            # Self-reflection step
            return MockResponse({"response": "CONFIDENCE: 0.50\nCRITIQUE: The cancer cure claim is completely unsupported by literature."})
        elif "cautious and accurate" in prompt:
            # Rewrite step (triggered because risk_score >= threshold override)
            return MockResponse({"response": "Metformin is widely recognized as the first-line treatment for type 2 diabetes (Source: PubMed:PMC7654321). Some claims about cancer are unsubstantiated."})
        
        return MockResponse({"response": "Default mock response."})

    # 3. Patch and execute
    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        req = QueryRequest(query=mock_query, adapter="medical")
        result = await pipeline.run(req)
        
        # 4. Verify orchestration
        assert result.adapter_used == "medical"
        
        # Check that the medical safety threshold override (0.45) was respected.
        # With 1 UNSUPPORTED claim out of 2 claims, the floor penalty ensures risk_score >= 0.50.
        # 0.50 risk score >= 0.45 threshold -> RiskLevel.HIGH (and ControllerAction.REWRITE).
        assert result.hallucination_report.overall_risk == RiskLevel.HIGH
        assert result.action_taken == ControllerAction.REWRITE
        
        # Verify custom clinical disclaimer is appended to the safe answer
        assert "*Medical Disclaimer:" in result.safe_answer
        assert "PubMed:PMC7654321" in result.safe_answer
        
        # Verify claim verity counts
        verdicts = result.hallucination_report.claim_verdicts
        assert len(verdicts) == 2
        assert verdicts[0].supported is True
        assert verdicts[0].claim.text == "Metformin is the first-line treatment for type 2 diabetes."
        assert verdicts[1].supported is False
        assert verdicts[1].claim.text == "Metformin can also cure cancer."


@pytest.mark.asyncio
async def test_legal_pipeline_integration():
    """
    Test the legal pipeline under simulated response scenario.
    Verifies that the legal safety policy disclaimer is correctly applied.
    """
    pipeline = await get_pipeline("legal")
    assert pipeline is not None
    assert pipeline._adapter.name == "legal"

    mock_query = "What is a confidential NDA?"
    mock_docs = [
        Document(
            content="A Non-Disclosure Agreement (NDA) is a legally binding contract that establishes a confidential relationship.",
            source="Restatement of Contracts §21",
            score=0.95
        )
    ]
    pipeline._retriever.retrieve = AsyncMock(return_value=RetrievalResult(
        query=mock_query,
        documents=mock_docs,
        total_found=1
    ))

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
        if "Legal Analysis:" in prompt:
            return MockResponse({"response": "An NDA is a confidential legal contract."})
        elif "numbered list" in prompt:
            return MockResponse({"response": "1. An NDA is a confidential legal contract."})
        elif "Determine whether the following claim is supported" in prompt:
            return MockResponse({"response": "SUPPORTED\nMatches the Restatement section."})
        elif "critical fact-checking assistant" in prompt:
            return MockResponse({"response": "CONFIDENCE: 0.95\nCRITIQUE: Fully accurate."})
        
        return MockResponse({"response": "Default."})

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        req = QueryRequest(query=mock_query, adapter="legal")
        result = await pipeline.run(req)
        
        # Verify legal adapter outcomes
        assert result.adapter_used == "legal"
        assert result.hallucination_report.overall_risk == RiskLevel.LOW
        assert result.action_taken == ControllerAction.PASS_THROUGH
        
        # Legal adapter requires always adding the disclaimer
        assert "*Legal Disclaimer:" in result.safe_answer
