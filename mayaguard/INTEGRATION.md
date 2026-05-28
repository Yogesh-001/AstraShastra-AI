# 🔌 Integrating MayaGuard with Custom RAG Applications

This guide provides step-by-step instructions on how to integrate MayaGuard into your own Retrieval-Augmented Generation (RAG) applications, create custom domain adapters (such as **AOS - Aerospace/Orbital Systems**), and register them with ease.

---

## 🚀 1. Integration Strategies

You can integrate MayaGuard into your application using two main strategies: **Service Mode (API)** or **Library Mode (Direct Import)**.

### Option A: Service Mode (FastAPI Endpoint)
*Best for non-Python apps (Node.js, Go) or to isolate GPU/VRAM resources.*

1. **Start the API Service:**
   ```bash
   python main.py serve
   ```
2. **Call the `/api/v1/query` endpoint:**
   Send the user query to MayaGuard. MayaGuard will handle retrieval, generation, fact-checking, and safety routing.

   ```python
   import httpx

   async def check_safety(query: str, domain: str = "default"):
       payload = {"query": query, "adapter": domain}
       async with httpx.AsyncClient() as client:
           resp = await client.post("http://127.0.0.1:8080/api/v1/query", json=payload)
           data = resp.json()
       return data["safe_answer"], data["action_taken"]
   ```

### Option B: Library Mode (Direct Python Import)
*Best for native Python RAG applications.*

You can import MayaGuard's modular components to check your generated answers directly against your own retrieved context documents:

```python
import asyncio
from core.models import Claim, Document
from core.verification.verifier import ClaimExtractor, GroundingChecker

async def run_safety_guardrail(generated_llm_answer: str, retrieved_sources: list[dict]):
    # 1. Format your search results into MayaGuard Document models
    docs = [
        Document(content=src["text"], source=src["source_url"], score=src.get("relevance", 0.8))
        for src in retrieved_sources
    ]

    # 2. Extract atomic claims from the generated answer
    extractor = ClaimExtractor()
    claims = await extractor.extract(generated_llm_answer)

    # 3. Verify the claims against the retrieved sources
    checker = GroundingChecker()
    verdicts = await checker.verify(claims, docs)

    # 4. Filter or flag responses with unsupported claims
    is_safe = all(v.supported for v in verdicts)
    return is_safe, verdicts
```

---

## 🧬 2. Creating Custom Domain Adapters (e.g., AOS)

If your RAG application is about a topic other than the pre-built domains (Medical, Legal, DevOps) — let's say it is about **AOS (Aerospace/Advanced Orbit Systems)** — you have two choices:
1. **Use the Generic `"default"` Adapter:** It works out-of-the-box and requires zero custom code.
2. **Create a Custom `"aos"` Adapter:** This is highly recommended to enforce specialized safety rules, system prompts, retrieval collections, and warning footers.

### The Adapter Template
All domain adapters inherit from `DomainAdapter` defined in [base.py](file:///c:/Users/mural/Documents/AstraShastra-AI/mayaguard/adapters/base.py).

Here is a complete template for creating an **AOS Adapter**:

```python
# adapters/aos.py
from adapters.base import DomainAdapter, EvaluationSuite, PromptTemplate, SafetyPolicy

class AosAdapter(DomainAdapter):
    @property
    def name(self) -> str:
        return "aos"

    def get_retriever_config(self) -> dict:
        return {
            "collection": "mayaguard_aos",
            "embed_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        }

    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            system=(
                "You are an Aerospace and Orbital Systems (AOS) safety co-pilot. "
                "Always base your equations, metrics, and flight guidelines strictly "
                "on the reference documents. If the context does not contain sufficient "
                "telemetry or technical details, explicitly state the limitation."
            ),
            user_template=(
                "AOS Reference Context:\n{context}\n\n"
                "Aerospace Query:\n{query}\n\n"
                "Guidance Answer:"
            ),
        )

    def get_safety_policy(self) -> SafetyPolicy:
        return SafetyPolicy(
            risk_threshold_override=0.50,  # Stricter threshold for critical system guidance
            always_add_disclaimer=True,
            disclaimer_text="*AOS Disclaimer: Orbital telemetry guidance is simulated. Verify safety limits.*",
        )

    def get_evaluation_suite(self) -> EvaluationSuite:
        return EvaluationSuite(
            dataset_path="aos/sample_eval.jsonl",
            metric_names=["hallucination_rate", "faithfulness_mean", "f1"],
        )
```

---

## 🔌 3. Registering Your New Adapter Dynamically

You don't need to rebuild or hardcode changes into the MayaGuard core to register your new adapter. You can dynamically register it at startup:

```python
from serving.registry import register
from adapters.aos import AosAdapter

# Register the custom AOS adapter with the registry
register("aos", AosAdapter())
```

Once registered:
* The REST endpoint will instantly accept `{"adapter": "aos"}` in the `QueryRequest`.
* A dedicated Qdrant collection (`mayaguard_aos`) will be indexed during seeding or retrieval.
* Custom safety policies and clinical/aerospace warnings will automatically apply to safe outputs.
