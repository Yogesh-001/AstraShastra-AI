# MayaGuard

*Maya* (माया) means illusion. A hallucination is exactly that.

MayaGuard is a modular framework that sits between an LLM and the user. When
the model answers something, MayaGuard checks whether the claims in that answer
are actually grounded in real sources — and if they aren't, it rewrites or
blocks the response before it reaches anyone.

---

## How it works

Every query goes through a four-step pipeline:

1. **Retrieve** — relevant documents are pulled from a vector database (Qdrant)
2. **Generate** — the LLM produces an answer using that context
3. **Verify** — claims in the answer are checked against the sources; the same
   model also critiques its own response in a second pass
4. **Control** — based on the risk score, the response is either passed through,
   given a disclaimer, rewritten, or refused

The output includes the final safe answer, a full hallucination report with
claim-level verdicts, and the sources that were (or weren't) found.

---

## Project structure

```
mayaguard/
├── core/               # domain-agnostic detection logic
│   ├── retrieval/      # Qdrant vector search
│   ├── verification/   # claim extraction, grounding, self-reflection
│   ├── hallucination/  # risk scoring and reporting
│   ├── scoring/        # response controller (pass / rewrite / refuse)
│   └── evaluation/     # batch evaluation framework
│
├── adapters/           # domain-specific extensions
│   ├── base.py         # abstract interface every adapter must implement
│   ├── default.py      # generic adapter
│   ├── medical/        # (Phase 2)
│   ├── legal/          # (future)
│   └── devops/         # (future)
│
├── serving/            # FastAPI app + pipeline orchestrator
├── frontend/           # Streamlit monitoring dashboard
├── benchmarks/         # evaluation datasets + results
└── tests/
```

The `core/` directory has zero knowledge of any domain. Medical, legal, or
any other domain concerns live entirely inside `adapters/` — swapping domains
is a one-line config change.

---

## Stack

- **Inference** — [Ollama](https://ollama.com) (Mistral 7B Q4, fits in 6GB VRAM)
- **Vector DB** — [Qdrant](https://qdrant.tech)
- **Embeddings** — sentence-transformers
- **API** — FastAPI
- **Dashboard** — Streamlit
- **Metrics** — Prometheus

---

## Getting started

**Requirements:** Docker, Python 3.10+, Nvidia GPU with 6GB+ VRAM

```bash
# 1. Clone and enter the project
git clone https://github.com/Yogesh-001/mayaguard.git
cd mayaguard

# 2. Set up environment
cp .env.example .env
pip install -e ".[dev,frontend]"

# 3. Start Qdrant and Ollama
docker compose up qdrant ollama -d

# 4. Pull the model
python main.py pull-model

# 5. Start the API
python main.py serve

# 6. Open the dashboard (separate terminal)
python main.py dashboard
```

Dashboard runs at `http://localhost:8501`
API runs at `http://localhost:8080`
API docs at `http://localhost:8080/docs`

---

## Current status

Phase 1 (core platform) is complete:

- [x] Retrieval-grounded claim verification
- [x] Self-reflection agent
- [x] Hallucination risk scoring
- [x] Adaptive response controller
- [x] Evaluation framework
- [x] Streamlit dashboard
- [x] FastAPI serving layer

Phase 2 (Medical Adapter) — in progress

---

## Running tests

```bash
pytest tests/unit -v
```
