"""
Generate instruction-tuning dataset from MayaGuard seed data for QLoRA
fine-tuning of a hallucination-aware LLM.

Usage:
    python -m finetuning.data.generate_instruct_dataset

Output:
    finetuning/data/instruct_train.jsonl
    finetuning/data/instruct_val.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path


SEED_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "datasets"
OUTPUT_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 42
VAL_RATIO = 0.15

# Domain-specific query templates for generating instruction examples
MEDICAL_QUERIES = [
    "What is the mechanism of action of Metformin?",
    "What are the side effects of Metformin?",
    "Is Metformin safe for patients with renal impairment?",
    "What clinical trials support Metformin for diabetes?",
    "Can Metformin be used during pregnancy?",
    "What drug interactions does Metformin have?",
    "What are the contraindications for Metformin?",
    "How does Metformin affect insulin sensitivity?",
    "What is the bioavailability of Metformin?",
    "Can Metformin cure cancer?",
    "What is lactic acidosis and how is it related to Metformin?",
    "Is Metformin used for PCOS treatment?",
    "What is the Diabetes Prevention Program study?",
    "Should Metformin be stopped before contrast imaging?",
]

LEGAL_QUERIES = [
    "What constitutes a bailable offence?",
    "How does anticipatory bail work?",
    "What are the fundamental rights in the Constitution?",
    "How does the right to information work?",
    "What is the legal process for filing a complaint?",
]

DEVOPS_QUERIES = [
    "How does Kubernetes manage container orchestration?",
    "What is the purpose of Terraform state files?",
    "How should Docker images be built for production?",
    "What are best practices for CI/CD pipelines?",
    "How does Prometheus monitoring work?",
]


def _load_seed_docs(seed_file: Path) -> list[dict]:
    """Load documents from a seed JSON file."""
    with open(seed_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _format_context(docs: list[dict], max_docs: int = 5) -> str:
    """Format seed documents as numbered context for the instruction prompt."""
    selected = docs[:max_docs]
    lines = []
    for i, doc in enumerate(selected, 1):
        source = doc.get("source", "Unknown")
        lines.append(f"[{i}] (Source: {source})\n{doc['content']}")
    return "\n\n".join(lines)


def _generate_grounded_answer(query: str, relevant_docs: list[dict]) -> str:
    """
    Generate a grounded answer with citations based on the relevant documents.
    This creates training examples that teach the model to cite sources.
    """
    parts = []
    for i, doc in enumerate(relevant_docs[:3], 1):
        content = doc["content"]
        source = doc.get("source", "Unknown")
        # Use first 2 sentences of the document
        sentences = content.split(". ")[:2]
        summary = ". ".join(sentences)
        if not summary.endswith("."):
            summary += "."
        parts.append(f"{summary} (Source: {source})")

    answer = " ".join(parts)

    # Add a citation-aware footer
    answer += "\n\nNote: This answer is based on the provided reference documents. "
    answer += "Please consult the original sources for complete clinical guidance."
    return answer


def _generate_refusal_answer(query: str) -> str:
    """Generate a safe refusal answer for hallucination-prone queries."""
    return (
        f"I cannot provide a verified answer to the question: '{query}'. "
        "The available reference documents do not contain sufficient evidence to support "
        "a reliable response. Please consult a qualified domain expert or refer to "
        "authoritative primary sources for accurate information."
    )


def _generate_instruction_pairs(
    docs: list[dict], queries: list[str], domain: str
) -> list[dict]:
    """Generate (instruction, input, output) training examples."""
    examples = []

    for query in queries:
        # Find relevant documents by keyword matching
        query_words = set(query.lower().split())
        scored_docs = []
        for doc in docs:
            content_words = set(doc["content"].lower().split())
            overlap = len(query_words & content_words)
            scored_docs.append((overlap, doc))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_docs = [d for _, d in scored_docs[:5]]

        context = _format_context(top_docs)

        # Decide if this query should get a grounded answer or a refusal
        hallucination_keywords = {"cure", "cures", "eliminate", "guaranteed", "always", "never"}
        is_hallucination_query = bool(query_words & hallucination_keywords)

        if is_hallucination_query:
            output = _generate_refusal_answer(query)
        else:
            output = _generate_grounded_answer(query, top_docs)

        examples.append({
            "instruction": query,
            "input": context,
            "output": output,
            "domain": domain,
        })

    # Generate additional examples by combining random queries with random docs
    for _ in range(len(queries) * 2):
        num_docs = random.randint(2, min(5, len(docs)))
        selected_docs = random.sample(docs, num_docs)
        context = _format_context(selected_docs)

        # Create a query based on the first document's topic
        first_doc = selected_docs[0]
        topic = first_doc.get("metadata", {}).get("topic", "general")

        query_templates = [
            f"Based on the provided documents, what information is available about {topic.lower()}?",
            f"Summarize the key findings related to {topic.lower()} from the reference sources.",
            f"What do the available sources say about {topic.lower()}?",
        ]
        query = random.choice(query_templates)
        output = _generate_grounded_answer(query, selected_docs)

        examples.append({
            "instruction": query,
            "input": context,
            "output": output,
            "domain": domain,
        })

    return examples


def _format_as_chat(example: dict) -> dict:
    """
    Convert an instruction example to chat format compatible with
    Mistral/Llama instruction templates.
    """
    system_msg = (
        "You are a trustworthy AI assistant. Always ground your answers in the "
        "provided reference documents. Cite sources explicitly. If the documents "
        "do not contain enough evidence, refuse to answer and recommend consulting "
        "an expert."
    )

    user_msg = f"{example['instruction']}\n\nContext:\n{example['input']}"

    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": example["output"]},
        ],
        "domain": example["domain"],
    }


def generate_dataset() -> None:
    """Main dataset generation entry point."""
    random.seed(RANDOM_SEED)

    all_examples: list[dict] = []

    # Map domain names to query templates
    domain_queries = {
        "medical": MEDICAL_QUERIES,
        "legal": LEGAL_QUERIES,
        "devops": DEVOPS_QUERIES,
    }

    for seed_file in sorted(SEED_DIR.glob("*_seed.json")):
        domain = seed_file.stem.replace("_seed", "")
        queries = domain_queries.get(domain, MEDICAL_QUERIES[:5])

        print(f"[LOAD] Processing {seed_file.name} ({domain} domain)...")
        docs = _load_seed_docs(seed_file)
        examples = _generate_instruction_pairs(docs, queries, domain)

        # Convert to chat format
        chat_examples = [_format_as_chat(ex) for ex in examples]
        print(f"  -> {len(chat_examples)} instruction examples")
        all_examples.extend(chat_examples)

    # Shuffle and split
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * (1 - VAL_RATIO))
    train_data = all_examples[:split_idx]
    val_data = all_examples[split_idx:]

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "instruct_train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    val_path = OUTPUT_DIR / "instruct_val.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n[DONE] Instruction dataset generated:")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Val:   {len(val_data)} examples")
    print(f"  Files: {train_path}, {val_path}")


if __name__ == "__main__":
    generate_dataset()
