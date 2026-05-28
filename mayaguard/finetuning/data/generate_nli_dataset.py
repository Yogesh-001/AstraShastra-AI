"""
Generate NLI-style (claim, context) -> SUPPORTED/UNSUPPORTED training pairs
from MayaGuard seed datasets for fine-tuning the DeBERTa claim verifier.

Usage:
    python -m finetuning.data.generate_nli_dataset

Output:
    finetuning/data/train.jsonl
    finetuning/data/val.jsonl
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path


SEED_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "datasets"
OUTPUT_DIR = Path(__file__).resolve().parent
RANDOM_SEED = 42
VAL_RATIO = 0.15

# Synthetic hallucination claims injected as hard negatives
SYNTHETIC_HALLUCINATIONS = [
    "Metformin has been proven to cure all forms of cancer.",
    "Metformin eliminates the need for insulin in Type 1 diabetes patients.",
    "Metformin has no known side effects or contraindications.",
    "Metformin should be taken on an empty stomach for maximum efficacy.",
    "Metformin increases blood glucose levels in healthy individuals.",
    "The Kubernetes API server stores all data directly in etcd without encryption.",
    "Docker containers share the host kernel and therefore cannot be isolated.",
    "Terraform state files should be committed directly to Git repositories.",
    "The Indian Penal Code has been replaced and is no longer applicable.",
    "Bail is automatically granted for all bailable offences without exception.",
    "PCOS can only be treated with surgical intervention, not medication.",
    "Metformin is classified as a controlled substance in most countries.",
]


def _load_seed_docs(seed_file: Path) -> list[dict]:
    """Load documents from a seed JSON file."""
    with open(seed_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_sentences(text: str) -> list[str]:
    """Split a document into individual sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _generate_positive_pairs(docs: list[dict]) -> list[dict]:
    """
    Create SUPPORTED pairs by extracting sentences from each document
    and pairing them with their source document as context.
    """
    pairs = []
    for doc in docs:
        content = doc["content"]
        source = doc.get("source", "unknown")
        sentences = _extract_sentences(content)

        for sentence in sentences:
            pairs.append({
                "claim": sentence,
                "context": content,
                "label": "SUPPORTED",
                "source": source,
            })
    return pairs


def _generate_negative_pairs(docs: list[dict]) -> list[dict]:
    """
    Create UNSUPPORTED pairs by:
    1. Pairing claims from one document with unrelated documents as context
    2. Pairing synthetic hallucination claims with real documents
    """
    pairs = []

    # Cross-document negatives: claim from doc A, context from doc B
    for i, doc_a in enumerate(docs):
        sentences_a = _extract_sentences(doc_a["content"])
        if not sentences_a:
            continue

        # Pick a random unrelated document
        other_indices = [j for j in range(len(docs)) if j != i]
        if not other_indices:
            continue

        j = random.choice(other_indices)
        doc_b = docs[j]

        claim = random.choice(sentences_a)
        pairs.append({
            "claim": claim,
            "context": doc_b["content"],
            "label": "UNSUPPORTED",
            "source": f"cross:{doc_a.get('source', '')}->{doc_b.get('source', '')}",
        })

    # Synthetic hallucination negatives paired with real context
    for hallucination in SYNTHETIC_HALLUCINATIONS:
        doc = random.choice(docs)
        pairs.append({
            "claim": hallucination,
            "context": doc["content"],
            "label": "UNSUPPORTED",
            "source": "synthetic_hallucination",
        })

    return pairs


def generate_dataset() -> None:
    """Main dataset generation entry point."""
    random.seed(RANDOM_SEED)

    all_pairs: list[dict] = []

    # Process all available seed files
    for seed_file in sorted(SEED_DIR.glob("*_seed.json")):
        domain = seed_file.stem.replace("_seed", "")
        print(f"[LOAD] Processing {seed_file.name} ({domain} domain)...")

        docs = _load_seed_docs(seed_file)
        positives = _generate_positive_pairs(docs)
        negatives = _generate_negative_pairs(docs)

        print(f"  -> {len(positives)} positive pairs, {len(negatives)} negative pairs")
        all_pairs.extend(positives)
        all_pairs.extend(negatives)

    # Shuffle and split
    random.shuffle(all_pairs)
    split_idx = int(len(all_pairs) * (1 - VAL_RATIO))
    train_data = all_pairs[:split_idx]
    val_data = all_pairs[split_idx:]

    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    val_path = OUTPUT_DIR / "val.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Stats
    train_pos = sum(1 for d in train_data if d["label"] == "SUPPORTED")
    train_neg = len(train_data) - train_pos
    val_pos = sum(1 for d in val_data if d["label"] == "SUPPORTED")
    val_neg = len(val_data) - val_pos

    print(f"\n[DONE] Dataset generated:")
    print(f"  Train: {len(train_data)} samples ({train_pos} supported, {train_neg} unsupported)")
    print(f"  Val:   {len(val_data)} samples ({val_pos} supported, {val_neg} unsupported)")
    print(f"  Files: {train_path}, {val_path}")


if __name__ == "__main__":
    generate_dataset()
