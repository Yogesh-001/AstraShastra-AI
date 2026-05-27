"""
Abstract domain adapter interface.
Defines the contract that every domain adapter must implement to interact with the core pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from core.models import Document


# Prompt template definition

@dataclass
class PromptTemplate:
    """Wraps a system prompt and a query template."""
    system: str
    user_template: str       # must contain {query} and optionally {context}
    safety_footer: str = ""  # appended to every safe answer

    def format(self, query: str, context: str = "") -> str:
        return self.user_template.format(query=query, context=context)


# Safety policy overrides

@dataclass
class SafetyPolicy:
    """
    Thresholds and rules that override global defaults for this domain.
    """
    risk_threshold_override: float | None = None   # overrides settings value
    always_add_disclaimer: bool = False
    refuse_on_critical: bool = True
    disclaimer_text: str = ""
    forbidden_topics: list[str] = field(default_factory=list)


# Evaluation suite properties

@dataclass
class EvaluationSuite:
    """
    Points to the dataset and metrics relevant to this domain.
    """
    dataset_path: str          # relative to benchmarks/datasets/
    metric_names: list[str]    # e.g. ["hallucination_rate", "faithfulness_mean"]


# Abstract base class

class DomainAdapter(ABC):
    """
    Contract that every domain adapter must fulfil.

    The core pipeline calls these methods without knowing which adapter
    is active - dependency inversion at the architecture level.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'medical', 'legal'."""

    @abstractmethod
    def get_retriever_config(self) -> dict:
        """
        Return kwargs passed to Retriever.create().

        Minimum required keys:
          collection (str)       - Qdrant collection name
          embed_model_name (str) - HuggingFace embedding model
        """

    @abstractmethod
    def get_prompt_template(self) -> PromptTemplate:
        """Return the generation prompt template for this domain."""

    @abstractmethod
    def get_safety_policy(self) -> SafetyPolicy:
        """Return domain-specific safety overrides."""

    @abstractmethod
    def get_evaluation_suite(self) -> EvaluationSuite:
        """Return dataset path and metric names for evaluation."""

    # Optional lifecycle hooks

    def preprocess_query(self, query: str) -> str:
        """Hook: transform or sanitise the query before retrieval."""
        return query

    def postprocess_documents(self, docs: list[Document]) -> list[Document]:
        """Hook: re-rank or filter retrieved documents."""
        return docs

    def format_citations(self, docs: list[Document]) -> str:
        """Hook: produce a citation block appended to the safe answer."""
        if not docs:
            return ""
        lines = ["", "**Sources:**"]
        for i, doc in enumerate(docs[:5], 1):
            lines.append(f"{i}. {doc.source} (relevance: {doc.score:.2f})")
        return "\n".join(lines)
