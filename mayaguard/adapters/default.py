"""
adapters/default.py — Generic domain adapter (no domain specialisation).

Used when no adapter name is specified.  All domain-specific adapters
replace or extend these values.
"""

from adapters.base import DomainAdapter, EvaluationSuite, PromptTemplate, SafetyPolicy
from core.models import Document


class DefaultAdapter(DomainAdapter):

    @property
    def name(self) -> str:
        return "default"

    def get_retriever_config(self) -> dict:
        return {
            "collection": "mayaguard_core",
            "embed_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        }

    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            system=(
                "You are a helpful and accurate AI assistant. "
                "Base your answer only on the provided context. "
                "If the context does not contain enough information, say so."
            ),
            user_template=(
                "Context:\n{context}\n\n"
                "Question:\n{query}\n\n"
                "Answer:"
            ),
        )

    def get_safety_policy(self) -> SafetyPolicy:
        return SafetyPolicy(
            always_add_disclaimer=False,
            refuse_on_critical=True,
        )

    def get_evaluation_suite(self) -> EvaluationSuite:
        return EvaluationSuite(
            dataset_path="generic/sample_eval.jsonl",
            metric_names=["hallucination_rate", "faithfulness_mean", "f1"],
        )
