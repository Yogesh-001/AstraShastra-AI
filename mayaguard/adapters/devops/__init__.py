"""
DevOps and Systems Engineering domain adapter.

Provides DevOps-specific prompt guidelines, technical precision thresholds,
and citation formatting.
"""

from adapters.base import DomainAdapter, EvaluationSuite, PromptTemplate, SafetyPolicy
from core.models import Document


class DevopsAdapter(DomainAdapter):

    @property
    def name(self) -> str:
        return "devops"

    def get_retriever_config(self) -> dict:
        return {
            "collection": "mayaguard_devops",
            "embed_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        }

    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            system=(
                "You are an expert DevOps and systems engineering assistant. Base your answer "
                "strictly on the provided system configuration and architectural documentation. "
                "Ensure command syntax, configuration snippets, and security procedures are "
                "highly precise. If the context does not contain enough information, state "
                "clearly that there is insufficient evidence in the retrieved documentation."
            ),
            user_template=(
                "System Documentation:\n{context}\n\n"
                "Infrastructure/Troubleshooting Question:\n{query}\n\n"
                "Systems Answer:"
            ),
        )

    def get_safety_policy(self) -> SafetyPolicy:
        return SafetyPolicy(
            risk_threshold_override=0.55,  # Moderate for infrastructure reliability
            always_add_disclaimer=False,   # Keep technical output concise
            refuse_on_critical=True,
            disclaimer_text="",
        )

    def get_evaluation_suite(self) -> EvaluationSuite:
        return EvaluationSuite(
            dataset_path="devops/devops_eval.jsonl",
            metric_names=["hallucination_rate", "faithfulness_mean", "f1"],
        )
