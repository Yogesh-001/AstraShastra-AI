"""
Legal domain adapter.

Provides legal analysis prompt guidelines, strict risk thresholds, and legal
citations/disclaimer.
"""

from adapters.base import DomainAdapter, EvaluationSuite, PromptTemplate, SafetyPolicy
from core.models import Document


class LegalAdapter(DomainAdapter):

    @property
    def name(self) -> str:
        return "legal"

    def get_retriever_config(self) -> dict:
        return {
            "collection": "mayaguard_legal",
            "embed_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        }

    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            system=(
                "You are an expert legal AI assistant. Base your answer strictly on the "
                "provided legal references and statutes. Explain legal concepts and contracts "
                "with high precision. If the context does not contain enough information, "
                "clearly state that there is insufficient evidence in the retrieved statutes. "
                "Do not offer formal legal representation or binding advice."
            ),
            user_template=(
                "Legal Context:\n{context}\n\n"
                "Legal Question:\n{query}\n\n"
                "Legal Analysis:"
            ),
        )

    def get_safety_policy(self) -> SafetyPolicy:
        return SafetyPolicy(
            risk_threshold_override=0.50,  # Strict for legal compliance
            always_add_disclaimer=True,
            refuse_on_critical=True,
            disclaimer_text=(
                "\n\n---\n*Legal Disclaimer: Vetted by MayaGuard against statutory references. "
                "This response does not constitute formal legal representation or binding advice. "
                "Always consult a qualified attorney for legal counsel.*"
            ),
        )

    def get_evaluation_suite(self) -> EvaluationSuite:
        return EvaluationSuite(
            dataset_path="legal/legal_eval.jsonl",
            metric_names=["hallucination_rate", "faithfulness_mean", "f1"],
        )
