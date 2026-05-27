"""
Clinical domain adapter.

Provides clinical prompt guidelines, stricter safety thresholds, and medical
citations/disclaimer.
"""

from adapters.base import DomainAdapter, EvaluationSuite, PromptTemplate, SafetyPolicy
from core.models import Document


class MedicalAdapter(DomainAdapter):

    @property
    def name(self) -> str:
        return "medical"

    def get_retriever_config(self) -> dict:
        return {
            "collection": "mayaguard_medical",
            "embed_model_name": "sentence-transformers/all-MiniLM-L6-v2",
        }

    def get_prompt_template(self) -> PromptTemplate:
        return PromptTemplate(
            system=(
                "You are an expert clinical AI assistant. Base your answer strictly on the "
                "provided medical context. If the context does not contain enough information, "
                "clearly state that there is insufficient evidence in the retrieved literature. "
                "Maintain scientific accuracy and extreme clinical caution."
            ),
            user_template=(
                "Medical Context:\n{context}\n\n"
                "Patient Question:\n{query}\n\n"
                "Clinical Answer:"
            ),
        )

    def get_safety_policy(self) -> SafetyPolicy:
        return SafetyPolicy(
            risk_threshold_override=0.45,  # Stricter for patient safety
            always_add_disclaimer=True,
            refuse_on_critical=True,
            disclaimer_text=(
                "\n\n---\n*Medical Disclaimer: Vetted by MayaGuard against PubMed & WHO references. "
                "This response is for educational purposes only. Always consult a licensed "
                "medical professional before making healthcare decisions.*"
            ),
        )

    def get_evaluation_suite(self) -> EvaluationSuite:
        return EvaluationSuite(
            dataset_path="medical/medical_eval.jsonl",
            metric_names=["hallucination_rate", "faithfulness_mean", "f1"],
        )
