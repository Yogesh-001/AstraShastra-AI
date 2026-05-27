from core.hallucination.detector import (
    HallucinationDetector,
    compute_faithfulness,
    score_to_risk,
    token_entropy,
)
from core.models import Claim, ClaimVerdict, RiskLevel


def make_verdict(supported: bool, confidence: float = 0.9) -> ClaimVerdict:
    return ClaimVerdict(
        claim=Claim(text="test claim"),
        supported=supported,
        confidence=confidence,
    )


class TestFaithfulness:
    def test_all_supported(self):
        verdicts = [make_verdict(True), make_verdict(True)]
        assert compute_faithfulness(verdicts) == 1.0

    def test_all_unsupported(self):
        verdicts = [make_verdict(False), make_verdict(False)]
        assert compute_faithfulness(verdicts) == 0.0

    def test_half_supported(self):
        verdicts = [make_verdict(True), make_verdict(False)]
        score = compute_faithfulness(verdicts)
        assert 0.4 < score < 0.6

    def test_empty(self):
        assert compute_faithfulness([]) == 1.0


class TestRiskLevel:
    def test_low(self):
        assert score_to_risk(0.1) == RiskLevel.LOW

    def test_medium(self):
        assert score_to_risk(0.4) == RiskLevel.MEDIUM

    def test_high(self):
        assert score_to_risk(0.65) == RiskLevel.HIGH

    def test_critical(self):
        assert score_to_risk(0.80) == RiskLevel.CRITICAL


class TestEntropy:
    def test_uniform_high_entropy(self):
        # Uniform distribution over 4 tokens → entropy = log2(4) = 2
        import math
        logprobs = [math.log(0.25)] * 4
        assert token_entropy(logprobs) > 1.9

    def test_peaked_low_entropy(self):
        import math
        logprobs = [math.log(0.97), math.log(0.01), math.log(0.01), math.log(0.01)]
        assert token_entropy(logprobs) < 0.5

    def test_empty(self):
        assert token_entropy([]) == 0.0


class TestDetector:
    def test_low_risk_report(self):
        detector = HallucinationDetector()
        report = detector.build_report(
            response_id="test-001",
            claim_verdicts=[make_verdict(True), make_verdict(True)],
            retrieved_documents=[],
            self_reflection_confidence=0.9,
        )
        assert report.overall_risk == RiskLevel.LOW
        assert report.risk_score < 0.35

    def test_high_risk_report(self):
        detector = HallucinationDetector()
        report = detector.build_report(
            response_id="test-002",
            claim_verdicts=[make_verdict(False), make_verdict(False)],
            retrieved_documents=[],
            self_reflection_confidence=0.1,
        )
        assert report.overall_risk in {RiskLevel.HIGH, RiskLevel.CRITICAL}
        assert report.risk_score >= 0.6
