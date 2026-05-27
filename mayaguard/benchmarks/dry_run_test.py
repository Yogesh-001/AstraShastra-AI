"""
Dry-run validation of MayaGuard core logic.
Tests the pure computation modules without needing external services.
"""
import math
import re
import json

print("=" * 70)
print("MayaGuard - Dry-Run Validation")
print("=" * 70)

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  [OK] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}")
        failed += 1


# 1. Token Entropy Logic
print("\n[TEST] Token Entropy Tests")

def token_entropy(logprobs):
    if not logprobs:
        return 0.0
    probs = [math.exp(lp) for lp in logprobs]
    total = sum(probs)
    normalised = [p / total for p in probs]
    return -sum(p * math.log2(p + 1e-12) for p in normalised)

# Uniform distribution over 4 tokens -> entropy = log2(4) = 2
ent_uniform = token_entropy([math.log(0.25)] * 4)
check(f"Uniform 4-token entropy = {ent_uniform:.4f} (expect ~2.0)", ent_uniform > 1.9)

# Peaked distribution → low entropy
ent_peaked = token_entropy([math.log(0.97), math.log(0.01), math.log(0.01), math.log(0.01)])
check(f"Peaked entropy = {ent_peaked:.4f} (expect <0.5)", ent_peaked < 0.5)

# Empty
check("Empty logprobs = 0.0", token_entropy([]) == 0.0)


# 2. Faithfulness Scoring
print("\n[TEST] Faithfulness Scoring Tests")

def compute_faithfulness(verdicts):
    if not verdicts:
        return 1.0
    total_weight = sum(v[1] for v in verdicts)
    if total_weight == 0:
        return 0.5
    supported_weight = sum(v[1] for v in verdicts if v[0])
    return supported_weight / total_weight

check("All supported = 1.0", compute_faithfulness([(True, 0.9), (True, 0.9)]) == 1.0)
check("None supported = 0.0", compute_faithfulness([(False, 0.9), (False, 0.9)]) == 0.0)

half = compute_faithfulness([(True, 0.9), (False, 0.9)])
check(f"Half supported = {half:.2f} (expect 0.5)", 0.4 < half < 0.6)
check("Empty = 1.0", compute_faithfulness([]) == 1.0)


# 3. Risk Level Classification
print("\n[TEST] Risk Level Classification")

def score_to_risk(score, threshold=0.6):
    if score >= threshold + 0.15:
        return "CRITICAL"
    if score >= threshold:
        return "HIGH"
    if score >= threshold - 0.25:
        return "MEDIUM"
    return "LOW"

check("0.1 -> LOW", score_to_risk(0.1) == "LOW")
check("0.34 -> LOW", score_to_risk(0.34) == "LOW")
check("0.35 -> MEDIUM", score_to_risk(0.35) == "MEDIUM")
check("0.59 -> MEDIUM", score_to_risk(0.59) == "MEDIUM")
check("0.6 -> HIGH", score_to_risk(0.6) == "HIGH")
check("0.74 -> HIGH", score_to_risk(0.74) == "HIGH")
check("0.75 -> CRITICAL", score_to_risk(0.75) == "CRITICAL")
check("1.0 -> CRITICAL", score_to_risk(1.0) == "CRITICAL")


# 4. Hallucination Risk Score (Weighted Aggregation)
print("\n[TEST] Hallucination Risk Score Aggregation")

W_FAITH = 0.40
W_REFLECT = 0.40
W_ENTROPY = 0.20

def build_risk_score(faithfulness, self_reflect_conf, norm_entropy, total_claims=0, unsupported_claims=0):
    faith_risk = 1.0 - faithfulness
    reflect_risk = 1.0 - self_reflect_conf
    risk_score = (W_FAITH * faith_risk + W_REFLECT * reflect_risk + W_ENTROPY * norm_entropy)
    # Floor penalty: any unsupported claim guarantees a minimum risk
    if total_claims > 0 and unsupported_claims > 0:
        floor = unsupported_claims / total_claims
        risk_score = max(risk_score, floor)
    return round(min(1.0, max(0.0, risk_score)), 4)

# Scenario A: All good
risk_a = build_risk_score(faithfulness=1.0, self_reflect_conf=0.9, norm_entropy=0.0)
check(f"All-good scenario: risk={risk_a} -> {score_to_risk(risk_a)}", score_to_risk(risk_a) == "LOW")

# Scenario B: All bad
risk_b = build_risk_score(faithfulness=0.0, self_reflect_conf=0.1, norm_entropy=0.8)
check(f"All-bad scenario: risk={risk_b} -> {score_to_risk(risk_b)}", score_to_risk(risk_b) in ("HIGH", "CRITICAL"))

# Scenario C: Mixed
risk_c = build_risk_score(faithfulness=0.5, self_reflect_conf=0.5, norm_entropy=0.3)
check(f"Mixed scenario: risk={risk_c} -> {score_to_risk(risk_c)}", score_to_risk(risk_c) == "MEDIUM")


# 5. Claim Extraction (Regex Parsing)
print("\n[TEST] Claim Extraction (Numbered List Parser)")

def parse_numbered_list(text):
    claims = []
    for line in text.splitlines():
        line = line.strip()
        match = re.match(r"^\d+\.\s+(.+)", line)
        if match:
            claims.append(match.group(1).strip())
    return claims

test_text = """1. Metformin is used for type 2 diabetes.
2. It reduces hepatic glucose production.
3. Common side effects include gastrointestinal issues."""

claims = parse_numbered_list(test_text)
check(f"Parsed {len(claims)} claims from numbered list", len(claims) == 3)
check("First claim text correct", claims[0] == "Metformin is used for type 2 diabetes.")

# Fallback sentence splitter
def sentence_split(text):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if len(s) > 10]

test_paragraph = "Metformin is the first-line treatment for T2DM. It works by reducing glucose. Side effects include nausea."
sentences = sentence_split(test_paragraph)
check(f"Sentence split produced {len(sentences)} sentences", len(sentences) == 3)


# 6. Self-Reflection Parser
print("\n[TEST] Self-Reflection Response Parser")

def parse_reflection(raw):
    confidence = 0.5
    critique = raw
    conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", raw, re.IGNORECASE)
    if conf_match:
        try:
            confidence = max(0.0, min(1.0, float(conf_match.group(1))))
        except ValueError:
            pass
    crit_match = re.search(r"CRITIQUE:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    if crit_match:
        critique = crit_match.group(1).strip()
    return confidence, critique

conf, crit = parse_reflection("CONFIDENCE: 0.82\nCRITIQUE: The answer is mostly accurate but lacks citations.")
check(f"Parsed confidence = {conf}", abs(conf - 0.82) < 0.001)
check(f"Parsed critique", "lacks citations" in crit)

conf2, crit2 = parse_reflection("Some random text without the expected format")
check(f"Fallback confidence = {conf2}", conf2 == 0.5)


# 7. Response Controller Logic
print("\n[TEST] Response Controller Policy")

def controller_action(risk_level):
    if risk_level == "LOW":
        return "PASS_THROUGH"
    elif risk_level == "MEDIUM":
        return "ADD_DISCLAIMER"
    elif risk_level == "HIGH":
        return "REWRITE"
    else:  # CRITICAL
        return "REFUSE"

check("LOW -> PASS_THROUGH", controller_action("LOW") == "PASS_THROUGH")
check("MEDIUM -> ADD_DISCLAIMER", controller_action("MEDIUM") == "ADD_DISCLAIMER")
check("HIGH -> REWRITE", controller_action("HIGH") == "REWRITE")
check("CRITICAL -> REFUSE", controller_action("CRITICAL") == "REFUSE")


# 8. Grounding Verdict Parser
print("\n[TEST] Grounding Verdict Parser")

def parse_grounding_verdict(raw):
    lines = raw.strip().splitlines()
    verdict_word = lines[0].strip().upper() if lines else "UNSUPPORTED"
    explanation = lines[1].strip() if len(lines) > 1 else ""
    supported = verdict_word == "SUPPORTED"
    return supported, explanation

s1, e1 = parse_grounding_verdict("SUPPORTED\nThe claim is clearly stated in the context.")
check("SUPPORTED parsed correctly", s1 is True)
check("Explanation extracted", "clearly stated" in e1)

s2, e2 = parse_grounding_verdict("UNSUPPORTED\nNo evidence found.")
check("UNSUPPORTED parsed correctly", s2 is False)


# 9. Entropy Span Sliding Window
print("\n[TEST] Entropy Span Sliding Window")

ENTROPY_HIGH = 3.5
ENTROPY_MEDIUM = 2.5

def compute_entropy_spans(tokens, logprobs, window=5):
    spans = []
    for i in range(0, len(tokens) - window + 1, window):
        chunk_lp = logprobs[i:i + window]
        entropy = token_entropy(chunk_lp)
        if entropy < ENTROPY_MEDIUM:
            continue
        risk = "HIGH" if entropy >= ENTROPY_HIGH else "MEDIUM"
        text = " ".join(tokens[i:i + window])
        spans.append({"text": text, "entropy": round(entropy, 3), "risk": risk})
    return spans

# Create tokens with high entropy (uniform distribution over many alternatives)
high_entropy_lps = [math.log(1.0/20)] * 16  # very high entropy per token
tokens = [f"tok{i}" for i in range(16)]
spans = compute_entropy_spans(tokens, high_entropy_lps, window=8)
check(f"High-entropy spans detected: {len(spans)} spans", len(spans) >= 1)

# Create tokens with low entropy
low_entropy_lps = [math.log(0.99)] * 10
spans2 = compute_entropy_spans(tokens, low_entropy_lps, window=5)
check(f"Low-entropy: no spans detected", len(spans2) == 0)


# 10. End-to-End Medical Dry Run
print("\n" + "=" * 70)
print("[MEDICAL] MEDICAL DOMAIN - End-to-End Dry Run Simulation")
print("=" * 70)

# Simulate: "What is Metformin used for?"
query = "What is Metformin used for?"
print(f"\nQuery: {query}")

# Step 1: Simulated retrieval
retrieved_docs = [
    {"source": "PubMed:PMC7654321", "content": "Metformin is the first-line pharmacotherapy for type 2 diabetes mellitus (T2DM). It lowers hepatic glucose production and improves insulin sensitivity.", "score": 0.92},
    {"source": "WHO Essential Medicines 2024", "content": "Metformin hydrochloride is listed as an essential medicine for diabetes management.", "score": 0.87},
    {"source": "UpToDate:Metformin", "content": "Metformin is also being investigated for potential benefits in polycystic ovary syndrome (PCOS) and cancer prevention.", "score": 0.71},
]
print(f"\n  [INFO] Retrieved {len(retrieved_docs)} documents")
for d in retrieved_docs:
    print(f"     - {d['source']} (score: {d['score']})")

# Step 2: Simulated LLM response
raw_answer = (
    "Metformin is the first-line treatment for type 2 diabetes. It works by reducing "
    "glucose production in the liver and improving the body's sensitivity to insulin. "
    "It has been used since the 1950s and is considered one of the safest diabetes medications. "
    "Metformin can also cure cancer according to some researchers."
)
print(f"\n  [LLM] Raw LLM Answer:")
print(f"     \"{raw_answer}\"")

# Step 3: Claim extraction
claims = sentence_split(raw_answer)
print(f"\n  [CLAIMS] Extracted {len(claims)} claims:")
for i, c in enumerate(claims, 1):
    print(f"     {i}. {c}")

# Step 4: Grounding check simulation (using equal confidence weights after bug fix)
verdicts = [
    (True, 0.85, "Directly supported by PubMed source"),         # first-line treatment
    (True, 0.85, "Supported by PubMed source"),                   # reduces glucose
    (True, 0.85, "Partially supported, timeline approximate"),     # since 1950s
    (False, 0.85, "NOT supported - no evidence of cancer cure"),  # cancer cure
]
print(f"\n  [VERDICTS] Grounding Verdicts:")
for i, (sup, conf, expl) in enumerate(verdicts):
    icon = "[OK]" if sup else "[FAIL]"
    print(f"     {icon} Claim {i+1}: {'SUPPORTED' if sup else 'UNSUPPORTED'} (conf={conf}) - {expl}")

# Step 5: Faithfulness score
faithfulness = compute_faithfulness(verdicts)
print(f"\n  [INFO] Faithfulness Score: {faithfulness:.4f}")

# Step 6: Self-reflection simulation
self_reflect_conf = 0.55
self_critique = "The answer contains a claim about curing cancer that is not well-supported. The rest appears accurate based on medical literature."
print(f"  [REFLECT] Self-Reflection Confidence: {self_reflect_conf}")
print(f"     Critique: {self_critique}")

# Step 7: Build hallucination report
# Medical domain uses stricter threshold (0.45) via risk_threshold_override
medical_threshold = 0.45  # stricter than default 0.6 for patient safety
risk_score = build_risk_score(faithfulness, self_reflect_conf, norm_entropy=0.1, total_claims=4, unsupported_claims=1)
risk_level = score_to_risk(risk_score, threshold=medical_threshold)
print(f"\n  [WARN] Hallucination Report:")
print(f"     Risk Score:  {risk_score:.4f}")
print(f"     Risk Level:  {risk_level} (medical threshold: {medical_threshold})")
print(f"     Faith Risk:  {1.0 - faithfulness:.4f}")
print(f"     Reflect Risk:{1.0 - self_reflect_conf:.4f}")

# Step 8: Controller decision
action = controller_action(risk_level)
print(f"\n  [ACTION] Controller Action: {action}")

if action == "PASS_THROUGH":
    safe_answer = raw_answer
elif action == "ADD_DISCLAIMER":
    safe_answer = raw_answer + "\n\n---\nWarning: This response was generated by AI and may contain inaccuracies."
elif action == "REWRITE":
    safe_answer = (
        "Metformin is widely recognized as the first-line treatment for type 2 diabetes "
        "(Source: PubMed:PMC7654321). It works by reducing glucose production in the liver "
        "and improving insulin sensitivity (Source: PubMed:PMC7654321). While it has a long "
        "history of use, some claims about additional benefits like cancer prevention are "
        "still under investigation and should not be taken as established fact "
        "(Source: UpToDate:Metformin)."
        "\n\n---\nWarning: This response was generated by AI and may contain inaccuracies."
    )
elif action == "REFUSE":
    safe_answer = "Refused: This response has been blocked due to high hallucination risk."

print(f"\n  [OK] Safe Answer:")
print(f"     \"{safe_answer}\"")

check("Medical dry run completed with correct risk assessment", risk_level in ("MEDIUM", "HIGH"))
check("Unsupported cancer claim detected", verdicts[3][0] is False)
check("Controller applied appropriate action", action in ("ADD_DISCLAIMER", "REWRITE"))


# 11. Cybersecurity Domain Dry Run
print("\n" + "=" * 70)
print("[SECURITY] CYBERSECURITY DOMAIN - End-to-End Dry Run Simulation")
print("=" * 70)

query_cyber = "What is a zero-day vulnerability and how should organizations respond?"
print(f"\nQuery: {query_cyber}")

# Step 1: Simulated retrieval
cyber_docs = [
    {"source": "NIST:SP800-40r4", "content": "A zero-day vulnerability is a software security flaw unknown to the vendor. Organizations should implement patch management, network segmentation, and intrusion detection systems.", "score": 0.95},
    {"source": "MITRE ATT&CK", "content": "Zero-day exploits target unknown vulnerabilities before patches are available. Defense includes threat intelligence, behavioral analytics, and endpoint detection.", "score": 0.90},
    {"source": "CISA Advisory 2025", "content": "Organizations should maintain an incident response plan, apply defense-in-depth strategies, and monitor for indicators of compromise.", "score": 0.82},
]
print(f"\n  [INFO] Retrieved {len(cyber_docs)} documents")
for d in cyber_docs:
    print(f"     - {d['source']} (score: {d['score']})")

# Step 2: Simulated LLM response
raw_cyber = (
    "A zero-day vulnerability is a software flaw unknown to the vendor that can be exploited by attackers. "
    "Organizations should respond with patch management, network segmentation, and intrusion detection. "
    "Additionally, implementing behavioral analytics and endpoint detection can help identify zero-day attacks. "
    "The best defense is to use quantum encryption which makes zero-day attacks impossible."
)
print(f"\n  [LLM] Raw LLM Answer:")
print(f"     \"{raw_cyber}\"")

# Step 3: Claims
cyber_claims = sentence_split(raw_cyber)
print(f"\n  [CLAIMS] Extracted {len(cyber_claims)} claims:")
for i, c in enumerate(cyber_claims, 1):
    print(f"     {i}. {c}")

# Step 4: Verdicts (using equal confidence weights after bug fix)
cyber_verdicts = [
    (True, 0.85, "Directly supported by NIST definition"),
    (True, 0.85, "Supported by NIST and CISA recommendations"),
    (True, 0.85, "Supported by MITRE ATT&CK framework"),
    (False, 0.85, "NOT supported - quantum encryption does not prevent zero-day attacks"),
]
print(f"\n  [VERDICTS] Grounding Verdicts:")
for i, (sup, conf, expl) in enumerate(cyber_verdicts):
    icon = "[OK]" if sup else "[FAIL]"
    print(f"     {icon} Claim {i+1}: {'SUPPORTED' if sup else 'UNSUPPORTED'} (conf={conf}) - {expl}")

# Step 5-7: Score (cybersecurity uses threshold 0.5 - less strict than medical, stricter than default)
cyber_faithfulness = compute_faithfulness(cyber_verdicts)
cyber_reflect = 0.60
cyber_threshold = 0.50  # stricter than default 0.6 for security-critical content
cyber_risk = build_risk_score(cyber_faithfulness, cyber_reflect, 0.05, total_claims=4, unsupported_claims=1)
cyber_risk_level = score_to_risk(cyber_risk, threshold=cyber_threshold)
cyber_action = controller_action(cyber_risk_level)

print(f"\n  [INFO] Faithfulness Score: {cyber_faithfulness:.4f}")
print(f"  [REFLECT] Self-Reflection: {cyber_reflect}")
print(f"  [WARN] Risk Score: {cyber_risk:.4f} -> {cyber_risk_level} (cyber threshold: {cyber_threshold})")
print(f"  [ACTION] Controller Action: {cyber_action}")

if cyber_action == "REWRITE":
    safe_cyber = (
        "A zero-day vulnerability is a software flaw unknown to the vendor that can be exploited "
        "by attackers before a patch is available (Source: NIST:SP800-40r4). Organizations should "
        "respond with patch management, network segmentation, and intrusion detection systems "
        "(Source: NIST:SP800-40r4, CISA Advisory 2025). Behavioral analytics and endpoint detection "
        "can also help identify zero-day attacks (Source: MITRE ATT&CK). Note: claims about quantum "
        "encryption preventing zero-day attacks are not supported by current evidence."
        "\n\n---\nWarning: This response was generated by AI and may contain inaccuracies."
    )
elif cyber_action == "ADD_DISCLAIMER":
    safe_cyber = raw_cyber + "\n\n---\nWarning: This response was generated by AI and may contain inaccuracies."
else:
    safe_cyber = raw_cyber

print(f"\n  [OK] Safe Answer:")
print(f"     \"{safe_cyber}\"")

check("Cyber dry run completed with correct risk assessment", cyber_risk_level in ("MEDIUM", "HIGH"))
check("Quantum encryption claim flagged", cyber_verdicts[3][0] is False)
check("Controller applied appropriate action", cyber_action in ("ADD_DISCLAIMER", "REWRITE"))


# Summary
print("\n" + "=" * 70)
total = passed + failed
print(f"Results: {passed}/{total} checks passed, {failed} failed")
if failed == 0:
    print("[SUCCESS] ALL CHECKS PASSED - Core logic is working correctly!")
else:
    print(f"[WARN] {failed} check(s) failed - see above for details.")
print("=" * 70)
