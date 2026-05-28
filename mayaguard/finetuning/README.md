# 🧬 MayaGuard Fine-Tuning Guide

This directory contains the complete fine-tuning pipeline for MayaGuard. Two models are trained using parameter-efficient methods (LoRA and QLoRA) to enhance the hallucination detection and generation capabilities of the system.

---

## 📁 Directory Structure

```
finetuning/
├── README.md                  ← You are here
├── data/
│   ├── generate_nli_dataset.py       # NLI pair generator for claim verifier
│   ├── generate_instruct_dataset.py  # Instruction dataset for LLM fine-tuning
│   ├── train.jsonl                   # (generated) NLI training data
│   ├── val.jsonl                     # (generated) NLI validation data
│   ├── instruct_train.jsonl          # (generated) instruction training data
│   └── instruct_val.jsonl            # (generated) instruction validation data
├── notebooks/
│   ├── finetune_claim_verifier.ipynb  # Colab notebook - LoRA on DeBERTa
│   └── finetune_qlora_llm.ipynb       # Colab notebook - QLoRA on Mistral-7B
└── outputs/                           # (gitignored) trained adapter weights
    ├── claim_verifier_adapter/
    └── qlora_adapter/
```

---

## 🔬 Model 1: Claim Verification Classifier (LoRA)

### What It Does
Replaces the LLM-based `GroundingChecker` with a fast, specialized **DeBERTa-v3-base** NLI classifier. Given a `(claim, context)` pair, it predicts `SUPPORTED` or `UNSUPPORTED` in ~50ms per claim (vs 2-5 seconds with the LLM-based checker).

### Technical Details
| Property | Value |
|----------|-------|
| Base Model | `microsoft/deberta-v3-base` (184M params) |
| Method | LoRA (Low-Rank Adaptation) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Target Modules | `query_proj`, `value_proj` |
| Training Data | NLI pairs from MayaGuard seed datasets |
| Training Time | ~15-30 minutes on Colab T4 |
| Adapter Size | ~2-5 MB (vs 700 MB base model) |

### Training Steps

#### Step 1: Generate Training Data
```bash
cd mayaguard
python -m finetuning.data.generate_nli_dataset
```
This transforms the seed JSON files (`benchmarks/datasets/*.json`) into NLI training pairs:
- **Positive pairs:** Sentences extracted from documents, paired with their source → `SUPPORTED`
- **Negative pairs:** Cross-document mismatches + synthetic hallucinations → `UNSUPPORTED`

#### Step 2: Train on Google Colab
1. Open `finetuning/notebooks/finetune_claim_verifier.ipynb` in Google Colab
2. Switch to **T4 GPU** runtime
3. Upload `finetuning/data/train.jsonl` and `finetuning/data/val.jsonl`
4. Run all cells
5. Download the exported `claim_verifier_adapter.zip`

#### Step 3: Deploy Locally
```bash
# Unzip the adapter
mkdir -p finetuning/outputs/claim_verifier_adapter
unzip claim_verifier_adapter.zip -d finetuning/outputs/claim_verifier_adapter/

# Enable in .env
echo "FINETUNED_VERIFIER_PATH=finetuning/outputs/claim_verifier_adapter" >> .env
echo "FINETUNED_VERIFIER_ENABLED=true" >> .env

# Restart the API server
python main.py serve
```

---

## 🧠 Model 2: Hallucination-Aware LLM (QLoRA)

### What It Does
Fine-tunes **Mistral-7B-Instruct-v0.2** with QLoRA to generate more grounded, citation-aware responses. The fine-tuned model learns to:
- Cite sources explicitly with `(Source: X)` notation
- Refuse to answer when evidence is insufficient
- Ground all claims in the provided context documents

### Technical Details
| Property | Value |
|----------|-------|
| Base Model | `mistralai/Mistral-7B-Instruct-v0.2` (7B params) |
| Method | QLoRA (Quantized LoRA) |
| Quantization | 4-bit NF4 with double quantization |
| LoRA Rank | 64 |
| LoRA Alpha | 16 |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Optimizer | `paged_adamw_8bit` |
| Training Data | Instruction pairs from MayaGuard seed datasets |
| Training Time | ~45-90 minutes on Colab T4 |
| VRAM Usage | ~10-12 GB (fits on T4 16GB) |
| Adapter Size | ~50-100 MB (vs 14 GB base model) |

### Training Steps

#### Step 1: Generate Training Data
```bash
cd mayaguard
python -m finetuning.data.generate_instruct_dataset
```
This creates chat-formatted instruction examples:
- **System prompt:** Trustworthy AI assistant with citation requirements
- **User messages:** Domain queries with retrieved context
- **Assistant responses:** Grounded answers with citations, or safe refusals

#### Step 2: Train on Google Colab
1. Open `finetuning/notebooks/finetune_qlora_llm.ipynb` in Google Colab
2. Switch to **T4 GPU** runtime
3. Upload `finetuning/data/instruct_train.jsonl` and `finetuning/data/instruct_val.jsonl`
4. Run all cells (the notebook includes checkpoint resume logic for Colab disconnections)
5. Download the exported `qlora_adapter.zip`

#### Step 3: Deploy with vLLM
The QLoRA adapter can be served alongside the base model using vLLM's native LoRA support:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --enable-lora \
    --lora-modules mayaguard=finetuning/outputs/qlora_adapter \
    --max-model-len 4096
```

Then set `VLLM_ENABLED=true` in your `.env` and restart MayaGuard.

---

## ⚡ Memory Budget (Colab Free Tier T4)

### Claim Verifier (LoRA on DeBERTa)
| Component | VRAM |
|-----------|------|
| Base model (FP32) | ~0.7 GB |
| LoRA adapters | ~0.01 GB |
| Training overhead | ~2-3 GB |
| **Total** | **~3-4 GB** ✅ |

### LLM (QLoRA on Mistral-7B)
| Component | VRAM |
|-----------|------|
| Base model (4-bit NF4) | ~4.5 GB |
| LoRA adapters | ~0.3 GB |
| Optimizer states (8-bit) | ~1.5 GB |
| Activations + gradients | ~4-6 GB |
| **Total** | **~10-12 GB** ✅ |

---

## 🔧 Configuration Reference

Add these to your `.env` file after training:

```env
# Claim Verification Classifier (LoRA on DeBERTa)
FINETUNED_VERIFIER_PATH=finetuning/outputs/claim_verifier_adapter
FINETUNED_VERIFIER_ENABLED=true

# Hallucination-Aware LLM (QLoRA on Mistral-7B)
QLORA_ADAPTER_PATH=finetuning/outputs/qlora_adapter
QLORA_ADAPTER_ENABLED=true
```

---

## 📊 Expected Results

### Claim Verifier
- **Accuracy:** 85-92% on validation set
- **F1 Score:** 0.83-0.90
- **Inference Speed:** ~50ms per claim (vs 2-5s with LLM)

### QLoRA LLM
- **Qualitative:** More citations, fewer hallucinations, appropriate refusals
- **Training Loss:** Should converge to < 1.0 within 3 epochs
- **Validation:** Manual inspection of generated outputs for citation quality
