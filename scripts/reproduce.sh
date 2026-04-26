#!/usr/bin/env bash
# scripts/reproduce.sh
# Full end-to-end reproduction of PoolBench paper results.
# Prerequisites: pip install -e ".[dev]" && python -m spacy download en_core_web_sm
# GPU: single CUDA GPU with ≥20 GB VRAM recommended.
# CPU time: ~4 hours (corpus build). GPU time: ~2–4 hours per model.

set -euo pipefail

DEVICE="${1:-cuda:0}"
echo "Reproducing PoolBench on device: $DEVICE"

# ── Step 1: Corpus construction (CPU, ~4 hours total) ─────────────────────────
echo ""
echo "=== Step 1: Building corpora ==="
for concept in hedging legal_formality math_certainty frustration pos_sentiment \
               toxicity depression causation contrast conditionality academic_tone \
               code_docs bureaucratic uncertainty deference planning \
               negation_density numerical_precision; do
    echo "  Building: $concept"
    python scripts/dataset_builder.py --concept "$concept"
done

# ── Step 2: Power analysis ────────────────────────────────────────────────────
echo ""
echo "=== Step 2: Power analysis ==="
python scripts/power_analysis.py

# ── Step 3: Per-model experiments ────────────────────────────────────────────
echo ""
echo "=== Step 3: Running experiments ==="
for model in llama3_8b gemma2_9b mistral_7b qwen25_7b flan_t5_xl mamba2_2b7 bert_base_uncased; do
    echo ""
    echo "--- Model: $model ---"
    python scripts/run_model.py --model "$model" --device "$DEVICE"
done

# ── Step 4: Cross-model Nemenyi significance test ─────────────────────────────
echo ""
echo "=== Step 4: Nemenyi test ==="
python scripts/run_model.py --nemenyi_only

echo ""
echo "Reproduction complete. Results in results/"
