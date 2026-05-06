#!/usr/bin/env bash
# post_pipeline.sh — Run all post-pipeline steps for all 3 models.
# Run this AFTER all 3 models have completed Steps 1-8 + construction sweep.
#
# Usage:
#   bash scripts/post_pipeline.sh cuda:0 hf_YOUR_TOKEN nips234678
#
# Arguments:
#   $1 = device      (e.g. cuda:0)
#   $2 = HF token    (e.g. hf_xxxx)
#   $3 = HF username (e.g. nips234678)

set -euo pipefail

DEVICE="${1:?Pass device as first arg, e.g. cuda:0}"
HF_TOKEN="${2:?Pass HF token as second arg}"
HF_USER="${3:?Pass HF username as third arg}"

MODELS=("mistral_7b" "llama3_8b" "gemma2_9b")

# ── 1. Save ITI head scores (all 3 models) ────────────────────────────────────
echo "=== [1/4] Saving ITI head scores ==="
for MODEL in "${MODELS[@]}"; do
    echo "  → $MODEL"
    python scripts/save_iti_scores.py --model "$MODEL" --device "$DEVICE"
done

# ── 2. Save steering vectors (all 3 models) ───────────────────────────────────
echo ""
echo "=== [2/4] Saving steering vectors ==="
for MODEL in "${MODELS[@]}"; do
    echo "  → $MODEL"
    python scripts/save_steering_vectors.py --model "$MODEL" --device "$DEVICE"
done

# ── 3. Upload steering vectors to HuggingFace ────────────────────────────────
# (activations + bert-scorers already uploaded; SCP/D3 JSONs go to git repo)
echo ""
echo "=== [3/4] Uploading steering vectors to HuggingFace ==="
python scripts/upload_to_hf.py --token "$HF_TOKEN" --user "$HF_USER" --only steering-vectors

echo ""
echo "=== Done. ==="
