#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_from_step2.sh
# Full pipeline from Step 2 onwards for all 3 models + C2-C5 sweep.
# Run on Lightning Studio H100 after activations are already extracted.
#
# Prerequisites on Lightning:
#   cd ~/poolbench && git pull     ← must be done before running this script
#
# Usage:
#   bash scripts/run_from_step2.sh
#
# What this does (in order):
#   For each model (Mistral → Llama → Gemma):
#     Phase 1 — C1 DiffMean full pipeline, Steps 2-8
#     Phase 2 — C2/C3/C4 construction sweep (AUROC only, Appendix E)
#     Phase 3 — C5 SAE construction sweep (separate; requires SAELens weights)
#   Phase 4 — Nemenyi + layer rank correlation (after all models complete)
#   Phase 5 — External anchor validation (oracle eval, top-5 strategies)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

GPU="cuda:0"
PY="$(command -v python3 || command -v python)"

log() { echo -e "\n\033[1;34m[run_from_step2] $*\033[0m"; }
die() { echo -e "\033[1;31m[FATAL] $*\033[0m" >&2; exit 1; }

# ── Sanity check ─────────────────────────────────────────────────────────────
[[ -f scripts/run_model.py ]] || die "Must be run from the repo root or scripts/ parent."

log "Git status"
git --no-pager log --oneline -5

# ─────────────────────────────────────────────────────────────────────────────
# Per-model loop: C1 full pipeline + C2/C3/C4/C5 sweep, all within same model
# --force_from_step 2  →  recomputes Steps 2-8 (Step 1/extraction skipped)
#   Reason: A3_random seeding was wrong in D1 (now fixed); planning Classifier B
#   had wrong negatives (auto-retrain triggered by ANCHOR_FIXED_CONCEPTS).
#   Steps 3 (linearity) and 4 (ICC) are also forced so they are consistent
#   with the corrected Step 2 AUROC values.
# ─────────────────────────────────────────────────────────────────────────────

for MODEL in mistral_7b llama3_8b gemma2_9b; do

    log "PHASE 1 — $MODEL: C1 DiffMean full pipeline (Steps 2–8)"
    $PY scripts/run_model.py \
        --model "$MODEL" \
        --skip_extraction \
        --force_from_step 2 \
        --device "$GPU"

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2 (per model) — C2/C3/C4 construction sweep (Appendix E §35)
    # --skip_scp omits Steps 5-8 (SCP/D3/SAE) — only AUROC needed for the
    # Spearman ρ rank-correlation check vs C1.
    # Results go to results/auroc/{model}/{method}/ — never touches C1 files.
    # ─────────────────────────────────────────────────────────────────────────
    for METHOD in C2_pca C3_logreg C4_repe; do
        log "PHASE 2 — $MODEL: construction sweep method=$METHOD"
        $PY scripts/run_model.py \
            --model "$MODEL" \
            --skip_extraction \
            --skip_scp \
            --construction_method "$METHOD" \
            --force_from_step 2 \
            --device "$GPU"
    done

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3 (per model) — C5 SAE construction sweep
    # Separate from C2-C4 so a missing SAE weight file doesn't abort them.
    # ─────────────────────────────────────────────────────────────────────────
    log "PHASE 3 — $MODEL: C5 SAE construction sweep"
    $PY scripts/run_model.py \
        --model "$MODEL" \
        --skip_extraction \
        --skip_scp \
        --construction_method C5_sae_feature \
        --force_from_step 2 \
        --device "$GPU"

done

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Nemenyi + layer rank correlation (after all models complete)
# --nemenyi_only re-reads the saved AUROC files without re-running model steps.
# ─────────────────────────────────────────────────────────────────────────────

log "PHASE 4 — Nemenyi significance test + layer rank correlation"
$PY scripts/run_model.py \
    --nemenyi_only \
    --device cpu

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — External anchor validation (§42) — oracle eval on top-5 strategies
# run_oracle_eval.py reads the Nemenyi output to find the top-5 strategies,
# then re-evaluates them on the 17 external anchor datasets.
# This is the last step before compile_results.py (not yet written).
# ─────────────────────────────────────────────────────────────────────────────

log "PHASE 5 — Oracle / external anchor evaluation"
$PY scripts/run_oracle_eval.py --all --device "$GPU"

log "ALL PHASES COMPLETE"
echo ""
echo "Results tree:"
find results -name "*.json" | sort | head -60
