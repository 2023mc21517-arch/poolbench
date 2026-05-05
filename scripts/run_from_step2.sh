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
#   Phase 1 — C1 DiffMean pipeline, Steps 2-8, for each model
#   Phase 2 — Nemenyi + layer rank correlation (auto via --nemenyi_only)
#   Phase 3 — C2/C3/C4 construction sweep for all models (Appendix E)
#   Phase 4 — C5 SAE construction sweep (separate; requires SAELens weights)
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
# PHASE 1 — C1 DiffMean pipeline, all 3 models
# --force_from_step 2  →  recomputes Steps 2-8 (Step 1/extraction skipped)
#   Reason: A3_random seeding was wrong in D1 (now fixed); planning Classifier B
#   had wrong negatives (auto-retrain triggered by ANCHOR_FIXED_CONCEPTS).
#   Steps 3 (linearity) and 4 (ICC) are also forced so they are consistent
#   with the corrected Step 2 AUROC values.
# ─────────────────────────────────────────────────────────────────────────────

log "PHASE 1a — Mistral (Steps 2–8, forced from Step 2)"
$PY scripts/run_model.py \
    --model mistral_7b \
    --skip_extraction \
    --force_from_step 2 \
    --device "$GPU"

log "PHASE 1b — Llama (Steps 2–8, forced from Step 2)"
$PY scripts/run_model.py \
    --model llama3_8b \
    --skip_extraction \
    --force_from_step 2 \
    --device "$GPU"

log "PHASE 1c — Gemma (Steps 2–8, forced from Step 2)"
$PY scripts/run_model.py \
    --model gemma2_9b \
    --skip_extraction \
    --force_from_step 2 \
    --device "$GPU"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Step 9: Nemenyi + layer rank correlation
# All 3 models now complete → run significance test.
# layer_rank_correlation.py runs automatically inside run_model.py's main()
# but --nemenyi_only re-reads the saved AUROC files cleanly without re-running
# any model steps.
# ─────────────────────────────────────────────────────────────────────────────

log "PHASE 2 — Nemenyi significance test + layer rank correlation"
$PY scripts/run_model.py \
    --nemenyi_only \
    --device cpu

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — C2/C3/C4 construction sweep (Appendix E §35)
# Runs Step 2 only for each method × model combination.
# --skip_scp omits Steps 5-8 (SCP/D3/SAE) — only AUROC is needed for the
# Spearman ρ rank-correlation check vs C1.
# Results go to results/auroc/{model}/{method}/ — never touches C1 files.
# ─────────────────────────────────────────────────────────────────────────────

log "PHASE 3 — C2/C3/C4 construction sweep"

for METHOD in C2_pca C3_logreg C4_repe; do
    for MODEL in mistral_7b llama3_8b gemma2_9b; do
        log "  Construction sweep: model=$MODEL method=$METHOD"
        $PY scripts/run_model.py \
            --model "$MODEL" \
            --skip_extraction \
            --skip_scp \
            --construction_method "$METHOD" \
            --force_from_step 2 \
            --device "$GPU"
    done
done

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — C5 SAE construction sweep
# Requires SAELens weights to be available (GemmaScope / LlamaScope / jbloom).
# Runs separately so a missing SAE weight file doesn't abort phases 1-3.
# ─────────────────────────────────────────────────────────────────────────────

log "PHASE 4 — C5 SAE construction sweep"

for MODEL in mistral_7b llama3_8b gemma2_9b; do
    log "  C5 SAE sweep: model=$MODEL"
    $PY scripts/run_model.py \
        --model "$MODEL" \
        --skip_extraction \
        --skip_scp \
        --construction_method C5_sae_feature \
        --force_from_step 2 \
        --device "$GPU"
done

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
