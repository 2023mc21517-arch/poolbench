#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_from_step2.sh
# Full pipeline from Step 2 onwards for all 3 models.
# Run on Lightning Studio H100 after activations are already extracted.
#
# Prerequisites on Lightning:
#   cd ~/poolbench && git pull     ← must be done before running this script
#
# Usage:
#   bash scripts/run_from_step2.sh
#
# What this does (in order):
#   For each model (Mistral → Llama → Gemma) — ONE invocation, never revisited:
#     Steps 3,2,4,4b,5,6,6b,7,8 (C1 DiffMean full pipeline)
#     C2/C3/C4/C5 construction sweep (AUROC only, Appendix E) — inside run_model()
#   Phase 2 — Nemenyi + layer rank correlation (after all models complete)
#   Phase 3 — External anchor validation (oracle eval, top-5 strategies)
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
# PHASE 1 — One invocation per model. run_model.py handles everything:
#   Steps 3 → 2 → 4 → 4b → 5 → 6 → 6b → 7 → 8  (C1 DiffMean full pipeline)
#   Then C2/C3/C4/C5 construction sweep (AUROC only) automatically at the end.
#
# Checkpoint-resume behaviour:
#   --force_from_step is NOT passed intentionally.  This means skip_existing=True
#   applies to every step — any step whose output file already exists on disk is
#   skipped instantly and the pipeline continues from where it left off.
#   Re-run with --force_from_step N only if you need to recompute from Step N.
#
# Historical note: a previous run used --force_from_step 2 to fix an A3_random
# seeding bug and a planning Classifier B issue.  Those fixes are now baked into
# the checkpoints; forcing a rerun is no longer necessary.
# ─────────────────────────────────────────────────────────────────────────────

for MODEL in mistral_7b llama3_8b gemma2_9b; do
    log "PHASE 1 — $MODEL: resume / complete pipeline (Steps 2–8 + C2–C5 sweep, skip existing)"
    $PY scripts/run_model.py \
        --model "$MODEL" \
        --skip_extraction \
        --device "$GPU"
done

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Nemenyi + layer rank correlation (after all models complete)
# ─────────────────────────────────────────────────────────────────────────────

log "PHASE 2 — Nemenyi significance test + layer rank correlation"
$PY scripts/run_model.py \
    --nemenyi_only \
    --device cpu

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — External anchor validation (§42) — oracle eval on top-5 strategies
# ─────────────────────────────────────────────────────────────────────────────

log "PHASE 3 — Oracle / external anchor evaluation"
$PY scripts/run_oracle_eval.py --all --device "$GPU"

log "ALL PHASES COMPLETE"
echo ""
echo "Results tree:"
find results -name "*.json" | sort | head -60
