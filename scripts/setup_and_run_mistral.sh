#!/usr/bin/env bash
# ============================================================
# PoolBench — full production run for mistral_7b on A100 40GB
# Run this script once on a fresh instance:
#   bash setup_and_run_mistral.sh
# ============================================================
set -euo pipefail

# ── 0. Config — override with env vars if needed ────────────
REPO_URL="${REPO_URL:-https://github.com/2023mc21517-arch/poolbench.git}"
HF_TOKEN="${HF_TOKEN:-}"
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
# ────────────────────────────────────────────────────────────

REPO_DIR="$HOME/poolbench"
VENV_DIR="$HOME/venvs/poolbench"
LOG_DIR="$REPO_DIR/results/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/mistral_7b_full_${TIMESTAMP}.log"
USE_CURRENT_ENV="${POOLBENCH_USE_CURRENT_ENV:-0}"

echo "============================================================"
echo " PoolBench setup + mistral_7b production run"
echo " Log: $LOG_FILE"
echo "============================================================"

if [[ -z "$HF_TOKEN" ]]; then
    echo "ERROR: export HF_TOKEN before running this script"
    exit 1
fi
if [[ -z "$ANTHROPIC_API_KEY" ]]; then
    echo "ERROR: export ANTHROPIC_API_KEY before running this script"
    exit 1
fi

# ── 1. Clone or update repo ──────────────────────────────────
if [[ -d "$REPO_DIR/.git" ]]; then
    echo "[1/7] Repo exists — pulling latest..."
    cd "$REPO_DIR"
    git pull origin main
else
    echo "[1/7] Cloning repo..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi
echo "  Commit: $(git rev-parse --short HEAD)"

# ── 2. Python venv ──────────────────────────────────────────
if [[ "$USE_CURRENT_ENV" == "1" ]]; then
        echo "[2/7] Using current Python environment (POOLBENCH_USE_CURRENT_ENV=1)..."
        python -m pip install --upgrade pip setuptools wheel
        python -m pip install --no-cache-dir --force-reinstall \
            --index-url https://download.pytorch.org/whl/cu121 \
            "torch==2.5.1" \
            "torchvision==0.20.1"
        python -m pip install --no-cache-dir --force-reinstall \
            "numpy==1.26.4" \
            "scipy==1.12.0" \
            "scikit-learn==1.4.2"
        python -m pip install --no-cache-dir --force-reinstall \
            "transformers==4.46.3" \
            "tokenizers<0.21" \
            "huggingface_hub<1.0"
        python -m pip install --no-cache-dir -e .
        python -m pip install --no-cache-dir "accelerate>=0.26.0"
else
        echo "[2/7] Setting up Python venv at $VENV_DIR..."
        bash "$REPO_DIR/scripts/setup_cluster_venv.sh" "$VENV_DIR"
        source "$VENV_DIR/bin/activate"
fi

# spaCy model required for L-family pooling strategies
python -m spacy download en_core_web_sm --quiet

echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")')"

# ── 3. Environment variables ─────────────────────────────────
echo "[3/7] Setting environment variables..."
export PYTHONPATH="$REPO_DIR"
export HF_TOKEN="$HF_TOKEN"
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
export POOLBENCH_RESULTS_DIR="$REPO_DIR/results"
export HF_HUB_ENABLE_HF_TRANSFER=1         # faster HF downloads if hf_transfer is installed
pip install --quiet hf_transfer 2>/dev/null || true

# ── 4. Disk / VRAM check ─────────────────────────────────────
echo "[4/7] Pre-flight checks..."
DISK_FREE_GB=$(df -BG "$REPO_DIR" | awk 'NR==2{gsub("G",""); print $4}')
echo "  Free disk: ${DISK_FREE_GB} GB  (need ~50 GB for mistral activations)"
if (( DISK_FREE_GB < 50 )); then
    echo "  WARNING: less than 50 GB free — activations may not fit"
fi

python - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    print("ERROR: no CUDA GPU found"); sys.exit(1)
free_gb = torch.cuda.mem_get_info(0)[0] / 1e9
total_gb = torch.cuda.mem_get_info(0)[1] / 1e9
print(f"  VRAM: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
if free_gb < 18:
    print("  WARNING: <18 GB free VRAM — consider killing other GPU processes")
PY

# ── 5. Verify concepts load cleanly ──────────────────────────
echo "[5/7] Verifying corpus..."
python - <<'PY'
from pathlib import Path
from poolbench.data.concepts import CONCEPT_NAMES
corpus_root = Path("data/corpora")
present = [p for p in corpus_root.glob("*/train_pos.jsonl")]
if len(present) < len(CONCEPT_NAMES):
    raise SystemExit(
        "Corpus missing or incomplete under data/corpora.\n"
        "Download it with:\n"
        "  huggingface-cli download poolbench-anon/poolbench --repo-type dataset --include 'data/corpora/**' --local-dir .\n"
        "or rebuild it with:\n"
        "  python scripts/dataset_builder.py --all"
    )
print(f"  Concepts loaded: {len(CONCEPT_NAMES)}")
assert len(CONCEPT_NAMES) == 17, f"Expected 17 concepts, got {len(CONCEPT_NAMES)}"
print("  OK")
PY

# ── 6. Create results dirs ────────────────────────────────────
echo "[6/7] Creating output directories..."
mkdir -p "$LOG_DIR"
mkdir -p "$REPO_DIR/results/activations"
mkdir -p "$REPO_DIR/results/auroc"
mkdir -p "$REPO_DIR/results/linearity"
mkdir -p "$REPO_DIR/results/scp"
mkdir -p "$REPO_DIR/results/disentanglement"
mkdir -p "$REPO_DIR/results/classifiers"

# ── 7. Run mistral_7b — full pipeline (all 7 steps) ──────────
echo "[7/7] Starting mistral_7b full pipeline — logging to $LOG_FILE"
echo "  Steps: extraction → linearity → AUROC sweep → ICC → Classifier B → SCP → D3"
echo "  Estimated wall-clock on A100 40GB: ~8–11 hours"
echo ""

python "$REPO_DIR/scripts/run_model.py" \
    --model    mistral_7b \
    --device   auto \
    --min_free_gb 18 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo " mistral_7b run complete. Results in: $REPO_DIR/results/"
echo " Log: $LOG_FILE"
echo "============================================================"
