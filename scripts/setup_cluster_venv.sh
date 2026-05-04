#!/usr/bin/env bash
set -euo pipefail

# Create an isolated PoolBench environment so cluster/package changes do not
# affect other scripts that use the base conda environment.
VENV_DIR="${1:-$HOME/venvs/poolbench}"

python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# The cluster reports CUDA driver API 12.2. Use a CUDA 12.1 PyTorch wheel;
# newer default wheels may require a newer NVIDIA driver and make CUDA appear
# unavailable inside the venv.
python -m pip install --no-cache-dir --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu121 \
  "torch==2.5.1"

# Pin the scientific stack first to avoid mixed NumPy/SciPy binary installs.
python -m pip install --no-cache-dir --force-reinstall \
  "numpy==1.26.4" \
  "scipy==1.12.0" \
  "scikit-learn==1.4.2"

python -m pip install --no-cache-dir -e .
python -m pip install --no-cache-dir "accelerate>=0.26.0"

python - <<'PY'
import numpy, scipy, sklearn, accelerate, transformers, torch
print("numpy", numpy.__version__, numpy.__file__)
print("scipy", scipy.__version__, scipy.__file__)
print("sklearn", sklearn.__version__, sklearn.__file__)
print("accelerate", accelerate.__version__)
print("transformers", transformers.__version__)
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if not torch.cuda.is_available():
  raise SystemExit("torch CUDA is not available; install a PyTorch wheel compatible with the cluster driver")
print("poolbench_venv_ok")
PY

echo "Activate with: source $VENV_DIR/bin/activate"
