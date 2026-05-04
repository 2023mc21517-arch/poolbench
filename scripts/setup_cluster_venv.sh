#!/usr/bin/env bash
set -euo pipefail

# Create an isolated PoolBench environment so cluster/package changes do not
# affect other scripts that use the base conda environment.
VENV_DIR="${1:-$HOME/venvs/poolbench}"

python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Pin the scientific stack first to avoid mixed NumPy/SciPy binary installs.
python -m pip install --no-cache-dir --force-reinstall \
  "numpy==1.26.4" \
  "scipy==1.12.0" \
  "scikit-learn==1.4.2"

python -m pip install --no-cache-dir -e .
python -m pip install --no-cache-dir "accelerate>=0.26.0"

python - <<'PY'
import numpy, scipy, sklearn, accelerate, transformers
print("numpy", numpy.__version__, numpy.__file__)
print("scipy", scipy.__version__, scipy.__file__)
print("sklearn", sklearn.__version__, sklearn.__file__)
print("accelerate", accelerate.__version__)
print("transformers", transformers.__version__)
print("poolbench_venv_ok")
PY

echo "Activate with: source $VENV_DIR/bin/activate"
