#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-chau-2024-exact}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_BACKEND="${TORCH_BACKEND:-default}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v mamba >/dev/null 2>&1; then
  CONDA_EXE=mamba
elif command -v conda >/dev/null 2>&1; then
  CONDA_EXE=conda
else
  echo "ERROR: conda or mamba is required." >&2
  exit 1
fi

eval "$("$CONDA_EXE" shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  "$CONDA_EXE" create -y -n "$ENV_NAME" -c conda-forge \
    "python=${PYTHON_VERSION}" pip "setuptools>=64" wheel git curl unzip compilers make
fi

conda activate "$ENV_NAME"
python -m pip install --upgrade pip

case "$TORCH_BACKEND" in
  cpu)
    python -m pip install --index-url https://download.pytorch.org/whl/cpu \
      "torch==2.5.1"
    ;;
  cu124|cuda|gpu)
    python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
      "torch==2.5.1"
    ;;
  default)
    ;;
  *)
    echo "ERROR: TORCH_BACKEND must be one of: default, cpu, cu124." >&2
    exit 1
    ;;
esac

python -m pip install -r requirements.txt

python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_version={torch.version.cuda}")
print(f"device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"device_name={torch.cuda.get_device_name(0)}")
PY

echo "Environment '$ENV_NAME' is ready."
