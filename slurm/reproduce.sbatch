#!/usr/bin/env bash
#SBATCH --job-name=chau-repro
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

ENV_NAME="${ENV_NAME:-chau-2024-exact}"
MODE="${MODE:-core}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

if command -v module >/dev/null 2>&1; then
  module purge || true
  module load mamba || module load miniconda || module load anaconda || true
fi

source scripts/activate_env.sh

export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLBACKEND=Agg

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "MODE=$MODE"
echo "SEED=${SEED:-0}"
echo "PSI_N_SPACE_X=${PSI_N_SPACE_X:-unset}"
echo "PSI_N_SPACE_Y=${PSI_N_SPACE_Y:-unset}"
echo "PSI_N_ORI=${PSI_N_ORI:-unset}"
echo "PSI_MAX_NEURONS=${PSI_MAX_NEURONS:-unset}"
echo "Python: $(command -v python)"
python - <<'PY'
import importlib.util
import sys

missing = [name for name in ("pandas", "pytest", "torch") if importlib.util.find_spec(name) is None]
if missing:
    print(
        "ERROR: Active Python environment is missing required packages: "
        + ", ".join(missing),
        file=sys.stderr,
    )
    print(
        "Run this on a login node first: ENV_NAME=chau-2024-exact bash scripts/setup_env.sh",
        file=sys.stderr,
    )
    sys.exit(1)
print(f"Python executable: {sys.executable}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found; continuing without NVIDIA GPU diagnostics."
fi

bash scripts/run_reproduction.sh "$MODE"
