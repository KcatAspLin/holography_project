#!/usr/bin/env bash
#SBATCH --job-name=origin-perturb
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

ENV_NAME="${ENV_NAME:-chau-2024-exact}"
N_SPACE_X="${N_SPACE_X:-32}"
N_SPACE_Y="${N_SPACE_Y:-32}"
N_ORI="${N_ORI:-8}"
FIT_INDEX="${FIT_INDEX:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-origin_horizontal_perturbation}"
SEED="${SEED:-0}"
MAX_NEURONS="${MAX_NEURONS:-60000}"

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
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/mplconfig-${USER:-user}}"
mkdir -p "$MPLCONFIGDIR"

echo "Host: $(hostname)"
echo "Job: ${SLURM_JOB_ID:-none}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "ENV_NAME=$ENV_NAME"
echo "N_SPACE_X=$N_SPACE_X"
echo "N_SPACE_Y=$N_SPACE_Y"
echo "N_ORI=$N_ORI"
echo "FIT_INDEX=$FIT_INDEX"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "SEED=$SEED"
echo "MAX_NEURONS=$MAX_NEURONS"
echo "Python: $(command -v python)"

python - <<'PY'
import importlib.util
import sys

missing = [
    name
    for name in ("hyclib", "matplotlib", "torch")
    if importlib.util.find_spec(name) is None
]
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

cmd=(
  python paper/plot_origin_perturbation_celltype_comparison.py
  --N-space "$N_SPACE_X" "$N_SPACE_Y" \
  --N-ori "$N_ORI" \
  --fit-index "$FIT_INDEX" \
  --experiment-name "$EXPERIMENT_NAME" \
  --seed "$SEED" \
  --max-neurons "$MAX_NEURONS"
)

"${cmd[@]}"
