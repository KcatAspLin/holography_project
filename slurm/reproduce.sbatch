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

if command -v mamba >/dev/null 2>&1; then
  eval "$(mamba shell.bash hook)"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: conda or mamba is required." >&2
  exit 1
fi

conda activate "$ENV_NAME"

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
nvidia-smi || true

bash scripts/run_reproduction.sh "$MODE"
