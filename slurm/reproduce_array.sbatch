#!/usr/bin/env bash
#SBATCH --job-name=chau-array
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-49
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

ENV_NAME="${ENV_NAME:-chau-2024-exact}"
WORKFLOW="${WORKFLOW:-no_disorder_run_gain1}"

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
echo "Workflow: $WORKFLOW"
echo "Task: ${SLURM_ARRAY_TASK_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

case "$WORKFLOW" in
  fit)
    cd paper/model_fits/no_disorder
    niarb fit fit.toml -o fits --linfo --ignore-errors
    ;;
  no_disorder_run_gain1)
    cd paper/model_fits/no_disorder
    GAIN=1 niarb run run.toml --linfo
    ;;
  no_disorder_run_gain05)
    cd paper/model_fits/no_disorder
    GAIN=0.5 niarb run run.toml --linfo
    ;;
  no_disorder_run_space_gain05)
    cd paper/model_fits/no_disorder
    GAIN=0.5 niarb run run_space_gain.toml --linfo
    ;;
  no_disorder_run_ori_gain05)
    cd paper/model_fits/no_disorder
    GAIN=0.5 niarb run run_ori_gain.toml --linfo
    ;;
  no_disorder_run_space_ori_gain05)
    cd paper/model_fits/no_disorder
    GAIN=0.5 niarb run run_space_ori_gain.toml --linfo
    ;;
  disordered_run_gain1)
    cd paper/model_fits/disordered
    GAIN=1 niarb run run.toml --linfo
    ;;
  disordered_run_gain05)
    cd paper/model_fits/disordered
    GAIN=0.5 niarb run run.toml --linfo
    ;;
  disordered_weights_ori)
    cd paper/model_fits/disordered
    niarb run weights_ori.toml --linfo
    ;;
  disordered_weights_space_strength)
    cd paper/model_fits/disordered
    niarb run weights_space_strength.toml --linfo
    ;;
  disordered_weights_ori_strength)
    cd paper/model_fits/disordered
    niarb run weights_ori_strength.toml --linfo
    ;;
  no_disorder_weights_eigvals)
    cd paper/model_fits/no_disorder
    niarb run weights_eigvals.toml --linfo
    ;;
  *)
    echo "Unknown WORKFLOW=$WORKFLOW" >&2
    exit 2
    ;;
esac
