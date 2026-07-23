#!/usr/bin/env bash
#SBATCH --job-name=origin-perturb
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=1500G

set -euo pipefail

ENV_NAME="${ENV_NAME:-chau-2024-exact}"
N_SPACE_X="${N_SPACE_X:-100}"
N_SPACE_Y="${N_SPACE_Y:-100}"
HEATMAP_EXTENT_X="${HEATMAP_EXTENT_X:-50.0}"
HEATMAP_EXTENT_Y="${HEATMAP_EXTENT_Y:-50.0}"
PSI_N_RADIAL="${PSI_N_RADIAL:-$N_SPACE_X}"
PSI_N_ANGLE="${PSI_N_ANGLE:-$N_SPACE_Y}"
N_ORI="${N_ORI:-8}"
SPACE_EXTENT="${SPACE_EXTENT:-400.0}"
FIT_INDEX="${FIT_INDEX:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-origin_horizontal_perturbation}"
PSI_EXPERIMENT_NAME="${PSI_EXPERIMENT_NAME:-${EXPERIMENT_NAME}_polar_psi}"
DH="${DH:-10000.0}"
DH_VALUES="${DH_VALUES:--10000.0 -5000.0 0.0 5000.0 10000.0}"
SEED="${SEED:-0}"
MAX_NEURONS="${MAX_NEURONS:-200000}"
RESPONSE_MODE="${RESPONSE_MODE:-matrix_approx}"
APPROX_ORDER="${APPROX_ORDER:-8}"
RUN_CARTESIAN="${RUN_CARTESIAN:-1}"
RUN_POLAR_PSI="${RUN_POLAR_PSI:-0}"
DH_SIGN="$(awk -v dh="$DH" 'BEGIN { if (dh > 0) print "positive"; else if (dh < 0) print "negative"; else print "zero" }')"
read -r -a DH_VALUES_ARRAY <<< "$DH_VALUES"
CARTESIAN_SCRIPT="paper/plot_origin_perturbation_celltype_comparison.py"
POLAR_SCRIPT="paper/plot_origin_perturbation_polar_celltype_comparison.py"

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
echo "CARTESIAN_SCRIPT=$CARTESIAN_SCRIPT"
echo "POLAR_SCRIPT=$POLAR_SCRIPT"
echo "N_SPACE_X=$N_SPACE_X"
echo "N_SPACE_Y=$N_SPACE_Y"
echo "HEATMAP_EXTENT_X=$HEATMAP_EXTENT_X"
echo "HEATMAP_EXTENT_Y=$HEATMAP_EXTENT_Y"
echo "PSI_N_RADIAL=$PSI_N_RADIAL"
echo "PSI_N_ANGLE=$PSI_N_ANGLE"
echo "N_ORI=$N_ORI"
echo "SPACE_EXTENT=$SPACE_EXTENT"
echo "FIT_INDEX=$FIT_INDEX"
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"
echo "PSI_EXPERIMENT_NAME=$PSI_EXPERIMENT_NAME"
echo "DH=$DH"
echo "DH_SIGN=$DH_SIGN"
echo "DH_VALUES=$DH_VALUES"
echo "SEED=$SEED"
echo "MAX_NEURONS=$MAX_NEURONS"
echo "RESPONSE_MODE=$RESPONSE_MODE"
echo "APPROX_ORDER=$APPROX_ORDER"
echo "RUN_CARTESIAN=$RUN_CARTESIAN"
echo "RUN_POLAR_PSI=$RUN_POLAR_PSI"
echo "Python: $(command -v python)"

python - <<'PY'
import importlib.util
import sys

missing = [
    name
    for name in ("hyclib", "matplotlib", "tdfl", "torch")
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

if [[ "$RUN_CARTESIAN" == "1" ]]; then
  cartesian_cmd=(
    python "$CARTESIAN_SCRIPT"
    --N-space "$N_SPACE_X" "$N_SPACE_Y"
    --heatmap-extent "$HEATMAP_EXTENT_X" "$HEATMAP_EXTENT_Y"
    --N-ori "$N_ORI"
    --space-extent "$SPACE_EXTENT"
    --fit-index "$FIT_INDEX"
    --experiment-name "$EXPERIMENT_NAME"
    --dh "$DH"
    --dh-values "${DH_VALUES_ARRAY[@]}"
    --seed "$SEED"
    --max-neurons "$MAX_NEURONS"
    --response-mode "$RESPONSE_MODE"
    --approx-order "$APPROX_ORDER"
    --skip-psi
  )
  "${cartesian_cmd[@]}"
fi

if [[ "$RUN_POLAR_PSI" == "1" ]]; then
  polar_cmd=(
    python "$POLAR_SCRIPT"
    --N-space "$PSI_N_RADIAL" "$PSI_N_ANGLE"
    --N-ori "$N_ORI"
    --space-extent "$SPACE_EXTENT"
    --fit-index "$FIT_INDEX"
    --experiment-name "$PSI_EXPERIMENT_NAME"
    --dh "$DH"
    --seed "$SEED"
    --max-neurons "$MAX_NEURONS"
    --response-mode "$RESPONSE_MODE"
    --approx-order "$APPROX_ORDER"
    --only-psi
  )
  "${polar_cmd[@]}"
fi
