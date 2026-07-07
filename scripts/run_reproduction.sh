#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

MODE="${1:-core}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/reproduction_${MODE}_$(date +%Y%m%d_%H%M%S).log"

run_logged() {
  echo "+ $*" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
}

echo "Logging to $LOG_FILE"
python - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import os, platform, torch
print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_version={torch.version.cuda}")
print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

case "$MODE" in
  smoke)
    run_logged pytest -q
    run_logged python paper/fit_kernel.py paper/product_data/EI.csv paper/product_data/IE.csv -o paper/figures/1b.pdf
    run_logged python paper/contourplot_space.py E -o paper/figures/2a.pdf
    ;;
  core)
    run_logged python paper/fit_kernel.py paper/product_data/EI.csv paper/product_data/IE.csv -o paper/figures/1b.pdf
    run_logged python paper/response.py --wee 3 --wei 4 --wie 4 --wii 5.25 --kee 0.5 --kei -0.25 --kie -0.25 --kii 0.25 --N-space 100 100 --N-ori 12 --N-osi 7 -m eigvals --eps 1e-2 -o paper/figures/1c.pdf
    run_logged python paper/contourplot_space.py E -o paper/figures/2a.pdf
    run_logged python paper/contourplot_space.py E --rho 1 2 -o paper/figures/2b.pdf
    run_logged python paper/contourplot_space.py r0 -y 0 15 --rho 0.72 --w00 5 -l -1 1 -n 9 --shade 0.443 0.691 -o paper/figures/2d.pdf
    run_logged python paper/contourplot_ori.py -o paper/figures/4b.pdf
    run_logged python paper/contourplot_space.py dr0dg -y 0 15 --rho 0.72 --w00 5 -l -0.25 1.25 -n 7 -s neghalflog -t 2 -o paper/figures/5c.pdf
    ;;
  paper)
    run_logged bash -lc 'python paper/fit_kernel.py paper/product_data/EI.csv paper/product_data/IE.csv -o paper/figures/1b.pdf'
    run_logged bash -lc 'python paper/response.py --wee 3 --wei 4 --wie 4 --wii 5.25 --kee 0.5 --kei -0.25 --kie -0.25 --kii 0.25 --N-space 100 100 --N-ori 12 --N-osi 7 -m eigvals --eps 1e-2 -o paper/figures/1c.pdf'
    run_logged bash -lc 'python paper/response.py --wee 3 --wei 4 --wie 4 --wii 5.25 --kee 0.5 --kei -0.25 --kie -0.25 --kii 0.25 --N-space 100 100 --N-ori 12 --N-osi 7 -m compare -k space_ori --dh 10000 -o paper/figures/1d_a.pdf'
    run_logged bash -lc 'python paper/response.py --wee 3 --wei 4 --wie 4 --wii 5.25 --kee 0.5 --kei -0.25 --kie -0.25 --kii 0.25 --N-space 100 100 --N-ori 12 --N-osi 7 -m compare -k ori_osi --dh 10000 --tau-i 1.0 -o paper/figures/1d_b.pdf'
    for cmd in \
      "python paper/contourplot_space.py E -o paper/figures/2a.pdf" \
      "python paper/contourplot_space.py E --rho 1 2 -o paper/figures/2b.pdf" \
      "python paper/contourplot_space.py r0 -y 0 15 --rho 0.72 --w00 5 -l -1 1 -n 9 --shade 0.443 0.691 -o paper/figures/2d.pdf" \
      "python paper/contourplot_space.py rmin -y 0 15 --rho 0.72 --w00 5 -s linear -l 0 2 -n 9 --shade 0.260 0.530 -o paper/figures/2e.pdf" \
      "python paper/contourplot_space.py dr1dw11 -y 0 15 --rho 0.72 --w00 5 -s halflog -l -1.25 2 -n 14 -t 0.5 -o paper/figures/2f.pdf" \
      "python paper/contourplot_space.py decay -x -5 2.5 --rho 0.72 -o paper/figures/2g.pdf" \
      "python paper/contourplot_space.py EI -y 0 15 --rho 0.72 -o paper/figures/3a.pdf" \
      "python paper/contourplot_space.py rEI -y 0 15 --rho 0.72 --w00 5 -l -0.5 1.25 -s halflog -n 8 -t 2 -o paper/figures/3b.pdf" \
      "python paper/contourplot_ori.py -o paper/figures/4b.pdf" \
      "python paper/contourplot_space.py E -x -2 2 -y -2 2 --rho 0.72 --w00 0.2 --ori -o paper/figures/4d.pdf" \
      "python paper/contourplot_space.py dr0dg -y 0 15 --rho 0.72 --w00 5 -l -0.25 1.25 -n 7 -s neghalflog -t 2 -o paper/figures/5c.pdf"; do
      run_logged bash -lc "$cmd"
    done
    ;;
  psi-space-ori-smoke)
    run_logged python paper/response.py \
      --mode space_ori \
      --wee 1.5 1.5 \
      --wei 3 3 \
      --wie 3 3 \
      --wii 5 5 \
      --kee 0.15 0.15 \
      --kei 0.5 0.3 \
      --kie 0.4 0.15 \
      --kii 0.5 0.5 \
      --N-space "${PSI_N_SPACE_X:-8}" "${PSI_N_SPACE_Y:-8}" \
      --N-ori "${PSI_N_ORI:-6}" \
      --use-psi \
      --seed "${SEED:-0}" \
      --dh "${PSI_DH:-10000}" \
      --out "${PSI_OUT:-paper/figures/psi_no_visual_field_smoke.pdf}"
    ;;
  psi-space-ori)
    run_logged python paper/response.py \
      --mode space_ori \
      --wee 1.5 1.5 \
      --wei 3 3 \
      --wie 3 3 \
      --wii 5 5 \
      --kee 0.15 0.15 \
      --kei 0.5 0.3 \
      --kie 0.4 0.15 \
      --kii 0.5 0.5 \
      --N-space "${PSI_N_SPACE_X:-40}" "${PSI_N_SPACE_Y:-40}" \
      --N-ori "${PSI_N_ORI:-12}" \
      --use-psi \
      --seed "${SEED:-0}" \
      --max-neurons "${PSI_MAX_NEURONS:-40000}" \
      --dh "${PSI_DH:-10000}" \
      --out "${PSI_OUT:-paper/figures/psi_no_visual_field.pdf}"
    ;;
  psi-paper-response)
    PSI_PAPER_OUT_DIR="${PSI_PAPER_OUT_DIR:-paper/figures/psi_no_visual_field}"
    mkdir -p "$PSI_PAPER_OUT_DIR"
    run_logged python paper/response.py \
      --wee 3 \
      --wei 4 \
      --wie 4 \
      --wii 5.25 \
      --kee 0.5 \
      --kei -0.25 \
      --kie -0.25 \
      --kii 0.25 \
      --N-space "${PSI_PAPER_N_SPACE_X:-4}" "${PSI_PAPER_N_SPACE_Y:-4}" \
      --N-ori "${PSI_PAPER_N_ORI:-12}" \
      --N-osi "${PSI_PAPER_N_OSI:-7}" \
      --mode eigvals \
      --eps 1e-2 \
      --use-psi \
      --seed "${SEED:-0}" \
      --max-neurons "${PSI_MAX_NEURONS:-6000}" \
      --out "$PSI_PAPER_OUT_DIR/1c_psi.pdf"
    run_logged python paper/response.py \
      --wee 3 \
      --wei 4 \
      --wie 4 \
      --wii 5.25 \
      --kee 0.5 \
      --kei -0.25 \
      --kie -0.25 \
      --kii 0.25 \
      --N-space "${PSI_PAPER_N_SPACE_X:-4}" "${PSI_PAPER_N_SPACE_Y:-4}" \
      --N-ori "${PSI_PAPER_N_ORI:-12}" \
      --N-osi "${PSI_PAPER_N_OSI:-7}" \
      --mode compare \
      --kind space_ori \
      --dh "${PSI_DH:-10000}" \
      --use-psi \
      --seed "${SEED:-0}" \
      --max-neurons "${PSI_MAX_NEURONS:-6000}" \
      --out "$PSI_PAPER_OUT_DIR/1d_a_psi.pdf"
    run_logged python paper/response.py \
      --wee 3 \
      --wei 4 \
      --wie 4 \
      --wii 5.25 \
      --kee 0.5 \
      --kei -0.25 \
      --kie -0.25 \
      --kii 0.25 \
      --N-space "${PSI_PAPER_N_SPACE_X:-4}" "${PSI_PAPER_N_SPACE_Y:-4}" \
      --N-ori "${PSI_PAPER_N_ORI:-12}" \
      --N-osi "${PSI_PAPER_N_OSI:-7}" \
      --mode compare \
      --kind ori_osi \
      --dh "${PSI_DH:-10000}" \
      --tau-i 1.0 \
      --use-psi \
      --seed "${SEED:-0}" \
      --max-neurons "${PSI_MAX_NEURONS:-6000}" \
      --out "$PSI_PAPER_OUT_DIR/1d_b_psi.pdf"
    run_logged python paper/response.py \
      --wee 1 \
      --wei 4 \
      --wie 4 \
      --wii 0 \
      --kee 0.5 0.5 \
      --kei 0.25 0.5 \
      --kie 0.25 0.5 \
      --kii 0 0 \
      --N-space "${PSI_PAPER_N_SPACE_X:-4}" "${PSI_PAPER_N_SPACE_Y:-4}" \
      --N-ori "${PSI_PAPER_N_ORI:-12}" \
      --N-osi "${PSI_PAPER_N_OSI:-7}" \
      --mode ori \
      --dh "${PSI_DH:-10000}" \
      --use-psi \
      --seed "${SEED:-0}" \
      --max-neurons "${PSI_MAX_NEURONS:-6000}" \
      --out "$PSI_PAPER_OUT_DIR/4a_psi.pdf"
    run_logged python paper/response.py \
      --wee 1.5 1.5 \
      --wei 3 3 \
      --wie 3 3 \
      --wii 5 5 \
      --kee 0.15 0.15 \
      --kei 0.5 0.3 \
      --kie 0.4 0.15 \
      --kii 0.5 0.5 \
      --N-space "${PSI_PAPER_N_SPACE_X:-4}" "${PSI_PAPER_N_SPACE_Y:-4}" \
      --N-ori "${PSI_PAPER_N_ORI:-12}" \
      --N-osi "${PSI_PAPER_N_OSI:-7}" \
      --mode space_ori \
      --dh "${PSI_DH:-10000}" \
      --normalize \
      --rlim 35 300 \
      --use-psi \
      --seed "${SEED:-0}" \
      --max-neurons "${PSI_MAX_NEURONS:-6000}" \
      --out "$PSI_PAPER_OUT_DIR/4c_psi.pdf"
    ;;
  psi-tests)
    run_logged python -m pytest -q test/test_nn/test_modules/test_kernels.py -k "psi_tuning"
    ;;
  model-plots)
    (
      cd paper/model_fits/no_disorder
      run_logged bash -lc 'INDICES=0-49 niarb plot plot/pairplot.toml -o figures/dist_0-49 --linfo --progress'
      run_logged bash -lc 'INDICES=0-49 niarb plot plot/resp.toml -o figures/resp_0-49 --linfo --progress'
      run_logged bash -lc 'INDICES=0-49 niarb plot plot/compare_EI_space.toml -o figures --linfo --progress'
      run_logged bash -lc 'GAIN=0.5 INDICES=0-49 niarb plot plot/compare_gain_space_ct.toml -o figures --linfo --progress'
      run_logged bash -lc 'GAIN=0.5 INDICES=0-49 niarb plot plot/compare_gain_ori_ct.toml -o figures --linfo --progress'
      run_logged bash -lc 'INDICES=0-49 niarb plot plot/weights_eigvals.toml -o figures --linfo --progress'
    )
    (
      cd paper/model_fits/disordered
      run_logged bash -lc 'INDICES=0-49 niarb plot plot/weights_ori.toml -o figures --linfo --progress'
      run_logged bash -lc 'INDICES=0-49 niarb plot plot/weights_space_strength.toml -o figures --linfo --progress'
      run_logged bash -lc 'INDICES=0-49 niarb plot plot/weights_ori_strength.toml -o figures --linfo --progress'
      run_logged bash -lc 'INDICES=0-49 niarb plot plot/resp.toml -o figures/resp_0-49 --linfo --progress'
      run_logged bash -lc 'INDICES=0-49 niarb plot plot/compare_EI_space.toml -o figures --linfo --progress'
      run_logged bash -lc 'GAIN=0.5 INDICES=0-49 niarb plot plot/compare_gain_space_ct.toml -o figures --linfo --progress'
      run_logged bash -lc 'GAIN=0.5 INDICES=0-49 niarb plot plot/compare_gain_ori_ct.toml -o figures --linfo --progress'
    )
    ;;
  verify)
    run_logged test -s paper/figures/1b.pdf
    run_logged test -s paper/figures/2a.pdf
    run_logged test -s paper/figures/4b.pdf
    run_logged test -s paper/model_fits/no_disorder/figures/compare_EI_space_0-49.pdf
    run_logged test -s paper/model_fits/disordered/figures/connection_probability_ori_0-49.pdf
    run_logged python - <<'PY'
from pathlib import Path
paths = [
    "paper/figures/1b.pdf",
    "paper/figures/2a.pdf",
    "paper/figures/4b.pdf",
    "paper/model_fits/no_disorder/figures/compare_EI_space_0-49.pdf",
    "paper/model_fits/disordered/figures/connection_probability_ori_0-49.pdf",
]
for path in paths:
    p = Path(path)
    print(f"{path}\t{p.stat().st_size} bytes")
PY
    ;;
  *)
    echo "Usage: $0 {smoke|core|paper|psi-space-ori-smoke|psi-space-ori|psi-paper-response|psi-tests|model-plots|verify}" >&2
    exit 2
    ;;
esac

echo "Completed MODE=$MODE"
