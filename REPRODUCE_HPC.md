# HPC Reproduction Guide

This repository contains code for Chau et al. 2025, "Exact linear theory of perturbation response in a space- and feature-dependent cortical circuit model". The upstream README gives a minimal Python 3.11 setup; `paper/README.md` contains the paper figure commands.

## Repository Map

- `src/niarb`: installable Python package and `niarb` CLI.
- `src/niarb/cli/{fit,run,plot}.py`: main CLI entry points.
- `paper`: scripts and processed data for paper figures.
- `paper/README.md`: command table for Figures 1-5 and supplementary figures.
- `paper/model_fits/no_disorder`: Figure 6, S6, and S8 model-fit workflow.
- `paper/model_fits/disordered`: Figure S7 model-fit workflow.
- `paper/model_fits/*/fit.toml`: fitting configuration.
- `paper/model_fits/*/run*.toml`, `weights*.toml`: array-job simulation configurations.
- `paper/model_fits/*/plot/*.toml`: plot configurations consuming `fits` and `runs`.
- `paper/*_data`: processed public/extracted data already committed to the repository.
- `paper/model_fits/no_disorder/fits`: fitted model parameter `.pt` files already committed.
- `paper/model_fits/*/figures`: generated upstream figure PDFs already committed.

## Dependencies

The project requires Python 3.11 exactly. Core dependencies are declared in `pyproject.toml`, including:

- `torch==2.5.1`
- `torchdiffeq==0.2.5`
- `hyclib==0.1.40`
- `tdfl==0.1.15`
- `ricciardi==0.1.5`
- `torch-bessel==0.0.5`
- NumPy, SciPy, pandas, seaborn, statsmodels, matplotlib, tqdm

`requirements.txt` installs the package in editable mode with test/development tools. The added `environment.yml` and `scripts/setup_env.sh` create a Conda/Mamba environment around the existing dependency metadata.

## Data and Model Artifacts

Most processed input data and fitted parameter files are already present in the repository. Two model simulation directories are not committed and must either be downloaded or regenerated:

- `paper/model_fits/no_disorder/runs`: Google Drive file ID `1Y5Qlz97joDoEscmQTgLyZS7CCAv7FZAt`
- `paper/model_fits/disordered/runs`: Google Drive file ID `1NWVixT5ou9ukNxqmdzHHfH8QbAXZ9hsA`

Use:

```bash
bash scripts/download_data.sh
```

If Google Drive access is blocked on your cluster, download the two `runs.zip` files on a machine with browser access, transfer them to the cluster, unzip each into its matching `paper/model_fits/...` directory, and ensure the resulting path is `paper/model_fits/<workflow>/runs`.

## Setup on the Login Node

Do only setup and submission from the login node:

```bash
git clone https://github.com/hchau630/chau-2024-exact.git
cd chau-2024-exact

# after applying or copying these HPC helper files into the clone:
bash scripts/setup_env.sh
```

For CPU-only PyTorch wheels:

```bash
TORCH_BACKEND=cpu bash scripts/setup_env.sh
```

For CUDA 12.4 PyTorch wheels:

```bash
TORCH_BACKEND=cu124 bash scripts/setup_env.sh
```

Check GPU/CUDA availability inside an allocation:

```bash
nvidia-smi
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name", torch.cuda.get_device_name(0))
PY
```

## Main Reproduction Commands

Run on a compute node or Slurm job, not the login node.

Smoke test:

```bash
bash scripts/run_reproduction.sh smoke
```

Core paper figures, lightweight subset:

```bash
bash scripts/run_reproduction.sh core
```

Broader direct paper figures from `paper/README.md`:

```bash
bash scripts/run_reproduction.sh paper
```

Model-fit plots, requiring downloaded or regenerated `runs`:

```bash
bash scripts/download_data.sh
bash scripts/run_reproduction.sh model-plots
```

Verify key output files exist and are non-empty:

```bash
bash scripts/run_reproduction.sh verify
```

Logs are written under `logs/`.

## Slurm: Reproduce Figures from Existing/Downloaded Artifacts

CPU or cluster-default job:

```bash
sbatch slurm/reproduce.sbatch
```

Choose a mode:

```bash
MODE=smoke sbatch slurm/reproduce.sbatch
MODE=core sbatch slurm/reproduce.sbatch
MODE=paper sbatch slurm/reproduce.sbatch
MODE=model-plots sbatch slurm/reproduce.sbatch
MODE=verify sbatch slurm/reproduce.sbatch
```

If your site requires GPU flags, submit with your cluster's syntax, for example:

```bash
sbatch --gres=gpu:1 slurm/reproduce.sbatch
sbatch --partition=gpu --gres=gpu:a40:1 slurm/reproduce.sbatch
```

The code uses CUDA automatically when `torch.cuda.is_available()` is true.

## Slurm: Regenerate Model Runs with Arrays

The upstream README reports A40 GPU commands. The added array script wraps the same workflows and keeps stdout/stderr in `logs/`.

Regenerate no-disorder response runs:

```bash
WORKFLOW=no_disorder_run_gain1 sbatch -p gpu --array=0-49 --time=00:05:00 --cpus-per-task=8 --mem-per-cpu=4G --gres=gpu:1 slurm/reproduce_array.sbatch
WORKFLOW=no_disorder_run_gain05 sbatch -p gpu --array=0-49 --time=00:05:00 --cpus-per-task=8 --mem-per-cpu=4G --gres=gpu:1 slurm/reproduce_array.sbatch
WORKFLOW=no_disorder_run_space_gain05 sbatch -p gpu --array=0-49 --time=00:05:00 --cpus-per-task=8 --mem-per-cpu=4G --gres=gpu:1 slurm/reproduce_array.sbatch
WORKFLOW=no_disorder_run_ori_gain05 sbatch -p gpu --array=0-49 --time=00:05:00 --cpus-per-task=8 --mem-per-cpu=4G --gres=gpu:1 slurm/reproduce_array.sbatch
WORKFLOW=no_disorder_run_space_ori_gain05 sbatch -p gpu --array=0-49 --time=00:05:00 --cpus-per-task=8 --mem-per-cpu=4G --gres=gpu:1 slurm/reproduce_array.sbatch
```

Regenerate disordered response/connectivity runs:

```bash
WORKFLOW=disordered_run_gain1 sbatch -p gpu --array=0-49 --time=00:05:00 --cpus-per-task=8 --mem-per-cpu=4G --gres=gpu:1 slurm/reproduce_array.sbatch
WORKFLOW=disordered_run_gain05 sbatch -p gpu --array=0-49 --time=00:05:00 --cpus-per-task=8 --mem-per-cpu=4G --gres=gpu:1 slurm/reproduce_array.sbatch
WORKFLOW=disordered_weights_ori sbatch -p gpu --array=0-49 --time=00:05:00 --cpus-per-task=8 --mem-per-cpu=4G --gres=gpu:1 slurm/reproduce_array.sbatch
WORKFLOW=disordered_weights_space_strength sbatch -p gpu --array=0-49 --time=00:05:00 --cpus-per-task=8 --mem-per-cpu=4G --gres=gpu:1 slurm/reproduce_array.sbatch
WORKFLOW=disordered_weights_ori_strength sbatch -p gpu --array=0-49 --time=00:05:00 --cpus-per-task=8 --mem-per-cpu=4G --gres=gpu:1 slurm/reproduce_array.sbatch
```

Regenerate fitted parameter files from scratch:

```bash
WORKFLOW=fit sbatch -p gpu --array=0-19 --time=01:00:00 --cpus-per-task=8 --mem-per-cpu=4G --gres=gpu:1 slurm/reproduce_array.sbatch
```

The fit workflow is stochastic in the upstream code. It does not bind `SLURM_ARRAY_TASK_ID` to a fixed seed, so exact fit filenames/losses may differ when regenerating from scratch. Prefer the committed `fits` files or the downloadable `runs` for exact figure reproduction.

## Expected Outputs

Direct paper figures:

- `paper/figures/1b.pdf`
- `paper/figures/1c.pdf`
- `paper/figures/2a.pdf`
- `paper/figures/2b.pdf`
- `paper/figures/4b.pdf`
- `paper/figures/5c.pdf`

Model-fit figures:

- `paper/model_fits/no_disorder/figures/dist_0-49/phase_diagram.pdf`
- `paper/model_fits/no_disorder/figures/resp_0-49/resp_space.pdf`
- `paper/model_fits/no_disorder/figures/compare_EI_space_0-49.pdf`
- `paper/model_fits/no_disorder/figures/compare_gain_0.5_space_ct_0-49.pdf`
- `paper/model_fits/disordered/figures/connection_probability_ori_0-49.pdf`
- `paper/model_fits/disordered/figures/resp_0-49/resp_space.pdf`

Because PDFs embed metadata and plotting libraries can differ, byte-identical PDFs are not expected across systems. Practical verification is:

```bash
bash scripts/run_reproduction.sh verify
find paper/figures paper/model_fits -name '*.pdf' -size +0 -print | sort
```

For numerical checks reported by the README:

```bash
python paper/fit_kernel.py paper/product_data/EI.csv paper/product_data/IE.csv
python paper/uncertainty_rho.py 150.1898239023003 11.31920466 107.5719331775833 8.43638973
python paper/fit_rossi_ori.py paper/rossi_data/fig2h.csv
python paper/calc_pmax.py
```

The README notes that bootstrap-derived confidence intervals can vary slightly from run to run.

## Reproduction Checklist

1. Clone the repository.
2. Copy or apply the added HPC helper files.
3. Create the Conda/Mamba environment with `scripts/setup_env.sh`.
4. Run only smoke tests on a compute node.
5. Download or regenerate `paper/model_fits/*/runs`.
6. Submit `MODE=core` or `MODE=paper` through `slurm/reproduce.sbatch`.
7. Submit `MODE=model-plots` after `runs` exists.
8. Run `MODE=verify`.
9. Compare generated PDFs and printed numeric summaries with the README/paper claims.

## Unresolved Blockers and Caveats

- The two large `runs.zip` files are hosted on Google Drive, not in the Git repository. Cluster-side download may fail if Google Drive is blocked; manual transfer is then required.
- Exact regenerated fits are not guaranteed because the upstream fit command is stochastic and does not fix seeds per Slurm task.
- GPU availability is cluster-specific. The upstream model-fit commands were run with A40 GPUs; CPU execution may work but can be slow or memory-limited.
- Some Slurm clusters use different module names or GPU request syntax. Edit only the `module load` lines and `--gres`/partition flags as needed.
