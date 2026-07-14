#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-chau-2024-exact}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
elif command -v mamba >/dev/null 2>&1; then
  eval "$(mamba shell hook --shell bash)"
  mamba activate "$ENV_NAME"
else
  echo "ERROR: conda or mamba is required to activate ENV_NAME=$ENV_NAME." >&2
  exit 1
fi
