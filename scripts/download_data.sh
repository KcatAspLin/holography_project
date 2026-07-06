#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

NO_DISORDER_ID="1Y5Qlz97joDoEscmQTgLyZS7CCAv7FZAt"
DISORDERED_ID="1NWVixT5ou9ukNxqmdzHHfH8QbAXZ9hsA"

download_gdrive() {
  local file_id="$1"
  local output="$2"

  if command -v gdown >/dev/null 2>&1; then
    gdown --fuzzy "https://drive.google.com/file/d/${file_id}/view?usp=sharing" -O "$output"
  else
    python -m pip install --user gdown
    python -m gdown --fuzzy "https://drive.google.com/file/d/${file_id}/view?usp=sharing" -O "$output"
  fi
}

fetch_runs() {
  local label="$1"
  local file_id="$2"
  local target_dir="$3"
  local zip_path="$target_dir/runs.zip"

  mkdir -p "$target_dir"
  if [ -d "$target_dir/runs" ]; then
    echo "$label runs already exist at $target_dir/runs"
    return
  fi

  echo "Downloading $label runs.zip from Google Drive..."
  download_gdrive "$file_id" "$zip_path"
  unzip -q "$zip_path" -d "$target_dir"
  rm -f "$zip_path"
}

fetch_runs "no_disorder" "$NO_DISORDER_ID" "paper/model_fits/no_disorder"
fetch_runs "disordered" "$DISORDERED_ID" "paper/model_fits/disordered"

if [ ! -e paper/model_fits/disordered/data ]; then
  ln -s ../no_disorder/data paper/model_fits/disordered/data
fi
if [ ! -e paper/model_fits/disordered/fits ]; then
  ln -s ../no_disorder/fits paper/model_fits/disordered/fits
fi

echo "Data check:"
find paper/model_fits -maxdepth 3 -type d \( -name data -o -name fits -o -name runs \) -print
