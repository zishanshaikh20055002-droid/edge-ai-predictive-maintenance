#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d ".venv-wsl-gpu" ]]; then
  echo "Missing .venv-wsl-gpu. Create it first in WSL."
  exit 1
fi

source .venv-wsl-gpu/bin/activate

LIB_PATHS="$(find "$PWD/.venv-wsl-gpu/lib" -type d -path "*/site-packages/nvidia/*/lib" | paste -sd: -)"
if [[ -z "$LIB_PATHS" ]]; then
  echo "No CUDA library folders found under .venv-wsl-gpu/site-packages/nvidia."
  exit 1
fi

export LD_LIBRARY_PATH="$LIB_PATHS:${LD_LIBRARY_PATH:-}"

exec python -m src.evaluate_multimodal_mtl "$@"
