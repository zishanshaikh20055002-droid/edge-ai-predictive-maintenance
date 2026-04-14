#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d ".venv-wsl-gpu" ]]; then
  echo "Missing .venv-wsl-gpu. Create it first in WSL."
  exit 1
fi

source .venv-wsl-gpu/bin/activate

exec python -m src.export_savedmodel_multimodal "$@"
