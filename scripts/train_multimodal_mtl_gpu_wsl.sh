#!/usr/bin/env bash
set -euo pipefail

PROBE_ONLY=0
if [[ "${1:-}" == "--probe-only" ]]; then
  PROBE_ONLY=1
  shift
fi

if [[ ! -d ".venv-wsl-gpu" ]]; then
  echo "Missing .venv-wsl-gpu. Create it first in WSL."
  exit 1
fi

source .venv-wsl-gpu/bin/activate

# Discover CUDA-related shared library folders installed via pip nvidia-* wheels.
LIB_PATHS="$(find "$PWD/.venv-wsl-gpu/lib" -type d -path "*/site-packages/nvidia/*/lib" | paste -sd: -)"
if [[ -z "$LIB_PATHS" ]]; then
  echo "No CUDA library folders found under .venv-wsl-gpu/site-packages/nvidia."
  exit 1
fi

export LD_LIBRARY_PATH="$LIB_PATHS:${LD_LIBRARY_PATH:-}"

python - <<'PY'
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow GPUs:", gpus)
if not gpus:
    raise SystemExit("TensorFlow cannot see a GPU in this environment.")
PY

if [[ "$PROBE_ONLY" == "1" ]]; then
  echo "GPU probe successful."
  exit 0
fi

if [[ "$#" -eq 0 ]]; then
  set -- --epochs 40 --batch-size 32 --require-gpu
elif [[ " $* " != *" --require-gpu "* ]]; then
  set -- "$@" --require-gpu
fi

exec python -m src.train_multimodal_mtl "$@"