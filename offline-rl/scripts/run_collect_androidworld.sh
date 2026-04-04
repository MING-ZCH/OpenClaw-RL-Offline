#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

python "${SCRIPT_DIR}/collect_from_benchmark.py" \
  --env androidworld \
  --output "${OUTPUT:-data/androidworld_trajs.jsonl}" \
  --n "${N:-100}" \
  --success-rate "${SUCCESS_RATE:-0.3}" \
  --seed "${SEED:-42}"