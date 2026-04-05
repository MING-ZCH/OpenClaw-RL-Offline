#!/bin/bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

ALGO="${OFFLINE_TRAIN_ALGO:-iql}"
DATA_PATH="${OFFLINE_TRAIN_DATA:-data/osworld_trajs.jsonl}"
STEPS="${OFFLINE_TRAIN_STEPS:-500}"
BATCH_SIZE="${OFFLINE_TRAIN_BATCH_SIZE:-32}"
DEVICE="${OFFLINE_TRAIN_DEVICE:-cuda}"
LEARNING_RATE="${OFFLINE_TRAIN_LR:-3e-4}"
STATE_DIM="${OFFLINE_TRAIN_STATE_DIM:-128}"
ACTION_DIM="${OFFLINE_TRAIN_ACTION_DIM:-128}"
HIDDEN_DIM="${OFFLINE_TRAIN_HIDDEN_DIM:-128}"

ARGS=(
  "${SCRIPT_DIR}/train_offline.py"
  --data "${DATA_PATH}"
  --algo "${ALGO}"
  --steps "${STEPS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LEARNING_RATE}"
  --state-dim "${STATE_DIM}"
  --action-dim "${ACTION_DIM}"
  --hidden-dim "${HIDDEN_DIM}"
  --device "${DEVICE}"
)

if [[ -n "${OFFLINE_TRAIN_OUTPUT:-}" ]]; then
  ARGS+=(--output "${OFFLINE_TRAIN_OUTPUT}")
fi

exec "${PYTHON_BIN}" "${ARGS[@]}" "$@"