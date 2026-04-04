#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

export OFFLINE_TRAJECTORY_STORE="${OFFLINE_TRAJECTORY_STORE:-${REPO_ROOT}/offline-rl/data/osworld_trajs.jsonl}"

bash "${SCRIPT_DIR}/run_qwen35_4b_offline_rl.sh" "$@"