#!/bin/bash
# Qwen3.5-4B offline RL fine-tuning with pre-collected trajectories
# Uses offline replay instead of live environment interaction
#
# Steps:
# 1. Collect trajectories: python offline-rl/scripts/collect_offline_data.py --env osworld --episodes 500
# 2. (Optional) Pre-compute weights: python openclaw-offline/compute_weights.py --algo iql --data trajectories.jsonl
# 3. Run this script

SKIP_CLUSTER_CLEANUP=${SKIP_CLUSTER_CLEANUP:-0}
if [ "${SKIP_CLUSTER_CLEANUP}" != "1" ]; then
  pkill -9 sglang
  sleep 3
  ray stop --force
  pkill -9 ray
  pkill -9 python
  sleep 3
fi

set -ex

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/tmp}"

NUM_GPUS=${NUM_GPUS:-8}
ACTOR_GPUS=${ACTOR_GPUS:-4}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}
PRM_GPUS=${PRM_GPUS:-0}  # No PRM needed for offline

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../slime" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
source "${SLIME_ROOT}/scripts/models/qwen3.5-4B.sh"

HF_CKPT=${HF_CKPT:-${REPO_ROOT}/models/Qwen3.5-4B}
REF_LOAD=${REF_LOAD:-${HF_CKPT}}
SAVE_CKPT=${SAVE_CKPT:-${REPO_ROOT}/ckpt/qwen35-4b-offline-rl}

# ===== Offline RL Configuration =====
export OFFLINE_TRAJECTORY_STORE="${OFFLINE_TRAJECTORY_STORE:-${REPO_ROOT}/data/trajectories.jsonl}"
export OFFLINE_MODE="${OFFLINE_MODE:-step}"                   # step | trajectory | dynamic_history
export OFFLINE_N_SAMPLES_PER_PROMPT="${OFFLINE_N_SAMPLES_PER_PROMPT:-1}"

# Optional: pre-computed advantage weights from IQL/CQL
export OFFLINE_WEIGHT_PATH="${OFFLINE_WEIGHT_PATH:-}"         # Path to weights.json
export OFFLINE_WEIGHT_TEMPERATURE="${OFFLINE_WEIGHT_TEMPERATURE:-3.0}"
# =====================================

CKPT_ARGS=(
   --megatron-to-hf-mode bridge
   --hf-checkpoint "${HF_CKPT}"
   --ref-load "${REF_LOAD}"
   --save "${SAVE_CKPT}"
   --save-interval 100
   --rotary-base 5000000
)

ROLLOUT_ARGS=(
   # Key difference: use offline rollout instead of online
   --disable-rollout-global-dataset
   --rollout-function-path offline_rollout.generate_rollout_offline

   --num-rollout 100000000
   --rollout-batch-size 16
   --n-samples-per-prompt "${OFFLINE_N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len 8192
   --rollout-max-context-len 32768
   --reward-key score

   --num-steps-per-rollout 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 32768
   --log-probs-chunk-size 1024
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --disable-rewards-normalization
   --use-kl-loss
   --kl-loss-coef 0.02
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# Uncomment to use advantage-weighted loss from IQL/CQL instead of GRPO
# GRPO_ARGS=(
#    --loss-type custom_loss
#    --custom-loss-function-path offline_loss.advantage_weighted_loss_function
#    --disable-compute-advantages-and-returns
# )

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

EVAL_ARGS=()

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.85
   --sglang-context-length 32768
   --sglang-language-only
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --no-rope-fusion
   --bf16
   --disable-bias-linear
   --untie-embeddings-and-output-weights
)

cd "${SCRIPT_DIR}"

python -m slime.launch \
   --backend megatron \
   --num-actor-gpus "${ACTOR_GPUS}" \
   --num-rollout-gpus "${ROLLOUT_GPUS}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
