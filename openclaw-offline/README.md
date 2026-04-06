# openclaw-offline

`openclaw-offline` is the bridge from replayed offline trajectories to the original slime-based OpenClaw training loop.

It does not replace the upstream trainer. Instead, it swaps live rollout collection for offline replay so the existing distributed training stack can fine-tune on benchmark trajectories collected in [offline-rl](../offline-rl/README.md).

## What This Module Does

- replays offline trajectories as slime `Sample` groups;
- optionally applies advantage-weighted loss using critic-derived weights;
- keeps the original Megatron and slime launch pattern intact;
- provides benchmark-specific wrappers that only preset `OFFLINE_TRAJECTORY_STORE`, including PowerShell forwarding launchers for Windows hosts.

## Typical Workflow

### 1. Collect trajectories in offline-rl

```bash
cd ../offline-rl

python scripts/collect_from_benchmark.py --env osworld --n 100 --output data/osworld_trajs.jsonl
python scripts/collect_from_benchmark.py --env androidworld --n 100 --output data/androidworld_trajs.jsonl
python scripts/collect_from_benchmark.py --env webarena --n 100 --output data/webarena_trajs.jsonl
python scripts/collect_from_benchmark.py --env alfworld --n 100 --output data/alfworld_trajs.jsonl
```

### 2. Optionally compute advantage weights

```bash
python compute_weights.py \
  --data ../offline-rl/data/osworld_trajs.jsonl \
  --output ../offline-rl/data/osworld_iql_weights.json \
  --algo iql \
  --train-steps 500 \
  --beta 3.0
```

Supported critic algorithms: all 32 offline RL algorithms from the [offline-rl](../offline-rl/README.md) package.

Key algorithm categories:
- **True advantage (Q−V)**: `iql`, `crr`, `edac`, `oreo`, `archer`, `bcq`, `ilql` — use `get_advantages()` for IQL-style advantages.
- **Q/value proxy**: `cql`, `td3bc`, `grpo`, `awac`, `sorl`, `arpo`, `dt`, `rwft`, `retrospex`, `glider`, `kto`, `rebel`, `digiq`, `agent_q` — use `get_action_values()` as proxy.
- **Success probability**: `webrl` (ORM P(success) ∈ [0,1]), `digirl` (V_step success probability ∈ [0,1]).
- **Log-prob preference proxy**: `dpo`, `ipo`, `cpo`, `dmpo`, `eto`, `rrhf` — use `get_action_values()` returning log-probability ratio as proxy.
- **Reference-free preference proxy**: `simpo`, `orpo` — use `get_action_values()` returning β-scaled log-prob (no reference model).
- **Learned value model**: `vem` — use `get_action_values()` returning VEM-predicted value.

If `OFFLINE_WEIGHT_PATH` is not provided at launch time, `offline_loss.py` falls back to reward/outcome-based proxy weights. That path is useful for smoke tests, but critic-derived weights are the recommended route when you want a closer offline-RL signal.

Note: `compute_weights.py` now defaults to `--device cuda` to match the GPU-first convention of the training entry points. Pass `--device cpu` on CPU-only machines.

### 3. Launch offline fine-tuning

```bash
# benchmark-specific wrappers
bash run_qwen35_4b_osworld_offline_rl.sh
bash run_qwen35_4b_androidworld_offline_rl.sh
bash run_qwen35_4b_webarena_offline_rl.sh
bash run_qwen35_4b_alfworld_offline_rl.sh

# or the generic launcher after setting the trajectory path
export OFFLINE_TRAJECTORY_STORE=../offline-rl/data/osworld_trajs.jsonl
bash run_qwen35_4b_offline_rl.sh
```

```powershell
# benchmark-specific wrappers
.\run_qwen35_4b_osworld_offline_rl.ps1
.\run_qwen35_4b_androidworld_offline_rl.ps1
.\run_qwen35_4b_webarena_offline_rl.ps1
.\run_qwen35_4b_alfworld_offline_rl.ps1

# or the generic launcher after setting the trajectory path
$env:OFFLINE_TRAJECTORY_STORE = "..\offline-rl\data\osworld_trajs.jsonl"
.\run_qwen35_4b_offline_rl.ps1
```

The PowerShell entry points prefer WSL and then fall back to Git Bash. They are convenience launchers for Windows hosts; the actual distributed training stack still needs a Linux-like runtime.

## Files

| File | Role |
|---|---|
| `offline_rollout.py` | Replays trajectory data into slime rollout output |
| `offline_loss.py` | Optional advantage-weighted loss for offline policy extraction |
| `compute_weights.py` | Trains small offline critics (any of 32 algorithms) and exports advantage weights |
| `run_qwen35_4b_offline_rl.{sh,ps1}` | Generic offline training launcher |
| `run_qwen35_4b_{osworld,androidworld,webarena,alfworld}_offline_rl.{sh,ps1}` | Thin benchmark-specific wrappers |

## Key Environment Variables

| Variable | Meaning |
|---|---|
| `OFFLINE_TRAJECTORY_STORE` | Path to the trajectory JSONL file |
| `OFFLINE_MODE` | Replay mode: `step`, `trajectory`, or `dynamic_history` |
| `OFFLINE_N_SAMPLES_PER_PROMPT` | Group size used during replay |
| `OFFLINE_WEIGHT_PATH` | Optional path to advantage weights JSON |
| `OFFLINE_WEIGHT_TEMPERATURE` | Temperature used for weighting those advantages |

## When To Use This Module

Use `openclaw-offline` when:

- the trajectories are already collected;
- you want to preserve the slime and Megatron training path;
- the goal is full offline LLM fine-tuning rather than small standalone replay experiments.

If you only need replay data structures, collectors, adapters, or lightweight offline RL baselines, stay in [offline-rl](../offline-rl/README.md).

## Limitations

- This module assumes the original slime runtime stack is available.
- Full training still expects a Linux-like multi-GPU environment, even when launched from PowerShell.
- Offline replay does not add exploration; it only reuses pre-collected trajectories.
- When no explicit weight file is provided, the loss falls back to reward-based proxy weighting rather than learned advantages.
- Real benchmark quality still depends on the underlying external benchmark environments and how the data was collected.