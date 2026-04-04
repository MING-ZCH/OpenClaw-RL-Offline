# Upstream Integration Complete — Summary Log

## Task
Integrate the offline-rl extension into the upstream OpenClaw-RL / slime training framework.

## Integration Strategy

The upstream OpenClaw-RL uses slime's plugin architecture with two key hooks:
1. **`--rollout-function-path`**: Custom rollout function loaded at runtime
2. **`--custom-loss-function-path`**: Custom loss function for training

We created a standalone `openclaw-offline/` module that plugs into both hooks
without modifying any upstream code.

## Files Created

### openclaw-offline/offline_rollout.py
- **Purpose**: Replaces online environment interaction with offline data replay
- **Function**: `generate_rollout_offline(args, rollout_id, data_buffer, evaluation)`
- **Returns**: `RolloutFnTrainOutput(samples: list[list[Sample]], metrics: dict)`
- **Features**:
  - Caches `OfflineDataSource` globally for efficiency
  - Converts `SampleLite` → slime `Sample` with full field mapping
  - Supports 3 modes via `OFFLINE_MODE` env var: step, trajectory, dynamic_history
  - Computes and reports offline-specific metrics

### openclaw-offline/offline_loss.py
- **Purpose**: Advantage-weighted PPO/GRPO loss using offline critic advantages
- **Function**: `advantage_weighted_loss_function(args, log_probs, old_log_probs, advantages, loss_masks, samples)`
- **Returns**: `(loss: Tensor, metrics: dict)`
- **Features**:
  - PPO clipped surrogate with asymmetric clipping (eps=0.2, eps_high=0.28)
  - Per-sample offline weights from pre-computed IQL/CQL advantages
  - Falls back to reward-based proxy when no weights available
  - Configurable temperature via `OFFLINE_WEIGHT_TEMPERATURE`

### openclaw-offline/compute_weights.py
- **Purpose**: CLI tool to train IQL/CQL critic and export advantage weights
- **Usage**: `python compute_weights.py --data traj.jsonl --output weights.json --algo iql`
- **Output**: JSON mapping `"{traj_id}:{step_idx}"` → advantage_value

### openclaw-offline/run_qwen35_4b_offline_rl.sh
- **Purpose**: Example launch script matching upstream patterns exactly
- **Based on**: `openclaw-rl/run_qwen35_4b_openclaw_rl.sh`
- **Key diffs**: `--rollout-function-path offline_rollout.generate_rollout_offline`,
  `--disable-rollout-global-dataset`, `OFFLINE_*` env vars

### openclaw-offline/README.md
- **Purpose**: Integration documentation with architecture diagram, configuration reference, quick start guide

### openclaw-offline/tests/test_integration.py
- **Purpose**: 10 integration tests with mocked slime types
- **Tests**: rollout output format, sample conversion, metrics keys, evaluation mode,
  env var config, missing store error, loss computation, weight loading, PPO clipping

## Test Results

```
offline-rl/tests/          → 114 passed (6.04s)
openclaw-offline/tests/    →  25 passed (2.53s)
TOTAL                      → 139 passed
```

## Configuration Reference

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OFFLINE_TRAJECTORY_STORE` | (required) | Path to trajectories.jsonl |
| `OFFLINE_MODE` | `step` | Sampling mode: step/trajectory/dynamic_history |
| `OFFLINE_N_SAMPLES_PER_PROMPT` | `1` | Group size per rollout call |
| `OFFLINE_WEIGHT_PATH` | (none) | Path to advantage weights JSON |
| `OFFLINE_WEIGHT_TEMPERATURE` | `3.0` | Temperature for softmax weighting |

## Integration Modes

| Mode | slime Args | Description |
|------|------------|-------------|
| **Offline GRPO Replay** | `--rollout-function-path offline_rollout.generate_rollout_offline` | Standard GRPO loss on offline data |
| **Advantage-Weighted BC** | Above + `--custom-loss-function-path offline_loss.advantage_weighted_loss_function --loss-type custom_loss` | GRPO weighted by IQL/CQL advantages |
| **Pure Critic Training** | Use `offline-rl/` directly | Train only critic networks, no LLM update |

## Compatibility

- Python 3.7+ compatible (typing workarounds applied)
- Zero modifications to upstream OpenClaw-RL or slime code
- Follows exact function signatures expected by slime's `call_rollout_fn` and custom loss loader
- Matches upstream `Sample` dataclass field mapping precisely

## Improvement Round (v2)

### Robustness Enhancements

1. **offline_rollout.py**:
   - Added `reset_data_source()` function for testing/reconfiguration
   - Added `OFFLINE_MODE` value validation (raise on invalid mode)
   - Added empty group skipping in sample conversion

2. **offline_loss.py**:
   - Fixed `if not log_probs` crash on 2D tensor inputs (use `isinstance` check)
   - Added `reset_weight_cache()` function for testing
   - Added empty input handling (returns zero loss with valid metrics)
   - Added optional KL divergence penalty (`OFFLINE_KL_COEFF` env var)
   - Improved overflow protection: clamp exponent to [-10, 10] (was unbounded)
   - Added `offline_loss/mean_weight` metric to all outputs

3. **compute_weights.py**:
   - Fixed `os.makedirs` edge case for current directory output
   - Added end-to-end test coverage (IQL and CQL pipelines)

### Test Expansion (10 → 25)

| Category | New Tests |
|----------|-----------|
| Rollout | invalid mode error, consecutive calls cache, varying batch sizes, reset, rollout_id tracking, avg_reward validity |
| Loss | empty input, all-zero masks, negative advantages, KL penalty, extreme weights, gradient flow |
| Compute | IQL e2e, CQL e2e |

### Infrastructure

- Created `conftest.py` with shared test fixtures and slime mock setup
- Created `__init__.py` for proper package structure
- Added `cleanup_env_vars` autouse fixture to prevent test pollution
