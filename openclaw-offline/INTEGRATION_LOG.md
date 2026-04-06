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

## Bridge Extension (v3) — All 15 Algorithms

### Trigger
User request: "将算法接入 openclaw-offline 的 slime 训练桥接（compute_weights/offline_rollout）"

After completing the algorithm implementation phase (commit `f776126`, 15 algos, 179 tests),
the compute_weights bridge was extended to cover all 15 offline RL algorithms.

### compute_weights.py Rework

**Previous**: supported only `iql`, `cql`, `td3bc`
**Now**: supports all 15 algorithms

```
iql  cql  awac  grpo  td3bc  edac  dt  crr  rwft  oreo  sorl  arpo
retrospex  webrl  glider
```

#### Architecture Changes

1. **`_build_algo(args, buffer, device)`** — dispatcher replacing inline if/elif:
   - Each of 15 branches does lazy import + construct with all required kwargs
   - Shared kwargs: `state_dim=256, action_dim=256, hidden_dim=256, lr=3e-4, gamma=0.99, device`

2. **`_get_advantage(algo, algo_name, state, action)`** — unified extraction:
   - `_HAS_ADVANTAGES = {"iql", "awac", "crr", "edac", "oreo"}` → use `get_advantages()`
   - All others → use `get_action_values()` as Q/value proxy

3. **Per-algo hyperparameter CLI args added**:

   | New Flag | Default | Algorithm |
   |----------|---------|-----------|
   | `--retrospex-tau` | 0.7 | Retrospex expectile |
   | `--retrospex-lambda-scale` | 1.0 | Retrospex Q-weight λ |
   | `--webrl-alpha-orm` | 0.5 | WebRL ORM reward mix α |
   | `--glider-plan-dim` | hidden//4 | GLIDER plan embedding dim |
   | `--glider-beta` | 1.0 | GLIDER AWR temperature |
   | `--glider-tau` | 0.7 | GLIDER IQL expectile |

## Bridge Extension (v4) — 6 New Algorithms (15 → 21)

### Trigger
User request: extend the algorithm library with ArCHer, BCQ, DPO, KTO, REBEL, DigiRL.

After completing the 15-algo bridge (commit `efcb42c`), 6 new offline RL algorithms were
implemented and integrated across the entire stack.

### New Algorithms

| Algorithm | Paper | Type | Key Feature |
|-----------|-------|------|-------------|
| ArCHer | ICML 2024 (arXiv 2402.19446) | Hierarchical IQL+AWR | Multi-turn dialogue; tau=0.9; 3 optimizer groups |
| BCQ | ICML 2019 (arXiv 1812.02900) | Batch-Constrained Q | BehaviorCloningNetwork; prevents OOD extrapolation |
| DPO | NeurIPS 2023 (arXiv 2305.18290) | Direct Preference Opt | No reward model; intra-batch pairing |
| KTO | ICML 2024 (arXiv 2402.01306) | Kahneman-Tversky Opt | Binary labels; no preference pairs needed |
| REBEL | NeurIPS 2024 (arXiv 2404.16767) | Critic-Free Regression | Lightest-weight RL; pairwise reward regression |
| DigiRL | arXiv 2406.11896 (2024) | Doubly-Robust AWR | BCE V_step/V_instruct; hard-filter actor |

### compute_weights.py Updates

**Previous**: 15+4=19 algorithms (iql...glider + archer/bcq/dpo/kto)
**Now**: 21 algorithms (added rebel, digirl)

1. **`_build_algo()`**: Added `rebel` and `digirl` branches with full hyperparameter support.
2. **`_HAS_ADVANTAGES` fix**: Removed `"awac"` — AWAC only has `get_action_values()`, not `get_advantages()`.
   **Before**: `{"iql", "awac", "crr", "edac", "oreo", "archer", "bcq"}`
   **After**: `{"iql", "crr", "edac", "oreo", "archer", "bcq"}`
3. **New CLI args**:

   | New Flag | Default | Algorithm |
   |----------|---------|-----------|
   | `--rebel-eta` | 1.0 | REBEL scale parameter |
   | `--rebel-ref-interval` | 1 | REBEL reference update interval |
   | `--digirl-lam` | 0.5 | DigiRL MC vs TD mixing weight |
   | `--digirl-adv-threshold` | 0.1 | DigiRL hard-filter threshold |
   | `--digirl-max-grad-norm` | 1.0 | DigiRL gradient clipping |

### Bug Fixes
1. **AWAC `_HAS_ADVANTAGES`**: AWAC was incorrectly in the set; calling `get_advantages()` on AWAC would cause `AttributeError`. Fixed by removing from set.
2. **ArCHer `actor_lr=None`**: Constructor did not handle `None`; `Adam(lr=None)` raised `TypeError`. Fixed: `Optional[float]`, defaults to `lr`.
3. **REBEL `_encode_batch` unpacking**: Parent `OffPolicyGRPO._encode_batch()` returns `(s, a)` (2 values), not `(s, a, s', r, d)` (5). Fixed to unpack 2 + extract rewards separately.

### Test Results
- **offline-rl**: 185/185 passed (179 existing + 6 new)
- **openclaw-offline**: 29/29 passed (updated `_HAS_ADVANTAGES` assertion)
- **Total**: 214/214

### Commit
`638638e` — feat: add 6 new offline RL algorithms (ArCHer, BCQ, DPO, KTO, REBEL, DigiRL)

#### Advantage Dispatch Table

| Algorithm | Method | Notes |
|-----------|--------|-------|
| iql | `get_advantages()` | True Q-V advantage |
| awac | `get_advantages()` | True Q-V advantage |
| crr | `get_advantages()` | True Q-V advantage |
| edac | `get_advantages()` | True Q-V with ensemble std penalty |
| oreo | `get_advantages()` | Outcome-regularized Q-V |
| cql | `get_action_values()` | Conservative Q-value proxy |
| td3bc | `get_action_values()` | Value-weighted Q proxy |
| grpo | `get_action_values()` | Policy log-prob score |
| sorl | `get_action_values()` | Clipped-norm GRPO score |
| arpo | `get_action_values()` | Asymmetric ratio policy score |
| dt | `get_action_values()` | Sequence-model log-prob |
| rwft | `get_action_values()` | Reward-weighted BC score |
| retrospex | `get_action_values()` | Frozen-LLM twin-Q value |
| webrl | `get_action_values()` | ORM P(success) ∈ [0,1] |
| glider | `get_action_values()` | Plan-conditioned LL Q(s‖g, a) |

### WebRL ORM Semantics Note

`WebRL.get_action_values()` returns `σ(ORM(s,a))` — a probability in [0, 1].
When used as advantage weights in `offline_loss.py`, `exp(β * 0.5)` ≈ 4.48 for a
perfectly successful action. This is intentional: WebRL's ORM is specifically trained
to predict success probability, which is a valid monotone proxy for advantage.
If normalization is needed, downstream code can subtract the batch mean before `exp(β·A)`.

### Test Expansion (25 → 29)

| Test Class | New Tests Added |
|------------|----------------|
| `TestComputeWeights` | `test_cli_retrospex` |
| `TestComputeWeights` | `test_cli_webrl_produces_bounded_values` |
| `TestComputeWeights` | `test_cli_glider_uses_plan_conditioned_q` |
| `TestComputeWeights` | `test_advantage_dispatch_has_advantages_vs_q_value` |

### Test Results

```
offline-rl/tests/          → 179 passed (all 15 algorithms)
openclaw-offline/tests/    →  29 passed (full bridge coverage)
TOTAL                      → 208 passed
```

### Backup / Safety

`compute_weights.py.bak` — backup of the pre-v3 version (3-algo version)

## Bridge Extension (v5) — 6 More Algorithms (21 → 27)

### Trigger
User request: add 6 benchmark-driven offline RL algorithms from recent AgentBench/WebArena/OSWorld literature.

After completing the 21-algo library (commit `e6a36ae`), 6 new algorithms were implemented
from preference optimization and agent RL literature.

### New Algorithms

| Algorithm | Paper | Venue | Key Feature |
|-----------|-------|-------|-------------|
| IPO | arXiv 2310.12036 | AISTATS 2024 | Squared-error preference loss bypassing BT model |
| CPO | arXiv 2401.08417 | ICML 2024 | DPO + behavior cloning on winners |
| SimPO | arXiv 2405.14734 | NeurIPS 2024 | Reference-free, 50% less memory |
| DMPO | arXiv 2406.14868 | EMNLP 2024 | Length-normalized multi-turn DPO |
| ETO | arXiv 2403.02502 | ACL 2024 | Exploration-weighted DPO, near-miss upweighting |
| VEM | arXiv 2502.18906 | Microsoft 2025 | Value environment model + AWR policy |

### compute_weights.py Updates

**Previous**: 21 algorithms
**Now**: 30 algorithms

1. **`_build_algo()`**: Added 6 new branches (ipo, cpo, simpo, dmpo, eto, vem).
2. **Advantage dispatch**:
   - IPO/CPO/DMPO/ETO → `get_action_values()` as log-prob preference proxy
   - SimPO → `get_action_values()` returning β-scaled log-prob (no reference model)
   - VEM → `get_action_values()` returning VEM-predicted value
3. **New CLI args**:

   | New Flag | Default | Algorithm |
   |----------|---------|-----------|
   | `--ipo-beta` | 0.1 | IPO regularization strength |
   | `--cpo-beta` | 0.1 | CPO preference strength |
   | `--cpo-lambda-bc` | 1.0 | CPO BC regularization weight |
   | `--simpo-beta` | 2.0 | SimPO scaling factor |
   | `--simpo-gamma` | 0.5 | SimPO target margin |
   | `--dmpo-beta` | 0.1 | DMPO preference strength |
   | `--dmpo-length-power` | 0.5 | DMPO length normalization exponent |
   | `--eto-beta` | 0.1 | ETO preference strength |
   | `--eto-explore-alpha` | 1.0 | ETO exploration weight scale |
   | `--vem-beta` | 1.0 | VEM AWR temperature |
   | `--vem-alpha-awr` | 1.0 | VEM AWR advantage scale |

### Test Results
- **offline-rl**: 191/191 passed (185 existing + 6 new)
- **openclaw-offline**: 29/29 passed
- **Total**: 220/220

### Commit
`d211087` — feat: add 6 benchmark-driven offline RL algorithms (IPO, CPO, SimPO, DMPO, ETO, VEM)
