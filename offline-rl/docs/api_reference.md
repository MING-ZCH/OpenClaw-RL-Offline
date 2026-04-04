# API Reference

Complete API documentation for the Offline RL Extension.

## Table of Contents

- [Data Module (`offline_rl.data`)](#data-module)
  - [Step](#step)
  - [Trajectory](#trajectory)
  - [TrajectoryStore](#trajectorystore)
  - [Transition](#transition)
  - [TransitionBatch](#transitionbatch)
  - [ReplayBuffer](#replaybuffer)
  - [SampleLite](#samplelite)
  - [SampleStatus](#samplestatus)
  - [OfflineDataSource](#offlinedatasource)
- [Algorithms Module (`offline_rl.algorithms`)](#algorithms-module)
  - [TrainMetrics](#trainmetrics)
  - [TextEncoder / StateEncoder / ActionEncoder](#textencoder)
  - [BaseOfflineAlgorithm](#baseofflinealgorithm)
  - [IQL](#iql)
  - [CQL](#cql)
  - [AWAC](#awac)
  - [OffPolicyGRPO](#offpolicygrpo)
- [Environments Module (`offline_rl.envs`)](#environments-module)
  - [TaskConfig](#taskconfig)
  - [Observation](#observation)
  - [BaseEnvAdapter](#baseenvadapter)
  - [MockOSWorldAdapter](#mockosworldadapter)
  - [MockAndroidWorldAdapter](#mockandroidworldadapter)
  - [MockWebArenaAdapter](#mockwebarenaadapter)
  - [MockEnvPoolServer](#mockenvpoolserver)

---

## Data Module

```python
from offline_rl.data import (
    Step, Trajectory, TrajectoryStore,
    Transition, TransitionBatch, ReplayBuffer,
    SampleLite, SampleStatus, OfflineDataSource,
)
```

### Step

```python
@dataclass
class Step:
    step_idx: int                         # Step number within trajectory
    action: str                           # Action string (e.g., "click(540,50)")
    response: str = ""                    # Agent's text response
    reward: float = 0.0                   # Step-level reward (0 if unavailable)
    done: bool = False                    # Whether episode ended
    observation_file: Optional[str] = None  # Path to screenshot file
    info: dict = {}                       # Additional metadata
```

### Trajectory

```python
@dataclass
class Trajectory:
    trajectory_id: str      # Unique identifier
    domain: str             # Environment domain (e.g., "chrome", "terminal")
    example_id: str         # Task identifier
    instruction: str        # Natural language task instruction
    steps: list[Step]       # Sequence of steps
    outcome_reward: float   # Episode reward (1.0 success, -1.0 failure)
    eval_score: float       # Evaluation score (0.0-1.0)
    num_steps: int          # Number of steps
    status: str = "completed"  # "completed" | "failed" | "truncated" | "aborted"
    source: str = "manual"    # "gui-rl" | "manual" | "mock"
    metadata: dict = {}

    @property
    success -> bool             # True if eval_score > 0.5

    to_dict() -> dict           # Serialize to dict
    from_dict(d: dict) -> Trajectory  # Deserialize (classmethod)
```

### TrajectoryStore

Append-only JSONL storage with streaming read.

```python
class TrajectoryStore:
    __init__(path: str | Path)

    append(trajectory: Trajectory) -> None
    append_batch(trajectories: list[Trajectory]) -> None

    iter_trajectories(
        domain: str = None,        # Filter by domain
        success_only: bool = False, # Only successful trajectories
        max_count: int = None,      # Limit number of results
    ) -> Iterator[Trajectory]

    count() -> int                  # Total count (cached)
    stats() -> dict                 # {total_trajectories, successful, success_rate, domains, avg_steps}

    @classmethod
    import_from_gui_rl_results(results_dir, output_path) -> TrajectoryStore
```

### Transition

```python
@dataclass
class Transition:
    trajectory_id: str
    step_idx: int
    instruction: str
    observation_context: str        # Full text state (task + action history)
    action: str
    reward: float
    next_observation_context: str
    done: bool
    outcome_reward: float
    metadata: dict = {}
```

### TransitionBatch

```python
@dataclass
class TransitionBatch:
    instructions: list[str]
    observation_contexts: list[str]      # NOTE: not "states"
    actions: list[str]
    rewards: np.ndarray                  # shape: (batch_size,)
    next_observation_contexts: list[str]  # NOTE: not "next_states"
    dones: np.ndarray                    # shape: (batch_size,)
    outcome_rewards: np.ndarray          # shape: (batch_size,)

    @property
    batch_size -> int
```

### ReplayBuffer

```python
class ReplayBuffer:
    __init__(
        store: TrajectoryStore = None,
        max_trajectories: int = 5000,
        seed: int = 42,
    )

    load_from_store(
        domain: str = None,
        success_only: bool = False,
        max_count: int = None,
    ) -> int                    # Returns number loaded

    add_trajectory(traj: Trajectory) -> None  # Auto-evicts if full

    sample_trajectories(n: int) -> list[Trajectory]
    sample_transitions(batch_size: int) -> TransitionBatch

    sample_transitions_prioritized(
        batch_size: int, alpha: float = 0.6
    ) -> tuple[TransitionBatch, np.ndarray, np.ndarray]
    # Returns (batch, importance_weights, indices)

    update_priorities(indices: np.ndarray, new_priorities: np.ndarray) -> None

    @property num_trajectories -> int
    @property num_transitions -> int

    __len__() -> int            # Same as num_transitions
    stats() -> dict
```

### SampleLite

slime-compatible Sample object.

```python
@dataclass
class SampleLite:
    # Identification
    group_index: int = None
    index: int = None
    idx: int = 0

    # Content
    prompt: str = ""
    response: str = ""
    response_length: int = 0
    label: str = None

    # Token-level
    tokens: list[int] = []
    loss_mask: list[int] = []

    # Reward (float, dict, or None — flexible like slime)
    reward: Any = {}

    # Multimodal
    multimodal_inputs: dict = None

    # RL-specific
    rollout_log_probs: list[float] = None
    weight_versions: list[str] = []
    remove_sample: bool = False

    # Metadata
    metadata: dict = {}
    status: str = "completed"

    # Offline-RL specific
    trajectory_id: str = ""
    step_idx: int = 0

    to_slime_sample() -> Sample      # Convert to slime type (raises ImportError if slime unavailable)
    from_slime_sample(sample) -> SampleLite  # Create from slime Sample (classmethod)
```

### SampleStatus

```python
class SampleStatus:
    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"
    FAILED = "failed"
```

### OfflineDataSource

```python
class OfflineDataSource:
    __init__(
        store: TrajectoryStore,
        mode: str = "step",           # "trajectory" | "step" | "dynamic_history"
        n_samples_per_prompt: int = 1,
        max_token_length: int = 2048,
        shuffle: bool = True,
        hash_mod: int = 32000,
        save_dir: str = "./checkpoints",
    )

    # slime DataSource interface
    get_samples(num: int) -> list[list[SampleLite]]
    add_samples(samples: list[list[SampleLite]]) -> None
    save(rollout_id=None) -> None
    load(rollout_id=None) -> None

    __len__() -> int
    __repr__() -> str
```

---

## Algorithms Module

```python
from offline_rl.algorithms import (
    IQL, CQL, AWAC, OffPolicyGRPO,
    BaseOfflineAlgorithm, TextEncoder, StateEncoder, ActionEncoder,
    TrainMetrics, QNetwork, VNetwork,
)
```

### TrainMetrics

```python
@dataclass
class TrainMetrics:
    loss: float              # Combined loss value
    extra: dict[str, float]  # Algorithm-specific metrics

    log_str() -> str         # "loss=0.1234 v_loss=0.0567 q_loss=0.0667"
```

### TextEncoder

Unified text encoder used for both states and actions. `StateEncoder` and `ActionEncoder` are aliases.

```python
class TextEncoder(nn.Module):
    __init__(vocab_size=32000, embed_dim=256, hidden_dim=256)
    forward(token_ids: Tensor) -> Tensor  # (batch, seq_len) -> (batch, hidden_dim)

StateEncoder = TextEncoder  # Alias
ActionEncoder = TextEncoder  # Alias
```

### BaseOfflineAlgorithm

Abstract base class for all offline RL algorithms.

```python
class BaseOfflineAlgorithm(ABC):
    __init__(
        replay_buffer: ReplayBuffer,
        state_dim: int = 256,
        action_dim: int = 256,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
        max_token_len: int = 128,
        max_grad_norm: float = None,   # Optional gradient clipping
    )

    # Shared utilities (used by subclasses)
    _tokenize(texts: list[str]) -> Tensor
    _encode_batch(batch: TransitionBatch) -> tuple[states, actions, next_states, rewards, dones]
    _soft_update_target_pair(source: Module, target: Module, tau: float) -> None  # Static

    # Abstract (must be implemented by subclasses)
    train_step(batch: TransitionBatch) -> TrainMetrics
    get_action_values(states: list[str], actions: list[str]) -> Tensor
    save(path: str) -> None
    load(path: str) -> None

    # Convenience
    train(num_steps=1000, batch_size=64, log_interval=100) -> list[TrainMetrics]
```

### IQL

Implicit Q-Learning.

```python
class IQL(BaseOfflineAlgorithm):
    __init__(
        replay_buffer, state_dim=256, action_dim=256, hidden_dim=256,
        lr=3e-4, gamma=0.99,
        tau=0.7,                    # Expectile parameter
        beta=3.0,                   # Policy extraction temperature
        target_update_rate=0.005,
        device="cpu", **kwargs,
    )

    train_step(batch) -> TrainMetrics    # extra: {v_loss, q_loss}
    get_action_values(states, actions) -> Tensor
    get_advantages(states, actions) -> Tensor
    get_policy_weights(states, actions) -> Tensor  # For LLM fine-tuning
    save(path) / load(path)
```

### CQL

Conservative Q-Learning.

```python
class CQL(BaseOfflineAlgorithm):
    __init__(
        replay_buffer, state_dim=256, action_dim=256, hidden_dim=256,
        lr=3e-4, gamma=0.99,
        alpha=1.0,                  # CQL regularization coefficient
        n_random_actions=10,        # Random actions for logsumexp
        target_update_rate=0.005,
        device="cpu", **kwargs,
    )

    train_step(batch) -> TrainMetrics    # extra: {td_loss, cql_loss}
    get_action_values(states, actions) -> Tensor
    save(path) / load(path)
```

### AWAC

Advantage Weighted Actor-Critic.

```python
class AWAC(BaseOfflineAlgorithm):
    __init__(
        replay_buffer, state_dim=256, action_dim=256, hidden_dim=256,
        lr=3e-4, gamma=0.99,
        lam=1.0,                    # Advantage weighting temperature
        max_weight=100.0,           # Weight clipping
        target_update_rate=0.005,
        device="cpu", **kwargs,
    )

    train_step(batch) -> TrainMetrics    # extra: {critic_loss, actor_loss}
    get_action_values(states, actions) -> Tensor
    predict_actions(states: list[str]) -> Tensor  # Action embeddings
    save(path) / load(path)
```

### OffPolicyGRPO

Off-Policy Group Relative Policy Optimization.

```python
class OffPolicyGRPO(BaseOfflineAlgorithm):
    __init__(
        replay_buffer, state_dim=256, action_dim=256, hidden_dim=256,
        lr=1e-4, gamma=0.99,
        clip_ratio=0.2,             # PPO-style clipping
        kl_coeff=0.01,              # KL penalty coefficient
        n_policy_updates=4,         # Gradient steps per train_step
        device="cpu",
        max_token_len=128,
    )

    train_step(batch) -> TrainMetrics
    # extra: {surrogate_loss, kl_penalty, mean_advantage, mean_ratio}

    get_action_values(states, actions) -> Tensor  # Policy log-probs
    update_reference_policy() -> None              # Copy current → reference
    save(path) / load(path)
```

---

## Environments Module

```python
from offline_rl.envs import (
    BaseEnvAdapter, Observation, TaskConfig,
    MockEnvPoolServer,
    MockOSWorldAdapter, MockAndroidWorldAdapter, MockWebArenaAdapter,
)
```

### TaskConfig

```python
@dataclass
class TaskConfig:
    task_id: str                    # Unique task identifier
    instruction: str                # Natural language instruction
    domain: str                     # Domain category
    max_steps: int = 15             # Maximum episode length
    target_actions: list[str] = []  # Expected action sequence (for mock evaluation)
    metadata: dict = {}
```

### Observation

```python
@dataclass
class Observation:
    screenshot_b64: str = None      # Base64-encoded PNG screenshot
    step: int = 0                   # Current step number
    accessibility_tree: str = None  # A11y tree or UI hierarchy text
    url: str = None                 # Current URL (web environments)
    current_app: str = None         # Current app (Android)
    extra: dict = {}

    to_dict() -> dict               # Serialize (omits None fields)
```

### BaseEnvAdapter

Abstract interface matching gui-rl's env_pool_server API.

```python
class BaseEnvAdapter(ABC):
    BENCHMARK_NAME: str = "base"
    ACTION_TYPES: list[str] = []

    allocate(episode_id: str) -> dict          # {ok, lease_id}
    reset(lease_id, task_config=None) -> dict   # {ok, observation, task}
    get_obs(lease_id) -> dict                   # {ok, observation}
    step(lease_id, action, sleep_after=0) -> dict  # {ok, observation, reward, done, info}
    evaluate(lease_id) -> dict                  # {ok, score}
    close(lease_id) -> dict                     # {ok}
    heartbeat(lease_id) -> dict                 # {ok}  (default impl)

    get_task_configs() -> list[TaskConfig]
    get_benchmark_info() -> dict               # {name, action_types, num_tasks}
```

### MockOSWorldAdapter

15 tasks across 9 desktop domains. No VM required.

### MockAndroidWorldAdapter

15 tasks across 11 Android domains. No emulator required.

### MockWebArenaAdapter

15 tasks across 5 web apps. No Docker required.

### MockEnvPoolServer

Simulates gui-rl's Flask-based env pool HTTP server for compatibility testing.

```python
class MockEnvPoolServer:
    __init__(num_envs=4, max_steps=15, seed=42)
    
    allocate(episode_id) -> dict
    reset(lease_id) -> dict
    step(lease_id, action, sleep_after=0) -> dict
    evaluate(lease_id) -> dict
    close(lease_id) -> dict
    heartbeat(lease_id) -> dict
```
