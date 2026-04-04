# Offline RL Extension — User Guide

A complete guide to using the Offline RL Extension for OpenClaw-RL.

## Table of Contents

1. [Installation](#installation)
2. [Architecture Overview](#architecture-overview)
3. [Data Pipeline](#data-pipeline)
   - [Collecting Trajectories](#collecting-trajectories)
   - [TrajectoryStore](#trajectorystore)
   - [ReplayBuffer](#replaybuffer)
   - [OfflineDataSource (slime-compatible)](#offlinedatasource)
4. [Training Algorithms](#training-algorithms)
   - [IQL (Implicit Q-Learning)](#iql)
   - [CQL (Conservative Q-Learning)](#cql)
   - [AWAC (Advantage Weighted Actor-Critic)](#awac)
   - [Off-Policy GRPO](#off-policy-grpo)
   - [Choosing an Algorithm](#choosing-an-algorithm)
5. [Environment Adapters](#environment-adapters)
   - [Mock Environments (CPU Testing)](#mock-environments)
   - [Real Environments](#real-environments)
6. [Integration with OpenClaw-RL](#integration-with-openclaw-rl)
7. [Advanced Usage](#advanced-usage)
8. [FAQ](#faq)

---

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.12+ (CPU is sufficient for development and testing)
- No GPU required for mock environments

### Setup

```bash
# Clone the upstream repo (if not already done)
git clone --sparse --filter=blob:none https://github.com/THUDM/OpenClaw-RL.git openclaw-rl-upstream
cd openclaw-rl-upstream

# Install this extension
cd offline-rl
pip install -e .

# Verify installation
python -m pytest tests/ -v
# Expected: 114 passed
```

### Optional: Real Environment Dependencies

```bash
# For real OSWorld environments
pip install osworld  # or clone from xlang-ai/OSWorld

# For real AndroidWorld environments
pip install android-world  # requires Android emulator

# For real WebArena environments
pip install webarena  # requires Docker services
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Data Pipeline                                │
│                                                                     │
│  Environment Adapter ──► TrajectoryStore ──► ReplayBuffer           │
│  (Mock or Real)          (JSONL on disk)     (in-memory sampling)   │
│                                    │                                │
│                                    ▼                                │
│                          OfflineDataSource                          │
│                    (slime-compatible DataSource)                     │
│                                    │                                │
│                                    ▼                                │
│                        IQL / CQL / AWAC / GRPO                      │
│                        (offline RL training)                        │
└─────────────────────────────────────────────────────────────────────┘
```

**Key design principles:**

1. **Memory-efficient**: `TrajectoryStore` uses JSONL streaming — never loads all data at once.
2. **slime-compatible**: `OfflineDataSource` implements slime's `DataSource` interface exactly.
3. **CPU-testable**: All mock environments run without GPU, VM, or Docker.
4. **Modular**: Each component works independently.

---

## Data Pipeline

### Collecting Trajectories

Use mock environment adapters to collect trajectories on CPU:

```python
from offline_rl.envs import MockOSWorldAdapter
from offline_rl.data import Step, Trajectory, TrajectoryStore

# Create adapter and store
adapter = MockOSWorldAdapter()
store = TrajectoryStore("trajectories.jsonl")

# Collect one episode
task = adapter.get_task_configs()[0]
lease = adapter.allocate("episode_1")
reset_result = adapter.reset(lease["lease_id"])

steps = []
done = False
step_idx = 0
while not done:
    action = task.target_actions[step_idx] if step_idx < len(task.target_actions) else "wait(1)"
    result = adapter.step(lease["lease_id"], action)
    done = result["done"]
    steps.append(Step(step_idx=step_idx, action=action, response=str(result["observation"])))
    step_idx += 1

# Evaluate and save
score = adapter.evaluate(lease["lease_id"])["score"]
traj = Trajectory(
    trajectory_id="ep1",
    domain=task.domain,
    example_id=task.task_id,
    instruction=task.instruction,
    steps=steps,
    outcome_reward=1.0 if score > 0.5 else -1.0,
    eval_score=score,
    num_steps=len(steps),
    status="completed" if score > 0.5 else "failed",
    source="mock",
)
store.append(traj)
adapter.close(lease["lease_id"])

print(store.stats())
```

Or use the provided script for batch collection:

```bash
python scripts/collect_offline_data.py --env osworld --episodes 50
```

### TrajectoryStore

Persistent JSONL storage. Each line is one complete trajectory.

```python
from offline_rl.data import TrajectoryStore

# Create or open a store
store = TrajectoryStore("my_trajectories.jsonl")

# Append trajectories
store.append(traj)
store.append_batch([traj1, traj2, traj3])

# Count (cached after first call)
print(store.count())

# Stream with filters (memory-efficient)
for traj in store.iter_trajectories(domain="chrome", success_only=True, max_count=100):
    print(traj.trajectory_id, traj.eval_score)

# Statistics
print(store.stats())
# {'total_trajectories': 50, 'successful': 28, 'success_rate': 0.56,
#  'domains': {'chrome': 10, 'terminal': 8, ...}, 'avg_steps': 12.3}

# Import from gui-rl results
store = TrajectoryStore.import_from_gui_rl_results(
    results_dir="path/to/gui-rl/results/",
    output_path="trajectories_from_gui_rl.jsonl",
)
```

### ReplayBuffer

In-memory sampling interface for training.

```python
from offline_rl.data import ReplayBuffer, TrajectoryStore

store = TrajectoryStore("trajectories.jsonl")
buffer = ReplayBuffer(store=store, max_trajectories=5000, seed=42)

# Load from store (with optional filters)
loaded = buffer.load_from_store(domain="chrome", success_only=False)
print(f"Loaded {loaded} trajectories, {len(buffer)} transitions")

# Sample full trajectories (for GRPO)
trajs = buffer.sample_trajectories(n=8)

# Sample transitions (for IQL/CQL/AWAC)
batch = buffer.sample_transitions(batch_size=64)
print(batch.batch_size)               # 64
print(batch.observation_contexts[:1])  # ['Task: Open Chrome and...\nStep 0: click...\n']
print(batch.actions[:1])              # ['type(OpenClaw-RL)']
print(batch.rewards[:3])             # [0.0, 0.0, 1.0]
print(batch.dones[:3])               # [0.0, 0.0, 1.0]

# Prioritized Experience Replay (PER)
batch, weights, indices = buffer.sample_transitions_prioritized(batch_size=64, alpha=0.6)
# After computing TD errors:
td_errors = compute_td_errors(batch)
buffer.update_priorities(indices, td_errors.abs().numpy())

# Add new trajectories online
buffer.add_trajectory(new_traj)  # Auto-evicts if full
```

### OfflineDataSource

slime-compatible `DataSource` for integration with OpenClaw-RL's training loop.

```python
from offline_rl.data import TrajectoryStore, OfflineDataSource

store = TrajectoryStore("trajectories.jsonl")

# Three modes:
# - "trajectory": whole trajectory → one sample (final reward)
# - "step":       each step → one sample (step-level rewards)
# - "dynamic_history": cumulative prefix → one sample (like gui-rl)

ds = OfflineDataSource(
    store=store,
    mode="step",
    n_samples_per_prompt=4,  # Group size for GRPO
    max_token_length=2048,
    shuffle=True,
)

print(len(ds))     # Total number of samples
print(repr(ds))    # OfflineDataSource(mode='step', samples=150, n_per_prompt=4)

# slime-compatible interface
groups = ds.get_samples(num=8)  # list[list[SampleLite]]
print(len(groups))              # 8 groups
print(len(groups[0]))           # 4 samples per group

# Inspect a sample
sample = groups[0][0]
print(sample.prompt)          # "Task: Open Chrome and..."
print(sample.response)         # "Click toolbar...\nAction: click(540,50)"
print(sample.reward)           # {"score": 1.0, "step_reward": 0.33}
print(sample.tokens[:5])       # [12345, 6789, ...]
print(sample.loss_mask[:5])    # [0, 0, 0, 1, 1]  (loss on response only)
print(sample.metadata)         # {"domain": "chrome", "step_idx": 0, ...}
print(sample.status)           # "completed"

# Add new samples
ds.add_samples(new_groups)

# Save/load checkpoint
ds.save(rollout_id="epoch_1")
ds.load(rollout_id="epoch_1")
```

---

## Training Algorithms

All algorithms share a common interface:

```python
algo = SomeAlgorithm(replay_buffer=buffer, device="cpu", ...)

# Single step
metrics = algo.train_step(batch)
print(metrics.loss, metrics.extra, metrics.log_str())

# Training loop (convenience method)
all_metrics = algo.train(num_steps=1000, batch_size=64, log_interval=100)

# Evaluate
q_values = algo.get_action_values(states=["Task: ..."], actions=["click(100,50)"])

# Save/load
algo.save("checkpoint.pt")
algo.load("checkpoint.pt")
```

### IQL

**Best for**: Conservative offline learning without action generation.

```python
from offline_rl.algorithms import IQL

iql = IQL(
    replay_buffer=buffer,
    state_dim=256,        # State embedding dimension
    action_dim=256,       # Action embedding dimension
    hidden_dim=256,       # MLP hidden dimension
    lr=3e-4,              # Learning rate
    gamma=0.99,           # Discount factor
    tau=0.7,              # Expectile parameter (>0.5 biases toward upper quantiles)
    beta=3.0,             # Temperature for policy extraction
    target_update_rate=0.005,  # Polyak averaging rate
    device="cpu",
)

# Train
metrics = iql.train(num_steps=500, batch_size=32)

# Extract policy weights for LLM fine-tuning
weights = iql.get_policy_weights(states, actions)
# Use as: loss = -(weights * log_probs).mean()

# Compute advantages
advantages = iql.get_advantages(states, actions)
```

### CQL

**Best for**: When you need conservative Q-value estimates (prevents overestimation).

```python
from offline_rl.algorithms import CQL

cql = CQL(
    replay_buffer=buffer,
    alpha=1.0,            # CQL regularization strength (higher = more conservative)
    n_random_actions=10,  # Random actions for logsumexp estimate
    # ... other params same as IQL
)

metrics = cql.train(num_steps=500, batch_size=32)
# metrics.extra contains {"td_loss": ..., "cql_loss": ...}
```

### AWAC

**Best for**: Seamless offline→online transition when online data becomes available.

```python
from offline_rl.algorithms import AWAC

awac = AWAC(
    replay_buffer=buffer,
    lam=1.0,              # Advantage weighting temperature (lower = more selective)
    max_weight=100.0,     # Weight clipping for stability
    # ... other params
)

metrics = awac.train(num_steps=500, batch_size=32)
# metrics.extra contains {"critic_loss": ..., "actor_loss": ...}

# Predict actions (for online use)
action_embeddings = awac.predict_actions(states)
```

### Off-Policy GRPO

**Best for**: Aligning with upstream OpenClaw-RL's GRPO training (the recommended starting point).

```python
from offline_rl.algorithms import OffPolicyGRPO

grpo = OffPolicyGRPO(
    replay_buffer=buffer,
    clip_ratio=0.2,           # PPO-style clipping
    kl_coeff=0.01,            # KL penalty against reference policy
    n_policy_updates=4,       # Gradient steps per train_step call
    lr=1e-4,
    # ... other params
)

metrics = grpo.train(num_steps=500, batch_size=32)
# metrics.extra: {"surrogate_loss", "kl_penalty", "mean_advantage", "mean_ratio"}

# Periodically refresh reference policy
grpo.update_reference_policy()
```

### Choosing an Algorithm

| Algorithm | Strengths | When to Use |
|-----------|-----------|-------------|
| **IQL** | No OOD action queries, stable | Conservative offline-only training |
| **CQL** | Provable Q lower bound | When Q-value overestimation is a concern |
| **AWAC** | Offline→online seamless | When online fine-tuning is planned |
| **GRPO** | Matches upstream gui-rl | Default choice for OpenClaw-RL integration |

**Recommendation**: Start with **GRPO** for OpenClaw-RL integration, fall back to **IQL** if you need stable Q-value estimates for policy extraction.

---

## Environment Adapters

### Mock Environments

All mock adapters run on CPU without external dependencies.

```python
from offline_rl.envs import MockOSWorldAdapter, MockAndroidWorldAdapter, MockWebArenaAdapter

# All adapters share the same interface
for AdapterClass in [MockOSWorldAdapter, MockAndroidWorldAdapter, MockWebArenaAdapter]:
    adapter = AdapterClass()
    info = adapter.get_benchmark_info()
    print(f"{info['name']}: {info['num_tasks']} tasks, actions={info['action_types']}")
    
    # Standard episode flow
    tasks = adapter.get_task_configs()
    lease = adapter.allocate("ep1")
    adapter.reset(lease["lease_id"])
    result = adapter.step(lease["lease_id"], "click(100,100)")
    obs = adapter.get_obs(lease["lease_id"])
    score = adapter.evaluate(lease["lease_id"])
    adapter.close(lease["lease_id"])
```

**Available mock environments:**

| Adapter | Tasks | Domains | Action Types |
|---------|-------|---------|-------------|
| `MockOSWorldAdapter` | 15 | chrome, libreoffice, files, gimp, terminal, vscode, settings, vlc, composite | click, type, hotkey, scroll, wait, terminate |
| `MockAndroidWorldAdapter` | 15 | sms, email, contacts, calendar, clock, notes, settings, camera, gallery, browser, files | tap, long_press, scroll, type, press_key, swipe, open_app |
| `MockWebArenaAdapter` | 15 | shopping, cms, reddit, gitlab, maps | click, type, scroll, select, navigate, tab, fill_form |

### Real Environments

Real adapters require the corresponding benchmark packages:

```python
# OSWorld — requires VM setup
from offline_rl.envs.osworld_adapter import OSWorldAdapter
adapter = OSWorldAdapter(
    vm_config={"headless": True, "snapshot": "ubuntu-desktop"},
    max_steps=30,
)

# AndroidWorld — requires Android emulator
from offline_rl.envs.androidworld_adapter import AndroidWorldAdapter
adapter = AndroidWorldAdapter(avd_name="Pixel_6_API_33")

# WebArena — requires Docker services
from offline_rl.envs.webarena_adapter import WebArenaAdapter
adapter = WebArenaAdapter(
    shopping_url="http://localhost:7770",
    reddit_url="http://localhost:9999",
)
```

---

## Integration with OpenClaw-RL

### Using OfflineDataSource with slime

```python
# In your slime training config:
from offline_rl.data import TrajectoryStore, OfflineDataSource

store = TrajectoryStore("path/to/collected_trajectories.jsonl")
data_source = OfflineDataSource(
    store=store,
    mode="step",
    n_samples_per_prompt=4,
)

# Pass to slime's training loop as a DataSource
# data_source.get_samples(num) returns list[list[SampleLite]]
# SampleLite is compatible with slime's Sample type
```

### Converting between SampleLite and slime Sample

```python
from offline_rl.data import SampleLite

# SampleLite → slime Sample (when slime is importable)
try:
    slime_sample = sample_lite.to_slime_sample()
except ImportError:
    pass  # slime not available, use SampleLite directly

# slime Sample → SampleLite
sample_lite = SampleLite.from_slime_sample(slime_sample)
```

### Importing gui-rl Trajectories

```python
from offline_rl.data import TrajectoryStore

# Import from gui-rl's results directory structure
store = TrajectoryStore.import_from_gui_rl_results(
    results_dir="path/to/gui-rl/results/pyautogui/screenshot/",
    output_path="imported_trajectories.jsonl",
)
print(store.stats())
```

---

## Advanced Usage

### Prioritized Experience Replay with CQL

```python
from offline_rl.algorithms import CQL

cql = CQL(replay_buffer=buffer, alpha=2.0)

for step in range(1000):
    batch, weights, indices = buffer.sample_transitions_prioritized(
        batch_size=64, alpha=0.6
    )
    metrics = cql.train_step(batch)
    
    # Update priorities based on TD error
    with torch.no_grad():
        q_values = cql.get_action_values(
            batch.observation_contexts, batch.actions
        )
        td_errors = (batch.rewards - q_values.numpy()) ** 2
    buffer.update_priorities(indices, td_errors + 1e-6)
```

### Multi-Benchmark Training

```python
from offline_rl.envs import MockOSWorldAdapter, MockAndroidWorldAdapter, MockWebArenaAdapter
from offline_rl.data import TrajectoryStore

# Collect from all environments into one store
store = TrajectoryStore("multi_benchmark.jsonl")

for AdapterClass in [MockOSWorldAdapter, MockAndroidWorldAdapter, MockWebArenaAdapter]:
    adapter = AdapterClass()
    for task in adapter.get_task_configs():
        # ... collect trajectories ...
        store.append(traj)

# Train on mixed data
buffer = ReplayBuffer(store=store)
buffer.load_from_store()

grpo = OffPolicyGRPO(replay_buffer=buffer)
grpo.train(num_steps=2000, batch_size=64)
```

### Custom Environment Adapter

```python
from offline_rl.envs import BaseEnvAdapter, Observation, TaskConfig

class MyCustomAdapter(BaseEnvAdapter):
    BENCHMARK_NAME = "my_benchmark"
    ACTION_TYPES = ["click", "type", "submit"]
    
    def allocate(self, episode_id: str) -> dict:
        # Set up environment instance
        return {"ok": True, "lease_id": "..."}
    
    def reset(self, lease_id, task_config=None) -> dict:
        obs = Observation(screenshot_b64="...", step=0, accessibility_tree="...")
        return {"ok": True, "observation": obs.to_dict()}
    
    def step(self, lease_id, action, sleep_after=0.0) -> dict:
        # Execute action, return new observation
        return {"ok": True, "observation": {...}, "reward": 0.0, "done": False, "info": {}}
    
    def evaluate(self, lease_id) -> dict:
        return {"ok": True, "score": 1.0}
    
    def close(self, lease_id) -> dict:
        return {"ok": True}
    
    def get_obs(self, lease_id) -> dict:
        return {"ok": True, "observation": {...}}
    
    def get_task_configs(self):
        return [TaskConfig("task1", "Do something", "domain1")]
```

---

## FAQ

### Q: How much memory does this need?

**TrajectoryStore**: Near-zero — streams from disk.
**ReplayBuffer**: ~1 KB per transition. 10K trajectories × 10 steps = 100K transitions ≈ 100 MB.
**Algorithms**: ~10-50 MB for the lightweight test models (vocab_size=5000).

The entire test suite runs in under 10 seconds on an 8G CPU machine.

### Q: Can I use this with real LLMs?

Yes. The `TextEncoder` in the test setup uses `EmbeddingBag` as a lightweight proxy. In production, replace it with your LLM's encoder:

```python
class LLMStateEncoder(nn.Module):
    def __init__(self, llm_model):
        super().__init__()
        self.llm = llm_model
    
    def forward(self, input_ids):
        with torch.no_grad():
            outputs = self.llm(input_ids)
        return outputs.last_hidden_state[:, -1, :]  # Use last token
```

### Q: What's the difference between `ReplayBuffer` and `OfflineDataSource`?

- **ReplayBuffer**: Direct sampling of `Transition` objects (s, a, r, s', done). Used by IQL/CQL/AWAC algorithms.
- **OfflineDataSource**: Produces `SampleLite` objects compatible with slime's training loop. Used for GRPO and integration with upstream OpenClaw-RL.

### Q: How do I switch from mock to real environments?

Drop-in replacement — both implement `BaseEnvAdapter`:

```python
# Development
adapter = MockOSWorldAdapter()

# Production
adapter = OSWorldAdapter(vm_config={"headless": True})

# Same API
lease = adapter.allocate("ep1")
adapter.reset(lease["lease_id"])
result = adapter.step(lease["lease_id"], "click(540,50)")
```

### Q: How do I add a new offline RL algorithm?

Subclass `BaseOfflineAlgorithm`:

```python
from offline_rl.algorithms import BaseOfflineAlgorithm, TrainMetrics

class MyAlgorithm(BaseOfflineAlgorithm):
    def __init__(self, replay_buffer, **kwargs):
        super().__init__(replay_buffer=replay_buffer, **kwargs)
        # Initialize your networks here
        self.state_encoder = StateEncoder(self._vocab_size, 256, self.state_dim).to(self.device)
        self.action_encoder = ActionEncoder(self._vocab_size, 256, self.action_dim).to(self.device)
        # ... your networks ...
    
    def train_step(self, batch):
        # Use base class helpers:
        states, actions, next_states, rewards, dones = self._encode_batch(batch)
        # ... your training logic ...
        return TrainMetrics(loss=loss_value, extra={"my_metric": 0.5})
    
    def get_action_values(self, states, actions):
        # ...
    
    def save(self, path):
        torch.save({...}, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        # ...
```
