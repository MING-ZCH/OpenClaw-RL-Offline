"""
ReplayBuffer: Sampling interface for offline RL training.

Provides trajectory-level and transition-level sampling with optional
priority weighting. Memory-efficient: uses TrajectoryStore for streaming.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .trajectory_store import Step, Trajectory, TrajectoryStore

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """A single (s, a, r, s', done) transition."""
    trajectory_id: str
    step_idx: int
    instruction: str
    observation_context: str  # Textual state representation (action history up to this point)
    action: str
    reward: float
    next_observation_context: str
    done: bool
    outcome_reward: float  # Episode-level outcome
    metadata: dict = field(default_factory=dict)


@dataclass
class TransitionBatch:
    """Batch of transitions for training."""
    instructions: list[str]
    observation_contexts: list[str]
    actions: list[str]
    rewards: np.ndarray
    next_observation_contexts: list[str]
    dones: np.ndarray
    outcome_rewards: np.ndarray

    @property
    def batch_size(self) -> int:
        return len(self.instructions)


def _trajectory_to_transitions(traj: Trajectory) -> list[Transition]:
    """Convert a Trajectory into a list of Transitions."""
    transitions = []
    for i, step in enumerate(traj.steps):
        # Build observation context from action history
        obs_context = f"Task: {traj.instruction}\n"
        for prev in traj.steps[:i]:
            obs_context += f"Step {prev.step_idx}: {prev.action}\n"

        next_obs_context = obs_context + f"Step {step.step_idx}: {step.action}\n"

        is_last = (i == len(traj.steps) - 1)
        # For intermediate steps, reward is 0 unless process reward is available
        step_reward = step.reward if step.reward != 0.0 else (traj.outcome_reward if is_last else 0.0)

        transitions.append(Transition(
            trajectory_id=traj.trajectory_id,
            step_idx=step.step_idx,
            instruction=traj.instruction,
            observation_context=obs_context,
            action=step.action,
            reward=step_reward,
            next_observation_context=next_obs_context,
            done=is_last or step.done,
            outcome_reward=traj.outcome_reward,
        ))
    return transitions


class ReplayBuffer:
    """
    Offline replay buffer for agent trajectories.

    Supports two sampling modes:
    1. Trajectory-level: sample full episodes (for GRPO-style training)
    2. Transition-level: sample individual (s,a,r,s',done) tuples (for IQL/CQL/AWAC)

    Memory management:
    - `max_trajectories`: cap the in-memory trajectory pool
    - When capacity is exceeded, random trajectories are evicted
    """

    def __init__(
        self,
        store: Optional[TrajectoryStore] = None,
        max_trajectories: int = 5000,
        seed: int = 42,
    ):
        self.store = store
        self.max_trajectories = max_trajectories
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        self._trajectories: list[Trajectory] = []
        self._transitions: list[Transition] = []
        self._priorities: Optional[np.ndarray] = None

    def load_from_store(
        self,
        domain: Optional[str] = None,
        success_only: bool = False,
        max_count: Optional[int] = None,
    ) -> int:
        """Load trajectories from TrajectoryStore into memory."""
        if self.store is None:
            raise ValueError("No TrajectoryStore configured")

        loaded = 0
        effective_max = max_count or self.max_trajectories
        for traj in self.store.iter_trajectories(
            domain=domain, success_only=success_only, max_count=effective_max
        ):
            self._trajectories.append(traj)
            self._transitions.extend(_trajectory_to_transitions(traj))
            loaded += 1
            if loaded >= self.max_trajectories:
                break

        self._priorities = None  # Reset priorities
        logger.info(
            "Loaded %d trajectories (%d transitions) into replay buffer",
            loaded, len(self._transitions),
        )
        return loaded

    def add_trajectory(self, traj: Trajectory) -> None:
        """Add a trajectory to the buffer (with eviction if full)."""
        if len(self._trajectories) >= self.max_trajectories:
            # Evict random trajectory
            evict_idx = self.rng.randint(0, len(self._trajectories) - 1)
            evicted = self._trajectories.pop(evict_idx)
            self._transitions = [
                t for t in self._transitions
                if t.trajectory_id != evicted.trajectory_id
            ]

        self._trajectories.append(traj)
        self._transitions.extend(_trajectory_to_transitions(traj))
        self._priorities = None

        if self.store is not None:
            self.store.append(traj)

    def sample_trajectories(self, n: int) -> list[Trajectory]:
        """Sample n full trajectories uniformly."""
        if not self._trajectories:
            return []
        n = min(n, len(self._trajectories))
        return self.rng.sample(self._trajectories, n)

    def sample_transitions(self, batch_size: int) -> TransitionBatch:
        """Sample a batch of transitions uniformly."""
        if not self._transitions:
            raise ValueError("Replay buffer is empty")

        batch_size = min(batch_size, len(self._transitions))
        indices = self.np_rng.choice(len(self._transitions), batch_size, replace=False)
        selected = [self._transitions[i] for i in indices]

        return TransitionBatch(
            instructions=[t.instruction for t in selected],
            observation_contexts=[t.observation_context for t in selected],
            actions=[t.action for t in selected],
            rewards=np.array([t.reward for t in selected], dtype=np.float32),
            next_observation_contexts=[t.next_observation_context for t in selected],
            dones=np.array([t.done for t in selected], dtype=np.float32),
            outcome_rewards=np.array([t.outcome_reward for t in selected], dtype=np.float32),
        )

    def sample_transitions_prioritized(
        self, batch_size: int, alpha: float = 0.6
    ) -> tuple[TransitionBatch, np.ndarray, np.ndarray]:
        """
        Prioritized experience replay sampling.

        Returns (batch, weights, indices) where weights are importance sampling weights.
        Call update_priorities(indices, new_priorities) after computing TD errors.
        """
        if not self._transitions:
            raise ValueError("Replay buffer is empty")

        n = len(self._transitions)
        batch_size = min(batch_size, n)

        if self._priorities is None:
            self._priorities = np.ones(n, dtype=np.float64)

        probs = self._priorities[:n] ** alpha
        probs /= probs.sum()

        indices = self.np_rng.choice(n, batch_size, replace=False, p=probs)
        selected = [self._transitions[i] for i in indices]

        # Importance sampling weights
        weights = (n * probs[indices]) ** (-1.0)
        weights /= weights.max()

        batch = TransitionBatch(
            instructions=[t.instruction for t in selected],
            observation_contexts=[t.observation_context for t in selected],
            actions=[t.action for t in selected],
            rewards=np.array([t.reward for t in selected], dtype=np.float32),
            next_observation_contexts=[t.next_observation_context for t in selected],
            dones=np.array([t.done for t in selected], dtype=np.float32),
            outcome_rewards=np.array([t.outcome_reward for t in selected], dtype=np.float32),
        )

        return batch, weights.astype(np.float32), indices

    def update_priorities(self, indices: np.ndarray, new_priorities: np.ndarray) -> None:
        """Update priorities for sampled transitions (for PER)."""
        if self._priorities is None:
            self._priorities = np.ones(len(self._transitions), dtype=np.float64)
        for idx, prio in zip(indices, new_priorities):
            if idx < len(self._priorities):
                self._priorities[idx] = max(float(prio), 1e-6)

    @property
    def num_trajectories(self) -> int:
        return len(self._trajectories)

    @property
    def num_transitions(self) -> int:
        return len(self._transitions)

    def __len__(self) -> int:
        """Return total number of transitions (standard container protocol)."""
        return len(self._transitions)

    def stats(self) -> dict:
        if not self._trajectories:
            return {"num_trajectories": 0, "num_transitions": 0}
        success = sum(1 for t in self._trajectories if t.success)
        return {
            "num_trajectories": self.num_trajectories,
            "num_transitions": self.num_transitions,
            "success_rate": success / self.num_trajectories,
            "avg_steps": np.mean([t.num_steps for t in self._trajectories]),
        }
