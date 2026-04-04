"""
OfflineDataSource: slime-compatible DataSource for offline RL.

Bridges TrajectoryStore (offline trajectory data) with slime's training loop
by producing Sample objects from pre-collected trajectories.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class SampleStatus:
    """Status enum matching slime's Sample.Status values."""
    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass
class SampleLite:
    """
    Lightweight Sample-compatible object for offline data.

    Mirrors the key fields of slime's Sample type so downstream code
    (algorithms, training loops) can work with both online and offline data.
    When slime is available, use `to_slime_sample()` to convert.
    """

    # Identification (match slime's Sample naming)
    group_index: int = None
    index: int = None
    idx: int = 0  # internal counter

    # Prompt / Response
    prompt: str = ""
    response: str = ""
    response_length: int = 0
    label: str = None

    # Token-level data
    tokens: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)

    # Reward information — accepts float, dict, or None like slime
    reward: Any = field(default_factory=dict)

    # Multimodal (placeholder for GUI screenshots etc.)
    multimodal_inputs: dict = None

    # Log probs from rollout (for importance sampling / off-policy)
    rollout_log_probs: list[float] = None

    # Weight version tracking
    weight_versions: list[str] = field(default_factory=list)

    # Whether to exclude from training
    remove_sample: bool = False

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Status: matches slime's Sample.Status enum string values
    status: str = SampleStatus.COMPLETED

    # Trajectory context (offline-rl specific)
    trajectory_id: str = ""
    step_idx: int = 0

    def to_slime_sample(self):
        """
        Convert to slime's Sample type when slime is importable.

        Returns a slime Sample or raises ImportError.
        """
        from slime.utils.types import Sample
        s = Sample()
        s.group_index = self.group_index
        s.index = self.index
        s.prompt = self.prompt
        s.tokens = self.tokens
        s.response = self.response
        s.response_length = self.response_length or len(self.response)
        s.label = self.label
        s.reward = self.reward
        s.loss_mask = self.loss_mask
        s.weight_versions = self.weight_versions or []
        s.rollout_log_probs = self.rollout_log_probs
        s.multimodal_inputs = self.multimodal_inputs
        s.remove_sample = self.remove_sample
        s.metadata = dict(self.metadata)
        s.metadata["trajectory_id"] = self.trajectory_id
        s.metadata["step_idx"] = self.step_idx
        # Map status string to enum
        status_map = {
            "pending": Sample.Status.PENDING,
            "completed": Sample.Status.COMPLETED,
            "truncated": Sample.Status.TRUNCATED,
            "aborted": Sample.Status.ABORTED,
            "failed": Sample.Status.FAILED,
        }
        s.status = status_map.get(self.status, Sample.Status.COMPLETED)
        return s

    @classmethod
    def from_slime_sample(cls, sample) -> "SampleLite":
        """Create SampleLite from a slime Sample object."""
        return cls(
            group_index=sample.group_index,
            index=sample.index,
            prompt=sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt),
            response=sample.response,
            response_length=sample.response_length,
            label=sample.label,
            tokens=list(sample.tokens) if sample.tokens else [],
            loss_mask=list(sample.loss_mask) if sample.loss_mask else [],
            reward=sample.reward,
            multimodal_inputs=sample.multimodal_inputs,
            rollout_log_probs=list(sample.rollout_log_probs) if sample.rollout_log_probs else None,
            weight_versions=list(sample.weight_versions) if sample.weight_versions else [],
            remove_sample=sample.remove_sample,
            metadata=dict(sample.metadata) if sample.metadata else {},
            status=sample.status.value if hasattr(sample.status, "value") else str(sample.status),
            trajectory_id=sample.metadata.get("trajectory_id", "") if sample.metadata else "",
            step_idx=sample.metadata.get("step_idx", 0) if sample.metadata else 0,
        )


def _simple_tokenize(text: str, hash_mod: int = 32000) -> list[int]:
    """Hash-based tokenizer for CPU testing (no GPU model needed)."""
    words = text.split()
    tokens = []
    for w in words:
        h = int(hashlib.md5(w.encode()).hexdigest()[:8], 16) % hash_mod
        tokens.append(h)
    return tokens


class OfflineDataSource:
    """
    slime-compatible DataSource that loads from TrajectoryStore.

    Instead of generating new trajectories online, this DataSource reads
    pre-collected trajectories and converts them to training samples.

    Three modes:
        1. trajectory: Each trajectory → one Sample (final reward)
        2. step: Each step → one Sample (step-level rewards / PRM)
        3. dynamic_history: Like gui-rl's dynamic_history, each prefix → one Sample

    Usage:
        from offline_rl.data.trajectory_store import TrajectoryStore
        store = TrajectoryStore("/path/to/trajectories.jsonl")
        ds = OfflineDataSource(store, mode="step", n_samples_per_prompt=4)

        # slime-compatible interface
        batches = ds.get_samples(8)  # list[list[SampleLite]]
    """

    def __init__(
        self,
        store: Any,  # TrajectoryStore
        mode: str = "step",
        n_samples_per_prompt: int = 1,
        max_token_length: int = 2048,
        shuffle: bool = True,
        hash_mod: int = 32000,
        save_dir: str = "./checkpoints",
    ):
        self.store = store
        self.mode = mode
        self.n_samples_per_prompt = n_samples_per_prompt
        self.max_token_length = max_token_length
        self.shuffle = shuffle
        self.hash_mod = hash_mod
        self._save_dir = save_dir

        # Pre-build sample pool
        self._sample_pool: list[SampleLite] = []
        self._build_sample_pool()
        self._cursor = 0

        logger.info(
            "OfflineDataSource: mode=%s, samples=%d, n_per_prompt=%d",
            mode, len(self._sample_pool), n_samples_per_prompt,
        )

    def _build_sample_pool(self):
        """Convert all trajectories into SampleLite objects."""
        idx = 0
        for traj in self.store.iter_trajectories():
            if self.mode == "trajectory":
                samples = self._trajectory_to_samples(traj, idx)
            elif self.mode == "step":
                samples = self._steps_to_samples(traj, idx)
            elif self.mode == "dynamic_history":
                samples = self._dynamic_history_to_samples(traj, idx)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")
            self._sample_pool.extend(samples)
            idx += len(samples)

        if self.shuffle:
            random.shuffle(self._sample_pool)

    def _trajectory_to_samples(self, traj, start_idx: int) -> list[SampleLite]:
        """One trajectory → one Sample with full conversation concatenated."""
        prompt = f"Task: {traj.instruction}"
        response_parts = []
        for step in traj.steps:
            response_parts.append(f"Step {step.step_idx}: {step.response}")
            if step.action:
                response_parts.append(f"Action: {step.action}")
        response = "\n".join(response_parts)

        prompt_tokens = _simple_tokenize(prompt, self.hash_mod)
        response_tokens = _simple_tokenize(response, self.hash_mod)
        all_tokens = (prompt_tokens + response_tokens)[:self.max_token_length]
        loss_mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)
        loss_mask = loss_mask[:len(all_tokens)]

        return [SampleLite(
            idx=start_idx,
            prompt=prompt,
            response=response,
            tokens=all_tokens,
            loss_mask=loss_mask,
            reward={"score": traj.eval_score, "outcome": traj.outcome_reward},
            metadata={
                "domain": traj.domain,
                "example_id": traj.example_id,
                "num_steps": traj.num_steps,
                "source": traj.source,
            },
            status=traj.status,
            trajectory_id=traj.trajectory_id,
            step_idx=-1,
        )]

    def _steps_to_samples(self, traj, start_idx: int) -> list[SampleLite]:
        """Each step → one Sample with step-level reward."""
        samples = []
        prompt = f"Task: {traj.instruction}"
        for i, step in enumerate(traj.steps):
            response = f"{step.response}\nAction: {step.action}" if step.action else step.response
            prompt_tokens = _simple_tokenize(prompt, self.hash_mod)
            response_tokens = _simple_tokenize(response, self.hash_mod)
            all_tokens = (prompt_tokens + response_tokens)[:self.max_token_length]
            loss_mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)
            loss_mask = loss_mask[:len(all_tokens)]

            # Use step reward if available, else distribute outcome reward
            step_reward = step.reward if step.reward != 0.0 else (
                traj.outcome_reward / max(traj.num_steps, 1)
            )

            samples.append(SampleLite(
                idx=start_idx + i,
                prompt=prompt,
                response=response,
                tokens=all_tokens,
                loss_mask=loss_mask,
                reward={"score": traj.eval_score, "step_reward": step_reward},
                metadata={
                    "domain": traj.domain,
                    "example_id": traj.example_id,
                    "step_idx": step.step_idx,
                    "done": step.done,
                    "source": traj.source,
                },
                status=traj.status,
                trajectory_id=traj.trajectory_id,
                step_idx=step.step_idx,
            ))
        return samples

    def _dynamic_history_to_samples(self, traj, start_idx: int) -> list[SampleLite]:
        """
        Dynamic history mode: each step's prefix → one Sample.
        Mirrors gui-rl's dynamic_history training mode.
        """
        samples = []
        prompt_base = f"Task: {traj.instruction}\n"
        history_parts = []

        for i, step in enumerate(traj.steps):
            # Build prompt with history
            prompt = prompt_base + "\n".join(history_parts) if history_parts else prompt_base
            response = f"{step.response}\nAction: {step.action}" if step.action else step.response

            prompt_tokens = _simple_tokenize(prompt, self.hash_mod)
            response_tokens = _simple_tokenize(response, self.hash_mod)
            all_tokens = (prompt_tokens + response_tokens)[:self.max_token_length]
            loss_mask = [0] * len(prompt_tokens) + [1] * len(response_tokens)
            loss_mask = loss_mask[:len(all_tokens)]

            step_reward = step.reward if step.reward != 0.0 else (
                traj.outcome_reward / max(traj.num_steps, 1)
            )

            samples.append(SampleLite(
                idx=start_idx + i,
                prompt=prompt,
                response=response,
                tokens=all_tokens,
                loss_mask=loss_mask,
                reward={"score": traj.eval_score, "step_reward": step_reward},
                metadata={
                    "domain": traj.domain,
                    "example_id": traj.example_id,
                    "step_idx": step.step_idx,
                    "history_length": len(history_parts),
                    "source": traj.source,
                },
                status=traj.status,
                trajectory_id=traj.trajectory_id,
                step_idx=step.step_idx,
            ))

            # Append to history for next step
            history_parts.append(f"Step {step.step_idx}: {step.response}")
            if step.action:
                history_parts.append(f"Action: {step.action}")

        return samples

    # ---- slime-compatible DataSource interface ----

    def get_samples(self, num: int) -> list[list[SampleLite]]:
        """
        Get groups of samples, compatible with slime DataSource.get_samples().

        Returns list of lists: outer=groups, inner=n_samples_per_prompt copies.
        """
        groups: list[list[SampleLite]] = []
        for _ in range(num):
            if self._cursor >= len(self._sample_pool):
                if self.shuffle:
                    random.shuffle(self._sample_pool)
                self._cursor = 0
            if self._cursor >= len(self._sample_pool):
                break

            sample = self._sample_pool[self._cursor]
            self._cursor += 1

            # Duplicate for n_samples_per_prompt (each group gets n copies)
            group = [sample] * self.n_samples_per_prompt
            groups.append(group)

        return groups

    def add_samples(self, samples: list[list[SampleLite]]) -> None:
        """
        Add new samples, compatible with slime DataSource.add_samples().

        Args:
            samples: list of groups, each group is a list of SampleLite.
        """
        flat = [s for group in samples for s in group]
        self._sample_pool.extend(flat)
        logger.info("Added %d samples (%d groups), total=%d",
                     len(flat), len(samples), len(self._sample_pool))

    def save(self, rollout_id=None) -> None:
        """
        Save data source state, compatible with slime DataSource.save().

        Args:
            rollout_id: Identifier for this rollout checkpoint.
        """
        import json
        save_dir = os.path.join(self._save_dir, str(rollout_id or "default"))
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "offline_data_source_state.json")
        meta = {
            "mode": self.mode,
            "n_samples_per_prompt": self.n_samples_per_prompt,
            "pool_size": len(self._sample_pool),
            "cursor": self._cursor,
            "rollout_id": rollout_id,
        }
        with open(path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Saved OfflineDataSource state to %s", path)

    def load(self, rollout_id=None) -> None:
        """
        Load data source state, compatible with slime DataSource.load().

        Args:
            rollout_id: Identifier for rollout checkpoint to load.
        """
        import json
        save_dir = os.path.join(self._save_dir, str(rollout_id or "default"))
        path = os.path.join(save_dir, "offline_data_source_state.json")
        if os.path.exists(path):
            with open(path) as f:
                meta = json.load(f)
            self._cursor = meta.get("cursor", 0)
            logger.info("Loaded OfflineDataSource state from %s", path)
        else:
            logger.warning("No saved state at %s, starting fresh", path)

    def __len__(self) -> int:
        return len(self._sample_pool)

    def __repr__(self) -> str:
        return (
            f"OfflineDataSource(mode={self.mode!r}, samples={len(self._sample_pool)}, "
            f"n_per_prompt={self.n_samples_per_prompt})"
        )
