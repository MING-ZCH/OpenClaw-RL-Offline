"""Shared test fixtures for offline-rl tests."""

from __future__ import annotations

import os
import tempfile

import pytest

from offline_rl.data.trajectory_store import Step, Trajectory, TrajectoryStore


def _make_trajectory(
    traj_id: str = "t1",
    domain: str = "os",
    n_steps: int = 3,
    success: bool = True,
) -> Trajectory:
    """Helper to build a test trajectory."""
    steps = []
    for i in range(n_steps):
        steps.append(Step(
            step_idx=i,
            action=f"click({100*i}, {200*i})",
            response=f"I will click at ({100*i}, {200*i})",
            reward=0.0 if i < n_steps - 1 else (1.0 if success else -1.0),
            done=(i == n_steps - 1),
        ))
    return Trajectory(
        trajectory_id=traj_id,
        domain=domain,
        example_id=f"ex_{traj_id}",
        instruction=f"Test task for {traj_id}",
        steps=steps,
        outcome_reward=1.0 if success else -1.0,
        eval_score=1.0 if success else 0.0,
        num_steps=n_steps,
        status="completed" if success else "failed",
        source="test",
    )


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_trajectories():
    """Return a list of 10 mixed-outcome trajectories."""
    trajs = []
    for i in range(10):
        trajs.append(_make_trajectory(
            traj_id=f"traj_{i}",
            domain="os" if i % 3 == 0 else ("chrome" if i % 3 == 1 else "vlc"),
            n_steps=3 + (i % 4),
            success=(i % 2 == 0),
        ))
    return trajs


@pytest.fixture
def populated_store(tmp_dir, sample_trajectories):
    """Return a TrajectoryStore pre-populated with 10 trajectories."""
    path = os.path.join(tmp_dir, "test_store.jsonl")
    store = TrajectoryStore(path)
    store.append_batch(sample_trajectories)
    return store
