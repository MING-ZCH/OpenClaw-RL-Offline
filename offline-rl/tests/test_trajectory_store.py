"""Tests for TrajectoryStore."""

from __future__ import annotations

import os

from offline_rl.data.trajectory_store import Step, Trajectory, TrajectoryStore
from tests.conftest import _make_trajectory


class TestTrajectoryStore:
    """Core TrajectoryStore tests."""

    def test_append_and_count(self, tmp_dir):
        path = os.path.join(tmp_dir, "store.jsonl")
        store = TrajectoryStore(path)
        assert store.count() == 0

        store.append(_make_trajectory("t1"))
        assert store.count() == 1

        store.append(_make_trajectory("t2"))
        assert store.count() == 2

    def test_append_batch(self, tmp_dir):
        path = os.path.join(tmp_dir, "store.jsonl")
        store = TrajectoryStore(path)
        trajs = [_make_trajectory(f"t{i}") for i in range(5)]
        store.append_batch(trajs)
        assert store.count() == 5

    def test_iter_trajectories(self, populated_store):
        trajs = list(populated_store.iter_trajectories())
        assert len(trajs) == 10
        for t in trajs:
            assert isinstance(t, Trajectory)
            assert len(t.steps) >= 3

    def test_filter_by_domain(self, populated_store):
        os_trajs = list(populated_store.iter_trajectories(domain="os"))
        chrome_trajs = list(populated_store.iter_trajectories(domain="chrome"))
        vlc_trajs = list(populated_store.iter_trajectories(domain="vlc"))
        total = len(os_trajs) + len(chrome_trajs) + len(vlc_trajs)
        assert total == 10
        assert len(os_trajs) > 0
        assert len(chrome_trajs) > 0

    def test_filter_success_only(self, populated_store):
        success = list(populated_store.iter_trajectories(success_only=True))
        # Indices 0,2,4,6,8 are successful (even indices)
        assert len(success) == 5
        for t in success:
            assert t.eval_score > 0.5

    def test_max_count(self, populated_store):
        trajs = list(populated_store.iter_trajectories(max_count=3))
        assert len(trajs) == 3

    def test_stats(self, populated_store):
        stats = populated_store.stats()
        assert stats["total_trajectories"] == 10
        assert stats["successful"] == 5
        assert abs(stats["success_rate"] - 0.5) < 1e-6
        assert "os" in stats["domains"]
        assert stats["avg_steps"] > 0

    def test_trajectory_serialization_roundtrip(self, tmp_dir):
        path = os.path.join(tmp_dir, "rt.jsonl")
        store = TrajectoryStore(path)
        original = _make_trajectory("rt1", n_steps=5, success=True)
        store.append(original)

        loaded = list(store.iter_trajectories())[0]
        assert loaded.trajectory_id == original.trajectory_id
        assert loaded.domain == original.domain
        assert loaded.num_steps == original.num_steps
        assert loaded.eval_score == original.eval_score
        assert len(loaded.steps) == len(original.steps)
        for s_o, s_l in zip(original.steps, loaded.steps):
            assert s_o.action == s_l.action
            assert s_o.step_idx == s_l.step_idx

    def test_empty_store(self, tmp_dir):
        path = os.path.join(tmp_dir, "empty.jsonl")
        store = TrajectoryStore(path)
        assert store.count() == 0
        assert list(store.iter_trajectories()) == []
        stats = store.stats()
        assert stats["total_trajectories"] == 0

    def test_step_dataclass(self):
        step = Step(step_idx=0, action="click(10,20)", response="Clicking", reward=0.5, done=True)
        assert step.step_idx == 0
        assert step.done is True
        assert step.reward == 0.5

    def test_trajectory_success_property(self):
        t = _make_trajectory("s1", success=True)
        assert t.success is True
        t2 = _make_trajectory("s2", success=False)
        assert t2.success is False
