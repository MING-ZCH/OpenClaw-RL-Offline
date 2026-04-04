"""Tests for ReplayBuffer."""

from __future__ import annotations

import numpy as np

from offline_rl.data.replay_buffer import ReplayBuffer, Transition, TransitionBatch, _trajectory_to_transitions
from offline_rl.data.trajectory_store import TrajectoryStore
from tests.conftest import _make_trajectory


class TestTransitionConversion:
    """Test trajectory → transition conversion."""

    def test_basic_conversion(self):
        traj = _make_trajectory("t1", n_steps=3, success=True)
        transitions = _trajectory_to_transitions(traj)
        assert len(transitions) == 3
        for i, tr in enumerate(transitions):
            assert tr.trajectory_id == "t1"
            assert tr.step_idx == i
            assert "Task:" in tr.observation_context
            assert tr.action == traj.steps[i].action

    def test_last_step_done(self):
        traj = _make_trajectory("t1", n_steps=4, success=True)
        transitions = _trajectory_to_transitions(traj)
        # Last step should have done=True
        assert transitions[-1].done is True

    def test_reward_assignment(self):
        traj = _make_trajectory("t1", n_steps=3, success=True)
        transitions = _trajectory_to_transitions(traj)
        # Only the last step has nonzero reward (from step.reward) or outcome_reward
        assert transitions[-1].reward != 0.0
        # Intermediate steps have reward=0 (step.reward is 0.0 and not last)
        for tr in transitions[:-1]:
            assert tr.reward == 0.0

    def test_extracts_behavior_log_probs_from_step_metadata(self):
        traj = _make_trajectory("t1", n_steps=3, success=True)
        traj.steps[0].info["behavior_log_prob"] = -0.75
        traj.steps[1].info["rollout_log_probs"] = [-0.2, -0.3]

        transitions = _trajectory_to_transitions(traj)

        assert transitions[0].behavior_log_prob is not None
        assert transitions[1].behavior_log_prob is not None
        assert np.isclose(transitions[0].behavior_log_prob, -0.75)
        assert np.isclose(transitions[1].behavior_log_prob, -0.5)
        assert transitions[2].behavior_log_prob is None

    def test_extracts_behavior_log_probs_from_trajectory_metadata(self):
        traj = _make_trajectory("t1", n_steps=3, success=True)
        traj.metadata["behavior_log_probs"] = {"1": -1.25}

        transitions = _trajectory_to_transitions(traj)

        assert transitions[0].behavior_log_prob is None
        assert transitions[1].behavior_log_prob is not None
        assert np.isclose(transitions[1].behavior_log_prob, -1.25)


class TestReplayBuffer:
    """Test ReplayBuffer sampling functionality."""

    def test_load_from_store(self, populated_store):
        buf = ReplayBuffer(store=populated_store, max_trajectories=100)
        loaded = buf.load_from_store()
        assert loaded == 10
        assert len(buf._trajectories) == 10
        assert len(buf._transitions) > 0

    def test_sample_trajectories(self, populated_store):
        buf = ReplayBuffer(store=populated_store)
        buf.load_from_store()
        sampled = buf.sample_trajectories(3)
        assert len(sampled) == 3

    def test_sample_transitions(self, populated_store):
        buf = ReplayBuffer(store=populated_store)
        buf.load_from_store()
        batch = buf.sample_transitions(8)
        assert isinstance(batch, TransitionBatch)
        assert batch.batch_size == 8
        assert len(batch.instructions) == 8
        assert batch.rewards.shape == (8,)
        assert batch.dones.shape == (8,)
        assert batch.behavior_log_probs.shape == (8,)

    def test_sample_transitions_preserves_behavior_log_probs(self, tmp_dir):
        path = f"{tmp_dir}/behavior_store.jsonl"
        store = TrajectoryStore(path)
        traj = _make_trajectory("behavior_traj", n_steps=3, success=True)
        traj.steps[0].info["behavior_log_prob"] = -0.3
        traj.steps[1].info["old_log_prob"] = -0.6
        store.append(traj)

        buf = ReplayBuffer(store=store)
        buf.load_from_store()
        batch = buf.sample_transitions(3)

        finite = batch.behavior_log_probs[np.isfinite(batch.behavior_log_probs)]
        assert finite.shape == (2,)
        assert np.isclose(finite.sum(), -0.9)

    def test_prioritized_sampling(self, populated_store):
        buf = ReplayBuffer(store=populated_store)
        buf.load_from_store()
        batch, weights, indices = buf.sample_transitions_prioritized(8)
        assert batch.batch_size == 8
        assert weights.shape == (8,)
        assert indices.shape == (8,)
        # Weights should be positive
        assert np.all(weights > 0)

    def test_update_priorities(self, populated_store):
        buf = ReplayBuffer(store=populated_store)
        buf.load_from_store()
        _, _, indices = buf.sample_transitions_prioritized(4)
        new_prios = np.array([10.0, 5.0, 1.0, 0.1], dtype=np.float32)
        buf.update_priorities(indices, new_prios)
        # After updating, high-priority transitions should be sampled more
        assert buf._priorities is not None

    def test_add_trajectory_with_eviction(self, populated_store):
        buf = ReplayBuffer(store=populated_store, max_trajectories=5)
        buf.load_from_store(max_count=5)
        assert len(buf._trajectories) == 5

        new_traj = _make_trajectory("new_t", n_steps=2)
        buf.add_trajectory(new_traj)
        # Should still be at max
        assert len(buf._trajectories) == 5

    def test_empty_buffer_raises(self):
        buf = ReplayBuffer()
        try:
            buf.sample_transitions(1)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_domain_filter(self, populated_store):
        buf = ReplayBuffer(store=populated_store, max_trajectories=100)
        loaded = buf.load_from_store(domain="os")
        assert loaded > 0
        # All loaded trajectories should be "os" domain
        for t in buf._trajectories:
            assert t.domain == "os"

    def test_success_only_filter(self, populated_store):
        buf = ReplayBuffer(store=populated_store, max_trajectories=100)
        loaded = buf.load_from_store(success_only=True)
        assert loaded == 5
        for t in buf._trajectories:
            assert t.success

    def test_len(self, populated_store):
        """Test __len__ returns num_transitions."""
        buf = ReplayBuffer(store=populated_store)
        buf.load_from_store()
        assert len(buf) == buf.num_transitions
        assert len(buf) > 0

    def test_len_empty(self):
        """Test __len__ on empty buffer."""
        buf = ReplayBuffer()
        assert len(buf) == 0


class TestPackageExports:
    """Test that __init__.py exports work correctly."""

    def test_algorithm_exports(self):
        from offline_rl.algorithms import (
            IQL, CQL, AWAC, OffPolicyGRPO,
            BaseOfflineAlgorithm, TextEncoder, StateEncoder, ActionEncoder,
            TrainMetrics, QNetwork, VNetwork,
        )
        # TextEncoder and StateEncoder/ActionEncoder should be the same class
        assert TextEncoder is StateEncoder
        assert TextEncoder is ActionEncoder

    def test_data_exports(self):
        from offline_rl.data import (
            Step, Trajectory, TrajectoryStore,
            Transition, TransitionBatch, ReplayBuffer,
            SampleLite, SampleStatus, OfflineDataSource,
        )
        assert SampleStatus.COMPLETED == "completed"
