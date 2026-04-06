"""Tests for IQL, CQL, AWAC, and Off-Policy GRPO algorithms on CPU with mock data."""

from __future__ import annotations

import io
import os
import tempfile

import torch

from offline_rl.data.replay_buffer import ReplayBuffer
from offline_rl.data.trajectory_store import TrajectoryStore
from offline_rl.algorithms.iql import IQL
from offline_rl.algorithms.cql import CQL
from offline_rl.algorithms.awac import AWAC
from offline_rl.algorithms.off_policy_grpo import OffPolicyGRPO
from offline_rl.algorithms.base import TrainMetrics
from tests.conftest import _make_trajectory


def _create_buffer_with_data(tmp_dir: str, n_trajs: int = 20) -> ReplayBuffer:
    """Helper: create and populate a replay buffer."""
    path = os.path.join(tmp_dir, "alg_store.jsonl")
    store = TrajectoryStore(path)
    for i in range(n_trajs):
        store.append(_make_trajectory(
            f"t{i}", domain="os", n_steps=3 + i % 3, success=(i % 2 == 0)
        ))
    buf = ReplayBuffer(store=store, max_trajectories=n_trajs)
    buf.load_from_store()
    return buf


class TestIQL:
    """Test IQL algorithm on CPU."""

    def test_single_train_step(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        iql = IQL(replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64, device="cpu")
        batch = buf.sample_transitions(8)
        metrics = iql.train_step(batch)
        assert isinstance(metrics, TrainMetrics)
        assert metrics.loss > 0
        assert "v_loss" in metrics.extra
        assert "q_loss" in metrics.extra

    def test_multi_step_training(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        iql = IQL(replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64, device="cpu")
        all_metrics = iql.train(num_steps=10, batch_size=8, log_interval=5)
        assert len(all_metrics) == 10

    def test_get_action_values(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        iql = IQL(replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64, device="cpu")
        states = ["Task: do something\nStep 0: click(0,0)"]
        actions = ["click(100, 200)"]
        q_vals = iql.get_action_values(states, actions)
        assert q_vals.shape == (1,)

    def test_get_advantages(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        iql = IQL(replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64, device="cpu")
        states = ["Task: do X", "Task: do Y"]
        actions = ["click(1,2)", "type('hi')"]
        adv = iql.get_advantages(states, actions)
        assert adv.shape == (2,)

    def test_get_policy_weights(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        iql = IQL(replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64, device="cpu")
        # Train a few steps first
        iql.train(num_steps=5, batch_size=8, log_interval=100)
        states = ["Task: open file", "Task: close window"]
        actions = ["click(50, 50)", "hotkey('alt', 'F4')"]
        weights = iql.get_policy_weights(states, actions)
        assert weights.shape == (2,)
        assert torch.all(weights > 0)

    def test_save_load(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        iql = IQL(replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64, device="cpu")
        iql.train(num_steps=3, batch_size=8, log_interval=100)

        # Use BytesIO to avoid CJK path issues with torch C++ on Windows
        buffer = io.BytesIO()
        torch.save({
            "state_encoder": iql.state_encoder.state_dict(),
            "action_encoder": iql.action_encoder.state_dict(),
            "q1": iql.q1.state_dict(),
            "q2": iql.q2.state_dict(),
            "q1_target": iql.q1_target.state_dict(),
            "q2_target": iql.q2_target.state_dict(),
            "v": iql.v.state_dict(),
        }, buffer)
        buffer.seek(0)

        iql2 = IQL(replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64, device="cpu")
        ckpt = torch.load(buffer, map_location="cpu")
        iql2.state_encoder.load_state_dict(ckpt["state_encoder"])
        iql2.action_encoder.load_state_dict(ckpt["action_encoder"])
        iql2.q1.load_state_dict(ckpt["q1"])
        iql2.q2.load_state_dict(ckpt["q2"])
        iql2.q1_target.load_state_dict(ckpt["q1_target"])
        iql2.q2_target.load_state_dict(ckpt["q2_target"])
        iql2.v.load_state_dict(ckpt["v"])

        # Q-values should match after loading
        states = ["test state"]
        actions = ["test action"]
        q1 = iql.get_action_values(states, actions)
        q2 = iql2.get_action_values(states, actions)
        assert torch.allclose(q1, q2, atol=1e-5)


class TestCQL:
    """Test CQL algorithm on CPU."""

    def test_single_train_step(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        cql = CQL(replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64, device="cpu")
        batch = buf.sample_transitions(8)
        metrics = cql.train_step(batch)
        assert isinstance(metrics, TrainMetrics)
        assert "td_loss" in metrics.extra
        assert "cql_loss" in metrics.extra

    def test_multi_step_training(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        cql = CQL(replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64, device="cpu")
        all_metrics = cql.train(num_steps=10, batch_size=8, log_interval=5)
        assert len(all_metrics) == 10

    def test_cql_regularizer_nonzero(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        cql = CQL(replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64, alpha=5.0, device="cpu")
        batch = buf.sample_transitions(8)
        metrics = cql.train_step(batch)
        # CQL loss should be nonzero when alpha is large
        assert metrics.extra["cql_loss"] != 0.0

    def test_get_action_values(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        cql = CQL(replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64, device="cpu")
        q_vals = cql.get_action_values(["Task: test"], ["click(1,1)"])
        assert q_vals.shape == (1,)

    def test_save_load(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        cql = CQL(replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64, device="cpu")
        cql.train(num_steps=3, batch_size=8, log_interval=100)

        buffer = io.BytesIO()
        torch.save({
            "state_encoder": cql.state_encoder.state_dict(),
            "action_encoder": cql.action_encoder.state_dict(),
            "q1": cql.q1.state_dict(),
            "q2": cql.q2.state_dict(),
            "q1_target": cql.q1_target.state_dict(),
            "q2_target": cql.q2_target.state_dict(),
        }, buffer)
        buffer.seek(0)

        cql2 = CQL(replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64, device="cpu")
        ckpt = torch.load(buffer, map_location="cpu")
        cql2.state_encoder.load_state_dict(ckpt["state_encoder"])
        cql2.action_encoder.load_state_dict(ckpt["action_encoder"])
        cql2.q1.load_state_dict(ckpt["q1"])
        cql2.q2.load_state_dict(ckpt["q2"])
        cql2.q1_target.load_state_dict(ckpt["q1_target"])
        cql2.q2_target.load_state_dict(ckpt["q2_target"])

        q1 = cql.get_action_values(["state"], ["action"])
        q2 = cql2.get_action_values(["state"], ["action"])
        assert torch.allclose(q1, q2, atol=1e-5)


class TestAWAC:
    """Test AWAC algorithm on CPU."""

    def test_single_train_step(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        awac = AWAC(replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64, device="cpu")
        batch = buf.sample_transitions(8)
        metrics = awac.train_step(batch)
        assert isinstance(metrics, TrainMetrics)
        assert "critic_loss" in metrics.extra
        assert "actor_loss" in metrics.extra

    def test_multi_step_training(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        awac = AWAC(replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64, device="cpu")
        all_metrics = awac.train(num_steps=10, batch_size=8, log_interval=5)
        assert len(all_metrics) == 10

    def test_predict_actions(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        awac = AWAC(replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64, device="cpu")
        pred = awac.predict_actions(["Task: do something"])
        assert pred.shape == (1, 64)  # action_dim = 64

    def test_get_action_values(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        awac = AWAC(replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64, device="cpu")
        q_vals = awac.get_action_values(["Task: test"], ["click(1,1)"])
        assert q_vals.shape == (1,)

    def test_save_load(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        awac = AWAC(replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64, device="cpu")
        awac.train(num_steps=3, batch_size=8, log_interval=100)

        buffer = io.BytesIO()
        torch.save({
            "state_encoder": awac.state_encoder.state_dict(),
            "action_encoder": awac.action_encoder.state_dict(),
            "q1": awac.q1.state_dict(),
            "q2": awac.q2.state_dict(),
            "q1_target": awac.q1_target.state_dict(),
            "q2_target": awac.q2_target.state_dict(),
            "policy": awac.policy.state_dict(),
        }, buffer)
        buffer.seek(0)

        awac2 = AWAC(replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64, device="cpu")
        ckpt = torch.load(buffer, map_location="cpu")
        awac2.state_encoder.load_state_dict(ckpt["state_encoder"])
        awac2.action_encoder.load_state_dict(ckpt["action_encoder"])
        awac2.q1.load_state_dict(ckpt["q1"])
        awac2.q2.load_state_dict(ckpt["q2"])
        awac2.q1_target.load_state_dict(ckpt["q1_target"])
        awac2.q2_target.load_state_dict(ckpt["q2_target"])
        awac2.policy.load_state_dict(ckpt["policy"])

        q1 = awac.get_action_values(["state"], ["action"])
        q2 = awac2.get_action_values(["state"], ["action"])
        assert torch.allclose(q1, q2, atol=1e-5)


class TestTrainMetrics:
    """Test TrainMetrics helper."""

    def test_log_str(self):
        m = TrainMetrics(loss=0.5, extra={"v_loss": 0.3, "q_loss": 0.2})
        s = m.log_str()
        assert "loss=0.5000" in s
        assert "v_loss=0.3000" in s


class TestOffPolicyGRPO:
    """Test Off-Policy GRPO algorithm on CPU."""

    def test_single_train_step(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        grpo = OffPolicyGRPO(
            replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64,
            n_policy_updates=2, device="cpu",
        )
        batch = buf.sample_transitions(8)
        metrics = grpo.train_step(batch)
        assert isinstance(metrics, TrainMetrics)
        assert "surrogate_loss" in metrics.extra
        assert "kl_penalty" in metrics.extra
        assert "mean_advantage" in metrics.extra
        assert "mean_ratio" in metrics.extra
        assert "behavior_log_prob_coverage" in metrics.extra
        assert "behavior_fallback_fraction" in metrics.extra

    def test_multi_step_training(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        grpo = OffPolicyGRPO(
            replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64,
            n_policy_updates=2, device="cpu",
        )
        all_metrics = grpo.train(num_steps=10, batch_size=8, log_interval=5)
        assert len(all_metrics) == 10

    def test_get_action_values(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        grpo = OffPolicyGRPO(
            replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64,
            device="cpu",
        )
        vals = grpo.get_action_values(["Task: test"], ["click(1,1)"])
        assert vals.shape == (1,)

    def test_update_reference_policy(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        grpo = OffPolicyGRPO(
            replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64,
            n_policy_updates=2, device="cpu",
        )
        # Train a few steps to make current != ref
        grpo.train(num_steps=5, batch_size=8, log_interval=100)
        # Update reference policy
        grpo.update_reference_policy()
        # After update, ref should equal current
        for p1, p2 in zip(grpo.policy.parameters(), grpo.ref_policy.parameters()):
            assert torch.allclose(p1.data, p2.data)
            assert not p2.requires_grad  # ref should remain frozen

    def test_clip_ratio_effect(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        # Small clip = more conservative updates
        grpo_small = OffPolicyGRPO(
            replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64,
            clip_ratio=0.05, n_policy_updates=2, device="cpu",
        )
        grpo_large = OffPolicyGRPO(
            replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64,
            clip_ratio=0.5, n_policy_updates=2, device="cpu",
        )
        batch = buf.sample_transitions(8)
        m_small = grpo_small.train_step(batch)
        m_large = grpo_large.train_step(batch)
        # Both should produce valid metrics
        assert isinstance(m_small, TrainMetrics)
        assert isinstance(m_large, TrainMetrics)

    def test_prefers_behavior_log_probs_when_available(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        for transition in buf._transitions:
            transition.behavior_log_prob = -0.5

        grpo = OffPolicyGRPO(
            replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64,
            n_policy_updates=2, device="cpu",
        )
        batch = buf.sample_transitions(8)
        metrics = grpo.train_step(batch)

        assert metrics.extra["behavior_log_prob_coverage"] == 1.0
        assert metrics.extra["behavior_fallback_fraction"] == 0.0

    def test_save_load(self, tmp_dir):
        buf = _create_buffer_with_data(tmp_dir)
        grpo = OffPolicyGRPO(
            replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64,
            device="cpu",
        )
        grpo.train(num_steps=3, batch_size=8, log_interval=100)

        buffer = io.BytesIO()
        torch.save({
            "state_encoder": grpo.state_encoder.state_dict(),
            "action_encoder": grpo.action_encoder.state_dict(),
            "policy": grpo.policy.state_dict(),
            "ref_policy": grpo.ref_policy.state_dict(),
        }, buffer)
        buffer.seek(0)

        grpo2 = OffPolicyGRPO(
            replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64,
            device="cpu",
        )
        ckpt = torch.load(buffer, map_location="cpu")
        grpo2.state_encoder.load_state_dict(ckpt["state_encoder"])
        grpo2.action_encoder.load_state_dict(ckpt["action_encoder"])
        grpo2.policy.load_state_dict(ckpt["policy"])
        grpo2.ref_policy.load_state_dict(ckpt["ref_policy"])

        v1 = grpo.get_action_values(["state"], ["action"])
        v2 = grpo2.get_action_values(["state"], ["action"])
        assert torch.allclose(v1, v2, atol=1e-5)


def _create_grouped_buffer(tmp_dir: str, n_instructions: int = 5, trajs_per_inst: int = 4) -> ReplayBuffer:
    """Create a buffer where multiple trajectories share the same instruction."""
    path = os.path.join(tmp_dir, "grouped_store.jsonl")
    store = TrajectoryStore(path)
    idx = 0
    for inst_idx in range(n_instructions):
        for t_idx in range(trajs_per_inst):
            traj = _make_trajectory(
                traj_id=f"g{inst_idx}_t{t_idx}",
                domain="os",
                n_steps=3 + t_idx % 3,
                success=(t_idx % 2 == 0),
            )
            # Override instruction so multiple trajectories share the same one
            traj.instruction = f"Shared task {inst_idx}"
            store.append(traj)
            idx += 1
    buf = ReplayBuffer(store=store, max_trajectories=idx)
    buf.load_from_store()
    return buf


class TestGRPOGroupSampling:
    """Test GRPO group-relative advantage computation."""

    def test_sample_transition_groups_basic(self, tmp_dir):
        buf = _create_grouped_buffer(tmp_dir)
        batch = buf.sample_transition_groups(n_groups=3, min_group_size=2)
        assert batch.group_ids is not None
        unique_groups = set(batch.group_ids.tolist())
        assert len(unique_groups) == 3

    def test_sample_transition_groups_all_same_group(self, tmp_dir):
        buf = _create_grouped_buffer(tmp_dir)
        batch = buf.sample_transition_groups(n_groups=3, min_group_size=2)
        # Within each group, all instructions should be the same
        for gid in set(batch.group_ids.tolist()):
            mask = batch.group_ids == gid
            group_instructions = [
                batch.instructions[i] for i, m in enumerate(mask) if m
            ]
            assert len(set(group_instructions)) == 1

    def test_group_advantages_differ_from_batch(self, tmp_dir):
        buf = _create_grouped_buffer(tmp_dir)
        grpo = OffPolicyGRPO(
            replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64,
            device="cpu",
        )
        batch = buf.sample_transition_groups(n_groups=3, min_group_size=2)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32)
        group_ids = torch.as_tensor(batch.group_ids, dtype=torch.long)

        adv_grouped = grpo._compute_grpo_advantages(rewards, group_ids=group_ids)
        adv_batch = grpo._compute_grpo_advantages(rewards, group_ids=None)

        # With diverse groups, group-relative advantages should differ
        assert adv_grouped.shape == adv_batch.shape
        # They shouldn't be identical (unless by sheer coincidence)
        assert not torch.allclose(adv_grouped, adv_batch, atol=1e-3)

    def test_group_advantages_zero_mean_per_group(self, tmp_dir):
        buf = _create_grouped_buffer(tmp_dir)
        grpo = OffPolicyGRPO(
            replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64,
            device="cpu",
        )
        batch = buf.sample_transition_groups(n_groups=3, min_group_size=2)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32)
        group_ids = torch.as_tensor(batch.group_ids, dtype=torch.long)

        advantages = grpo._compute_grpo_advantages(rewards, group_ids=group_ids)

        # Each group's advantages should have zero mean
        for gid in group_ids.unique():
            mask = group_ids == gid
            group_adv = advantages[mask]
            assert abs(group_adv.mean().item()) < 1e-5

    def test_train_with_group_sampling(self, tmp_dir):
        buf = _create_grouped_buffer(tmp_dir)
        grpo = OffPolicyGRPO(
            replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64,
            n_policy_updates=2, device="cpu",
        )
        all_metrics = grpo.train(num_steps=5, batch_size=8, n_groups=3, log_interval=100)
        assert len(all_metrics) == 5
        for m in all_metrics:
            assert isinstance(m, TrainMetrics)
            assert "surrogate_loss" in m.extra

    def test_fallback_to_batch_when_no_groups(self, tmp_dir):
        """With unique instructions per trajectory, group sampling falls back gracefully."""
        buf = _create_buffer_with_data(tmp_dir)
        grpo = OffPolicyGRPO(
            replay_buffer=buf, state_dim=64, action_dim=64, hidden_dim=64,
            n_policy_updates=2, device="cpu",
        )
        # This should not raise — falls back to batch-level normalization
        all_metrics = grpo.train(num_steps=3, batch_size=8, n_groups=4, log_interval=100)
        assert len(all_metrics) == 3
