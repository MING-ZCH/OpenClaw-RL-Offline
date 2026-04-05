"""
Tests for the openclaw-offline integration module.

Since the real slime framework requires GPU / Megatron-LM which is not
available on this dev machine, these tests mock the slime types (via conftest)
and verify the offline integration logic works correctly.
"""

import json
import os
import sys

import pytest
import torch

# conftest.py installs slime mocks and adds paths
from conftest import MockSample, MockRolloutFnTrainOutput, MockRolloutFnEvalOutput

import offline_rollout
import offline_loss


# ---------- Tests for offline_rollout.py ----------

class TestOfflineRollout:
    def setup_method(self):
        offline_rollout.reset_data_source()

    def test_generate_rollout_returns_train_output(self, mock_args):
        result = offline_rollout.generate_rollout_offline(
            mock_args, rollout_id=1, data_buffer=None, evaluation=False
        )
        assert isinstance(result, MockRolloutFnTrainOutput)
        assert isinstance(result.samples, list)
        assert len(result.samples) > 0
        assert isinstance(result.metrics, dict)

    def test_samples_are_grouped(self, mock_args):
        result = offline_rollout.generate_rollout_offline(
            mock_args, rollout_id=1, data_buffer=None
        )
        for group in result.samples:
            assert isinstance(group, list)
            for sample in group:
                assert isinstance(sample, MockSample)

    def test_metrics_keys(self, mock_args):
        result = offline_rollout.generate_rollout_offline(
            mock_args, rollout_id=1, data_buffer=None
        )
        assert "offline/num_groups" in result.metrics
        assert "offline/avg_reward" in result.metrics
        assert "offline/replay_time" in result.metrics
        assert "offline/pool_size" in result.metrics

    def test_evaluation_returns_eval_output(self, mock_args):
        result = offline_rollout.generate_rollout_offline(
            mock_args, rollout_id=1, data_buffer=None, evaluation=True
        )
        assert isinstance(result, MockRolloutFnEvalOutput)
        assert "offline/eval_skip" in result.metrics

    def test_sample_conversion_metadata(self, mock_args):
        result = offline_rollout.generate_rollout_offline(
            mock_args, rollout_id=1, data_buffer=None
        )
        for group in result.samples:
            for sample in group:
                assert "offline_replay" in sample.metadata
                assert sample.metadata["offline_replay"] is True

    def test_env_var_config(self, trajectory_jsonl):
        """Test configuration via environment variables."""
        os.environ["OFFLINE_TRAJECTORY_STORE"] = trajectory_jsonl
        os.environ["OFFLINE_MODE"] = "trajectory"
        os.environ["OFFLINE_N_SAMPLES_PER_PROMPT"] = "2"

        class EmptyArgs:
            rollout_max_response_len = 128
            rollout_batch_size = 2

        result = offline_rollout.generate_rollout_offline(
            EmptyArgs(), rollout_id=0, data_buffer=None
        )
        assert isinstance(result, MockRolloutFnTrainOutput)

    def test_missing_store_raises_error(self):
        class EmptyArgs:
            offline_trajectory_store = None
            rollout_max_response_len = 128

        with pytest.raises(ValueError, match="No trajectory store"):
            offline_rollout.generate_rollout_offline(
                EmptyArgs(), rollout_id=0, data_buffer=None
            )

    def test_invalid_mode_raises_error(self, trajectory_jsonl):
        """Invalid OFFLINE_MODE should raise ValueError."""
        os.environ["OFFLINE_TRAJECTORY_STORE"] = trajectory_jsonl
        os.environ["OFFLINE_MODE"] = "invalid_mode"

        class EmptyArgs:
            rollout_max_response_len = 128
            rollout_batch_size = 2

        with pytest.raises(ValueError, match="Invalid OFFLINE_MODE"):
            offline_rollout.generate_rollout_offline(
                EmptyArgs(), rollout_id=0, data_buffer=None
            )

    def test_consecutive_rollout_calls_cached(self, mock_args):
        """Multiple calls should reuse the cached data source."""
        r1 = offline_rollout.generate_rollout_offline(
            mock_args, rollout_id=1, data_buffer=None
        )
        r2 = offline_rollout.generate_rollout_offline(
            mock_args, rollout_id=2, data_buffer=None
        )
        assert isinstance(r1, MockRolloutFnTrainOutput)
        assert isinstance(r2, MockRolloutFnTrainOutput)
        assert r1.metrics["offline/pool_size"] == r2.metrics["offline/pool_size"]

    def test_varying_batch_sizes(self, trajectory_jsonl):
        """Different batch sizes should work correctly."""
        os.environ["OFFLINE_TRAJECTORY_STORE"] = trajectory_jsonl

        for batch_size in [1, 2, 8]:
            offline_rollout.reset_data_source()

            class Args:
                rollout_max_response_len = 128
            Args.rollout_batch_size = batch_size

            result = offline_rollout.generate_rollout_offline(
                Args(), rollout_id=0, data_buffer=None
            )
            assert isinstance(result, MockRolloutFnTrainOutput)
            assert result.metrics["offline/num_groups"] <= batch_size

    def test_reset_data_source(self, mock_args):
        """reset_data_source should clear the cache."""
        offline_rollout.generate_rollout_offline(
            mock_args, rollout_id=1, data_buffer=None
        )
        assert offline_rollout._global_data_source is not None

        offline_rollout.reset_data_source()
        assert offline_rollout._global_data_source is None

    def test_rollout_id_in_metrics(self, mock_args):
        """Rollout ID should be tracked in metrics."""
        result = offline_rollout.generate_rollout_offline(
            mock_args, rollout_id=42, data_buffer=None
        )
        assert result.metrics["offline/rollout_id"] == 42.0

    def test_avg_reward_reasonable(self, mock_args):
        """Average reward should be a valid number."""
        result = offline_rollout.generate_rollout_offline(
            mock_args, rollout_id=1, data_buffer=None
        )
        avg_reward = result.metrics["offline/avg_reward"]
        assert isinstance(avg_reward, float)
        assert not (avg_reward != avg_reward)  # not NaN


# ---------- Tests for offline_loss.py ----------

class TestOfflineLoss:
    def setup_method(self):
        offline_loss.reset_weight_cache()

    def test_loss_without_weights(self):
        """Without weight file, should use standard PPO loss."""
        batch_size = 4
        seq_len = 8
        log_probs = torch.randn(batch_size, seq_len)
        old_log_probs = torch.randn(batch_size, seq_len)
        advantages = torch.randn(batch_size, seq_len)
        loss_masks = torch.ones(batch_size, seq_len)
        samples = [MockSample() for _ in range(batch_size)]

        loss, metrics = offline_loss.advantage_weighted_loss_function(
            args=None,
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            loss_masks=loss_masks,
            samples=samples,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # scalar
        assert "offline_loss/clipped_ratio" in metrics

    def test_loss_with_weight_file(self, weight_file):
        """With weight file, advantages should be modulated."""
        os.environ["OFFLINE_WEIGHT_PATH"] = weight_file

        batch_size = 3
        seq_len = 5
        log_probs = torch.zeros(batch_size, seq_len)
        old_log_probs = torch.zeros(batch_size, seq_len)
        advantages = torch.ones(batch_size, seq_len)
        loss_masks = torch.ones(batch_size, seq_len)

        samples = []
        for i in range(batch_size):
            s = MockSample()
            s.metadata = {"trajectory_id": "traj_{}".format(i), "step_idx": 0}
            samples.append(s)

        loss, metrics = offline_loss.advantage_weighted_loss_function(
            args=None,
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            loss_masks=loss_masks,
            samples=samples,
        )

        assert isinstance(loss, torch.Tensor)
        assert "offline_loss/mean_weight" in metrics
        assert metrics["offline_loss/mean_weight"] > 0

    def test_loss_clipping(self):
        """Verify PPO clipping works correctly."""
        batch_size = 2
        seq_len = 4
        log_probs = torch.ones(batch_size, seq_len) * 2.0
        old_log_probs = torch.zeros(batch_size, seq_len)
        advantages = torch.ones(batch_size, seq_len)
        loss_masks = torch.ones(batch_size, seq_len)
        samples = [MockSample() for _ in range(batch_size)]

        loss, metrics = offline_loss.advantage_weighted_loss_function(
            args=None,
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            loss_masks=loss_masks,
            samples=samples,
        )

        assert loss.item() < 0
        assert metrics["offline_loss/clipped_ratio"] > 0

    def test_empty_log_probs(self):
        """Empty input should return zero loss and valid metrics."""
        loss, metrics = offline_loss.advantage_weighted_loss_function(
            args=None,
            log_probs=[],
            old_log_probs=[],
            advantages=[],
            loss_masks=[],
            samples=[],
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.item() == 0.0
        assert metrics["offline_loss/pg_loss"] == 0.0

    def test_all_zero_masks(self):
        """All-zero loss masks should produce zero loss."""
        batch_size = 3
        seq_len = 5
        log_probs = torch.randn(batch_size, seq_len)
        old_log_probs = torch.randn(batch_size, seq_len)
        advantages = torch.ones(batch_size, seq_len)
        loss_masks = torch.zeros(batch_size, seq_len)
        samples = [MockSample() for _ in range(batch_size)]

        loss, metrics = offline_loss.advantage_weighted_loss_function(
            args=None,
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            loss_masks=loss_masks,
            samples=samples,
        )

        assert loss.item() == 0.0
        assert metrics["offline_loss/n_tokens"] == 0.0

    def test_negative_advantages(self):
        """Should handle negative advantages correctly."""
        batch_size = 2
        seq_len = 4
        log_probs = torch.zeros(batch_size, seq_len)
        old_log_probs = torch.zeros(batch_size, seq_len)
        advantages = -torch.ones(batch_size, seq_len)
        loss_masks = torch.ones(batch_size, seq_len)
        samples = [MockSample() for _ in range(batch_size)]

        loss, metrics = offline_loss.advantage_weighted_loss_function(
            args=None,
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            loss_masks=loss_masks,
            samples=samples,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0

    def test_kl_penalty(self):
        """KL penalty should be tracked when enabled."""
        os.environ["OFFLINE_KL_COEFF"] = "0.5"

        batch_size = 3
        seq_len = 5
        log_probs = torch.ones(batch_size, seq_len) * 0.5
        old_log_probs = torch.zeros(batch_size, seq_len)
        advantages = torch.ones(batch_size, seq_len)
        loss_masks = torch.ones(batch_size, seq_len)
        samples = [MockSample() for _ in range(batch_size)]

        loss, metrics = offline_loss.advantage_weighted_loss_function(
            args=None,
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            loss_masks=loss_masks,
            samples=samples,
        )

        assert "offline_loss/kl_penalty" in metrics
        assert "offline_loss/kl_coeff" in metrics
        assert metrics["offline_loss/kl_coeff"] == 0.5

    def test_extreme_weight_values(self, tmp_path):
        """Extreme advantage values should be clamped."""
        extreme = {"traj_0:0": 100.0, "traj_1:0": -100.0}
        path = str(tmp_path / "extreme_weights.json")
        with open(path, "w") as f:
            json.dump(extreme, f)

        os.environ["OFFLINE_WEIGHT_PATH"] = path

        batch_size = 2
        seq_len = 3
        log_probs = torch.zeros(batch_size, seq_len)
        old_log_probs = torch.zeros(batch_size, seq_len)
        advantages = torch.ones(batch_size, seq_len)
        loss_masks = torch.ones(batch_size, seq_len)

        samples = []
        for i in range(batch_size):
            s = MockSample()
            s.metadata = {"trajectory_id": "traj_{}".format(i), "step_idx": 0}
            samples.append(s)

        loss, metrics = offline_loss.advantage_weighted_loss_function(
            args=None,
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            loss_masks=loss_masks,
            samples=samples,
        )

        assert not torch.isinf(loss)
        assert not torch.isnan(loss)
        assert metrics["offline_loss/mean_weight"] < 1e10

    def test_reset_weight_cache(self, weight_file):
        """reset_weight_cache should clear cached weights."""
        os.environ["OFFLINE_WEIGHT_PATH"] = weight_file
        offline_loss._load_advantage_weights()
        assert offline_loss._cached_weights is not None

        offline_loss.reset_weight_cache()
        assert offline_loss._cached_weights is None

    def test_loss_gradient_flows(self):
        """Loss should have gradients for backprop."""
        batch_size = 2
        seq_len = 4
        log_probs = torch.randn(batch_size, seq_len, requires_grad=True)
        old_log_probs = torch.randn(batch_size, seq_len)
        advantages = torch.ones(batch_size, seq_len)
        loss_masks = torch.ones(batch_size, seq_len)
        samples = [MockSample() for _ in range(batch_size)]

        loss, _ = offline_loss.advantage_weighted_loss_function(
            args=None,
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            loss_masks=loss_masks,
            samples=samples,
        )

        loss.backward()
        assert log_probs.grad is not None
        assert not torch.all(log_probs.grad == 0)


# ---------- Tests for compute_weights.py ----------

class TestComputeWeights:
    def test_cli_iql(self, trajectory_jsonl, tmp_path):
        """End-to-end test of compute_weights with IQL."""
        output = str(tmp_path / "weights.json")
        sys.argv = [
            "compute_weights.py",
            "--data", trajectory_jsonl,
            "--output", output,
            "--algo", "iql",
            "--train-steps", "10",
            "--batch-size", "4",
        ]

        import compute_weights
        compute_weights.main()

        assert os.path.exists(output)
        with open(output) as f:
            weights = json.load(f)
        assert len(weights) == 15
        for v in weights.values():
            assert isinstance(v, float)
            assert v == v  # not NaN

    def test_cli_cql(self, trajectory_jsonl, tmp_path):
        """End-to-end test of compute_weights with CQL."""
        output = str(tmp_path / "cql_weights.json")
        sys.argv = [
            "compute_weights.py",
            "--data", trajectory_jsonl,
            "--output", output,
            "--algo", "cql",
            "--train-steps", "10",
            "--batch-size", "4",
        ]

        import compute_weights
        compute_weights.main()

        assert os.path.exists(output)
        with open(output) as f:
            weights = json.load(f)
        assert len(weights) == 15

    def test_cli_retrospex(self, trajectory_jsonl, tmp_path):
        """Retrospex uses IQL twin-Q/V; get_action_values() returns raw Q proxy."""
        output = str(tmp_path / "retrospex_weights.json")
        sys.argv = [
            "compute_weights.py",
            "--data", trajectory_jsonl,
            "--output", output,
            "--algo", "retrospex",
            "--train-steps", "10",
            "--batch-size", "4",
            "--retrospex-tau", "0.7",
            "--retrospex-lambda-scale", "1.0",
        ]

        import compute_weights
        compute_weights.main()

        assert os.path.exists(output)
        with open(output) as f:
            weights = json.load(f)
        # 5 trajectories * 3 steps = 15 entries
        assert len(weights) == 15
        for key, val in weights.items():
            assert ":" in key, "Keys should be 'traj_id:step_idx'"
            assert isinstance(val, float)
            assert val == val  # not NaN

    def test_cli_webrl_produces_bounded_values(self, trajectory_jsonl, tmp_path):
        """WebRL ORM produces P(success) ∈ [0, 1] as advantage proxy."""
        output = str(tmp_path / "webrl_weights.json")
        sys.argv = [
            "compute_weights.py",
            "--data", trajectory_jsonl,
            "--output", output,
            "--algo", "webrl",
            "--train-steps", "10",
            "--batch-size", "4",
            "--webrl-alpha-orm", "0.5",
        ]

        import compute_weights
        compute_weights.main()

        assert os.path.exists(output)
        with open(output) as f:
            weights = json.load(f)
        assert len(weights) == 15
        # ORM values should be in [0, 1] (sigmoid output)
        for val in weights.values():
            assert -0.01 <= val <= 1.01, "WebRL ORM value out of [0,1]: {}".format(val)

    def test_cli_glider_uses_plan_conditioned_q(self, trajectory_jsonl, tmp_path):
        """GLIDER plan-conditioned Q: get_action_values() internally encodes plan."""
        output = str(tmp_path / "glider_weights.json")
        sys.argv = [
            "compute_weights.py",
            "--data", trajectory_jsonl,
            "--output", output,
            "--algo", "glider",
            "--train-steps", "10",
            "--batch-size", "4",
            "--glider-plan-dim", "32",
            "--glider-beta", "1.0",
            "--glider-tau", "0.7",
        ]

        import compute_weights
        compute_weights.main()

        assert os.path.exists(output)
        with open(output) as f:
            weights = json.load(f)
        assert len(weights) == 15
        for val in weights.values():
            assert isinstance(val, float)
            assert val == val  # not NaN

    def test_advantage_dispatch_has_advantages_vs_q_value(self):
        """_HAS_ADVANTAGES set should exactly match algos with get_advantages() method."""
        import compute_weights as cw
        assert "iql" in cw._HAS_ADVANTAGES
        assert "awac" in cw._HAS_ADVANTAGES
        assert "crr" in cw._HAS_ADVANTAGES
        assert "edac" in cw._HAS_ADVANTAGES
        assert "oreo" in cw._HAS_ADVANTAGES
        # Algorithms that use get_action_values() should NOT be in _HAS_ADVANTAGES
        for qval_algo in ("cql", "td3bc", "grpo", "sorl", "arpo", "retrospex", "glider", "webrl"):
            assert qval_algo not in cw._HAS_ADVANTAGES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
