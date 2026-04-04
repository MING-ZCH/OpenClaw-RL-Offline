"""Tests for OfflineDataSource."""

from __future__ import annotations

import os

from offline_rl.data.offline_data_source import OfflineDataSource, SampleLite
from offline_rl.data.trajectory_store import TrajectoryStore
from tests.conftest import _make_trajectory


class TestSampleLite:
    """Test SampleLite dataclass."""

    def test_default_values(self):
        s = SampleLite()
        assert s.idx == 0
        assert s.prompt == ""
        assert s.tokens == []
        assert isinstance(s.reward, dict)
        assert s.status == "completed"
        assert s.group_index is None
        assert s.index is None
        assert s.response_length == 0
        assert s.multimodal_inputs is None
        assert s.rollout_log_probs is None
        assert s.remove_sample is False

    def test_custom_values(self):
        s = SampleLite(
            idx=5, prompt="Task: test", response="I click",
            tokens=[1, 2, 3], loss_mask=[0, 1, 1],
            reward={"score": 1.0}, status="failed",
            group_index=0, index=1,
        )
        assert s.idx == 5
        assert len(s.tokens) == 3
        assert s.group_index == 0

    def test_reward_accepts_float(self):
        s = SampleLite(reward=1.0)
        assert s.reward == 1.0

    def test_reward_accepts_none(self):
        s = SampleLite(reward=None)
        assert s.reward is None

    def test_to_slime_sample_import_error(self):
        s = SampleLite(idx=1, prompt="test")
        try:
            s.to_slime_sample()
            assert False, "Should have raised ImportError (slime not installed)"
        except ImportError:
            pass

    def test_from_slime_sample_classmethod(self):
        """Verify from_slime_sample is a classmethod."""
        assert hasattr(SampleLite, "from_slime_sample")
        assert callable(SampleLite.from_slime_sample)


class TestOfflineDataSourceStepMode:
    """Test step mode (each step → one sample)."""

    def test_build_samples(self, populated_store):
        ds = OfflineDataSource(populated_store, mode="step", shuffle=False)
        assert len(ds) > 0

    def test_get_samples(self, populated_store):
        ds = OfflineDataSource(populated_store, mode="step")
        groups = ds.get_samples(4)
        assert len(groups) == 4
        for group in groups:
            assert len(group) == 1  # n_samples_per_prompt defaults to 1
            assert isinstance(group[0], SampleLite)

    def test_n_samples_per_prompt(self, populated_store):
        ds = OfflineDataSource(populated_store, mode="step", n_samples_per_prompt=3)
        groups = ds.get_samples(2)
        assert len(groups) == 2
        for group in groups:
            assert len(group) == 3

    def test_sample_fields(self, populated_store):
        ds = OfflineDataSource(populated_store, mode="step", shuffle=False)
        groups = ds.get_samples(1)
        sample = groups[0][0]
        assert sample.prompt != ""
        assert sample.response != ""
        assert len(sample.tokens) > 0
        assert len(sample.loss_mask) > 0
        assert len(sample.loss_mask) == len(sample.tokens)
        assert "score" in sample.reward
        assert "step_reward" in sample.reward

    def test_wrap_around(self, populated_store):
        ds = OfflineDataSource(populated_store, mode="step")
        total = len(ds)
        # Request more than available → should wrap around
        groups = ds.get_samples(total + 5)
        assert len(groups) == total + 5


class TestOfflineDataSourceTrajectoryMode:
    """Test trajectory mode (each trajectory → one sample)."""

    def test_build_samples(self, populated_store):
        ds = OfflineDataSource(populated_store, mode="trajectory", shuffle=False)
        # 10 trajectories → 10 samples
        assert len(ds) == 10

    def test_sample_has_full_response(self, populated_store):
        ds = OfflineDataSource(populated_store, mode="trajectory", shuffle=False)
        groups = ds.get_samples(1)
        sample = groups[0][0]
        assert "Step 0:" in sample.response
        assert "outcome" in sample.reward


class TestOfflineDataSourceDynamicHistory:
    """Test dynamic_history mode."""

    def test_build_samples(self, populated_store):
        ds = OfflineDataSource(populated_store, mode="dynamic_history", shuffle=False)
        assert len(ds) > 10  # More samples than trajectories

    def test_history_grows(self, populated_store):
        ds = OfflineDataSource(populated_store, mode="dynamic_history", shuffle=False)
        groups = ds.get_samples(len(ds))
        # Find samples from same trajectory
        traj_id = groups[0][0].trajectory_id
        same_traj = [g[0] for g in groups if g[0].trajectory_id == traj_id]
        if len(same_traj) > 1:
            # Later steps should have longer prompts (more history)
            assert len(same_traj[-1].prompt) >= len(same_traj[0].prompt)


class TestOfflineDataSourceInterface:
    """Test slime-compatible interface methods."""

    def test_add_samples(self, populated_store):
        ds = OfflineDataSource(populated_store, mode="step")
        original_len = len(ds)
        # New signature: list[list[SampleLite]] (groups of samples)
        ds.add_samples([[SampleLite(idx=999, prompt="new")]])
        assert len(ds) == original_len + 1

    def test_add_samples_multi_group(self, populated_store):
        ds = OfflineDataSource(populated_store, mode="step")
        original_len = len(ds)
        ds.add_samples([
            [SampleLite(idx=100, prompt="a"), SampleLite(idx=101, prompt="b")],
            [SampleLite(idx=102, prompt="c")],
        ])
        assert len(ds) == original_len + 3

    def test_save_load(self, populated_store, tmp_dir):
        ds = OfflineDataSource(populated_store, mode="step", save_dir=tmp_dir)
        ds.get_samples(5)  # Advance cursor
        ds.save("rollout_01")  # slime-compatible signature
        import os
        assert os.path.exists(os.path.join(tmp_dir, "rollout_01", "offline_data_source_state.json"))

        ds2 = OfflineDataSource(populated_store, mode="step", save_dir=tmp_dir)
        ds2.load("rollout_01")

    def test_save_load_default_id(self, populated_store, tmp_dir):
        ds = OfflineDataSource(populated_store, mode="step", save_dir=tmp_dir)
        ds.save()  # no rollout_id → uses "default"
        ds2 = OfflineDataSource(populated_store, mode="step", save_dir=tmp_dir)
        ds2.load()  # loads from "default"

    def test_repr(self, populated_store):
        ds = OfflineDataSource(populated_store, mode="step", n_samples_per_prompt=4)
        r = repr(ds)
        assert "step" in r
        assert "n_per_prompt=4" in r

    def test_invalid_mode_raises(self, populated_store):
        try:
            OfflineDataSource(populated_store, mode="invalid")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "invalid" in str(e)
