"""Tests for train_offline device selection and lightweight launcher wiring."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import pytest

from offline_rl.data.replay_buffer import ReplayBuffer
from offline_rl.data.trajectory_store import TrajectoryStore
from tests.conftest import _make_trajectory


SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

TRAIN_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "train_offline.py")
TRAIN_SCRIPT_SPEC = importlib.util.spec_from_file_location("offline_train_script", TRAIN_SCRIPT_PATH)
assert TRAIN_SCRIPT_SPEC is not None and TRAIN_SCRIPT_SPEC.loader is not None
train_script = importlib.util.module_from_spec(TRAIN_SCRIPT_SPEC)
TRAIN_SCRIPT_SPEC.loader.exec_module(train_script)


def _create_buffer(tmp_dir: str, n_trajs: int = 8) -> ReplayBuffer:
    path = os.path.join(tmp_dir, "train_script_store.jsonl")
    store = TrajectoryStore(path)
    for i in range(n_trajs):
        store.append(_make_trajectory(
            traj_id="script_traj_%d" % i,
            domain="os",
            n_steps=3 + (i % 3),
            success=(i % 2 == 0),
        ))
    buf = ReplayBuffer(store=store, max_trajectories=n_trajs)
    buf.load_from_store()
    return buf


def _make_args(algo: str) -> argparse.Namespace:
    return argparse.Namespace(
        algo=algo,
        state_dim=64,
        action_dim=64,
        hidden_dim=64,
        lr=3e-4,
        gamma=0.99,
        tau=0.7,
        beta=3.0,
        alpha=1.0,
        lam=1.0,
        clip_ratio=0.2,
        kl_coeff=0.01,
        n_policy_updates=2,
        td3bc_alpha=2.5,
        td3bc_target_noise=0.2,
        td3bc_noise_clip=0.5,
        td3bc_policy_freq=2,
    )


def test_resolve_device_auto_prefers_cpu_when_cuda_missing(monkeypatch):
    monkeypatch.setattr(train_script.torch.cuda, "is_available", lambda: False)
    assert train_script._resolve_device("auto") == "cpu"


def test_resolve_device_accepts_visible_cuda(monkeypatch):
    monkeypatch.setattr(train_script.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train_script.torch.cuda, "device_count", lambda: 2)
    assert train_script._resolve_device("auto") == "cuda"
    assert train_script._resolve_device("cuda:1") == "cuda:1"


def test_resolve_device_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(train_script.torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError):
        train_script._resolve_device("cuda")


def test_build_algorithm_uses_requested_device(tmp_dir):
    buf = _create_buffer(tmp_dir)
    algo = train_script._build_algorithm(_make_args("grpo"), buf, device="cpu")
    assert str(algo.device) == "cpu"


def test_build_td3bc_trains_and_returns_q_values(tmp_dir):
    """TD3+BC creates, trains, and returns Q-values without errors."""
    buf = _create_buffer(tmp_dir, n_trajs=10)
    args = _make_args("td3bc")
    algo = train_script._build_algorithm(args, buf, device="cpu")
    assert str(algo.device) == "cpu"
    metrics = algo.train(num_steps=10, batch_size=8, log_interval=20)
    assert len(metrics) == 10
    assert all(hasattr(m, "loss") for m in metrics)
    q_vals = algo.get_action_values(["state A", "state B"], ["action A", "action B"])
    assert q_vals.shape == (2,)


def test_td3bc_delayed_actor_update(tmp_dir):
    """Actor loss is non-zero only on even steps (policy_freq=2)."""
    buf = _create_buffer(tmp_dir, n_trajs=10)
    args = _make_args("td3bc")
    algo = train_script._build_algorithm(args, buf, device="cpu")
    # Step 1: no actor update expected
    from offline_rl.data.replay_buffer import ReplayBuffer
    batch = buf.sample_transitions(8)
    metrics1 = algo.train_step(batch)
    assert metrics1.extra["actor_loss"] == 0.0  # step 1, no actor update
    # Step 2: actor update expected
    metrics2 = algo.train_step(batch)
    assert metrics2.extra["actor_loss"] != 0.0  # step 2, actor updates