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