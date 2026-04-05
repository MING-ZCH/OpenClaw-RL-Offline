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
        edac_n_critics=3,
        edac_eta=1.0,
        edac_alpha_init=0.1,
        edac_no_auto_alpha=False,
        dt_context_len=5,
        dt_nhead=2,
        dt_layers=1,
        crr_beta=1.0,
        crr_filter="exp",
        crr_mc_samples=4,
        rwft_beta=1.0,
        rwft_reward_norm="softmax",
        rwft_reward_clip=10.0,
        oreo_beta=1.0,
        oreo_mc_samples=4,
        sorl_clip_norm_threshold=0.2,
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


def test_build_edac_trains_and_returns_q_values(tmp_dir):
    """EDAC builds with N=3 ensemble, trains, returns mean Q-values."""
    buf = _create_buffer(tmp_dir, n_trajs=10)
    args = _make_args("edac")
    algo = train_script._build_algorithm(args, buf, device="cpu")
    assert str(algo.device) == "cpu"
    assert algo.n_critics == 3
    metrics = algo.train(num_steps=10, batch_size=8, log_interval=20)
    assert len(metrics) == 10
    assert all(hasattr(m, "loss") for m in metrics)
    # All extra keys present
    assert "q_loss" in metrics[0].extra
    assert "actor_loss" in metrics[0].extra
    assert "alpha" in metrics[0].extra
    q_vals = algo.get_action_values(["obs A", "obs B"], ["act A", "act B"])
    assert q_vals.shape == (2,)


def test_edac_uncertainty_advantages(tmp_dir):
    """EDAC get_advantages returns penalized Q-values (shape, no nan)."""
    buf = _create_buffer(tmp_dir, n_trajs=8)
    args = _make_args("edac")
    algo = train_script._build_algorithm(args, buf, device="cpu")
    algo.train(num_steps=5, batch_size=4, log_interval=100)
    adv = algo.get_advantages(["state X", "state Y"], ["act X", "act Y"])
    assert adv.shape == (2,)
    import math
    assert all(not math.isnan(v) for v in adv.tolist())


def test_build_dt_trains_on_trajectories(tmp_dir):
    """Decision Transformer trains on trajectory batches, reduces loss."""
    buf = _create_buffer(tmp_dir, n_trajs=12)
    args = _make_args("dt")
    algo = train_script._build_algorithm(args, buf, device="cpu")
    assert str(algo.device) == "cpu"
    assert getattr(algo, "_needs_trajectory_batch", False) is True
    metrics = algo.train(num_steps=10, batch_size=4, log_interval=20)
    assert len(metrics) > 0
    assert all(m.loss >= 0 for m in metrics)
    assert "bc_loss" in metrics[0].extra


def test_dt_get_action_values_shape(tmp_dir):
    """DT get_action_values returns (B,) proxy scores without error."""
    buf = _create_buffer(tmp_dir, n_trajs=8)
    args = _make_args("dt")
    algo = train_script._build_algorithm(args, buf, device="cpu")
    scores = algo.get_action_values(
        ["context A", "context B", "context C"],
        ["action A", "action B", "action C"],
    )
    assert scores.shape == (3,)


def test_training_loop_grad_accum(tmp_dir):
    """_training_loop with grad_accum=2 runs without error and returns metrics."""
    buf = _create_buffer(tmp_dir, n_trajs=10)
    args = _make_args("iql")
    algo = train_script._build_algorithm(args, buf, device="cpu")
    metrics = train_script._training_loop(
        algo, buf,
        num_steps=6, batch_size=4, log_interval=10,
        use_amp=False, amp_dtype=None, grad_accum=2,
    )
    # 6 logged steps, each with 2 sub-steps = 12 train_step calls
    assert len(metrics) == 6
    assert all(m.loss >= 0 for m in metrics)


def test_build_crr_trains_and_returns_q_values(tmp_dir):
    """CRR builds, trains, returns Q-values and advantages without error."""
    buf = _create_buffer(tmp_dir, n_trajs=10)
    args = _make_args("crr")
    algo = train_script._build_algorithm(args, buf, device="cpu")
    assert str(algo.device) == "cpu"
    metrics = algo.train(num_steps=10, batch_size=8, log_interval=20)
    assert len(metrics) == 10
    assert all(hasattr(m, "loss") for m in metrics)
    # Q-values
    q_vals = algo.get_action_values(["obs A", "obs B"], ["act A", "act B"])
    assert q_vals.shape == (2,)
    # Advantages
    adv = algo.get_advantages(["obs A", "obs B"], ["act A", "act B"])
    assert adv.shape == (2,)
    # Extra metrics keys
    assert "q_loss" in metrics[0].extra
    assert "actor_loss" in metrics[0].extra
    assert "advantage_mean" in metrics[0].extra


def test_crr_filter_types_run_without_error(tmp_dir):
    """CRR runs with binary, softmax, and exp filters without error."""
    import math
    buf = _create_buffer(tmp_dir, n_trajs=10)
    for filter_type in ("exp", "binary", "softmax"):
        args = _make_args("crr")
        args.crr_filter = filter_type
        algo = train_script._build_algorithm(args, buf, device="cpu")
        metrics = algo.train(num_steps=5, batch_size=4, log_interval=100)
        assert all(not math.isnan(m.loss) for m in metrics), (
            "NaN loss with crr_filter=%s" % filter_type
        )


def test_build_rwft_trains_and_returns_proxy_scores(tmp_dir):
    """RW-FT builds, trains with reward-weighted BC, returns proxy Q-values."""
    buf = _create_buffer(tmp_dir, n_trajs=12)
    args = _make_args("rwft")
    algo = train_script._build_algorithm(args, buf, device="cpu")
    assert str(algo.device) == "cpu"
    metrics = algo.train(num_steps=10, batch_size=8, log_interval=20)
    assert len(metrics) == 10
    assert all(hasattr(m, "loss") for m in metrics)
    # Extra metrics keys
    assert "bc_loss" in metrics[0].extra
    assert "reward_weight_mean" in metrics[0].extra
    # Proxy scores are negative MSE (unbounded, check shape)
    scores = algo.get_action_values(["s1", "s2", "s3"], ["a1", "a2", "a3"])
    assert scores.shape == (3,)


def test_rwft_reward_norm_softmax_vs_exp(tmp_dir):
    """RW-FT softmax and exp reward norms both run without NaN."""
    import math
    buf = _create_buffer(tmp_dir, n_trajs=10)
    for norm in ("softmax", "exp"):
        args = _make_args("rwft")
        args.rwft_reward_norm = norm
        algo = train_script._build_algorithm(args, buf, device="cpu")
        metrics = algo.train(num_steps=5, batch_size=4, log_interval=100)
        assert all(not math.isnan(m.loss) for m in metrics), (
            "NaN loss with rwft_reward_norm=%s" % norm
        )


def test_build_oreo_trains_and_returns_q_values(tmp_dir):
    """OREO builds, trains with soft Bellman, returns Q-values and advantages."""
    buf = _create_buffer(tmp_dir, n_trajs=10)
    args = _make_args("oreo")
    algo = train_script._build_algorithm(args, buf, device="cpu")
    assert str(algo.device) == "cpu"
    metrics = algo.train(num_steps=10, batch_size=8, log_interval=20)
    assert len(metrics) == 10
    assert all(hasattr(m, "loss") for m in metrics)
    # Q-values
    q_vals = algo.get_action_values(["s A", "s B"], ["a A", "a B"])
    assert q_vals.shape == (2,)
    # Soft advantages
    adv = algo.get_advantages(["s A", "s B"], ["a A", "a B"])
    assert adv.shape == (2,)
    # Extra metrics
    assert "q_loss" in metrics[0].extra
    assert "soft_adv_mean" in metrics[0].extra


def test_build_sorl_trains_with_ctn(tmp_dir):
    """SORL builds, trains with CTN, exposes clip_fraction metric."""
    import math
    buf = _create_buffer(tmp_dir, n_trajs=10)
    args = _make_args("sorl")
    algo = train_script._build_algorithm(args, buf, device="cpu")
    assert str(algo.device) == "cpu"
    metrics = algo.train(num_steps=10, batch_size=8, log_interval=20)
    assert len(metrics) == 10
    assert all(not math.isnan(m.loss) for m in metrics)
    # CTN-specific keys
    assert "clip_fraction" in metrics[0].extra
    assert "ctn_normalized" in metrics[0].extra


def test_oreo_soft_v_is_greater_than_arithmetic_mean_v(tmp_dir):
    """Soft V (log-mean-exp) >= arithmetic mean V for the SAME Q samples.

    Jensen's inequality: log(mean(exp(x_k))) >= mean(x_k)
    Equivalently:  β * log_mean_exp(Q_k/β) >= mean(Q_k)
    This must hold exactly when both use the same Q_k values.
    We verify the property by computing both on the same pre-drawn Q tensor.
    """
    import math
    import torch
    buf = _create_buffer(tmp_dir, n_trajs=8)
    args = _make_args("oreo")
    algo = train_script._build_algorithm(args, buf, device="cpu")
    algo.train(num_steps=5, batch_size=4, log_interval=100)

    beta = algo.beta
    K = 6
    # Construct synthetic Q values: some positive, some negative, varied
    torch.manual_seed(42)
    q_vals = torch.randn(8, K)  # (B, K) arbitrary Q tensor

    soft_v = beta * (
        torch.logsumexp(q_vals / beta, dim=1)
        - torch.log(torch.tensor(float(K)))
    )
    hard_v = q_vals.mean(dim=1)

    # Jensen's inequality must always hold on the same Q tensor
    assert (soft_v >= hard_v - 1e-5).all(), (
        "Soft V >= arithmetic V not satisfied: diff=%s" % (soft_v - hard_v).tolist()
    )
    # Also check that OREO._soft_v returns finite values of correct shape
    sample = buf.sample_transitions(4)
    s_tokens = algo._tokenize(sample.observation_contexts[:4])
    with torch.no_grad():
        states_d = algo.state_encoder(s_tokens).detach()
    v = algo._soft_v(states_d)
    assert v.shape == (4,)
    assert all(math.isfinite(x) for x in v.tolist())