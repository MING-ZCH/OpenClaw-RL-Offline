#!/usr/bin/env python
"""
Train offline RL algorithms (IQL/CQL/AWAC/GRPO/TD3BC) on pre-collected trajectories.

Usage:
    python scripts/train_offline.py --data data/trajectories.jsonl --algo iql --steps 500
    python scripts/train_offline.py --data data/trajectories.jsonl --algo cql --alpha 2.0
    python scripts/train_offline.py --data data/trajectories.jsonl --algo awac --lam 0.5
    python scripts/train_offline.py --data data/trajectories.jsonl --algo grpo --n-policy-updates 2
    python scripts/train_offline.py --data data/trajectories.jsonl --algo td3bc --td3bc-alpha 2.5
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offline_rl.data.trajectory_store import TrajectoryStore
from offline_rl.data.replay_buffer import ReplayBuffer
from offline_rl.algorithms.iql import IQL
from offline_rl.algorithms.cql import CQL
from offline_rl.algorithms.awac import AWAC
from offline_rl.algorithms.off_policy_grpo import OffPolicyGRPO
from offline_rl.algorithms.td3bc import TD3BC

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


ALGO_MAP = {"iql": IQL, "cql": CQL, "awac": AWAC, "grpo": OffPolicyGRPO, "td3bc": TD3BC}


def _save_checkpoint(algo, output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    buffer = io.BytesIO()
    algo.save(buffer)
    with open(output_path, "wb") as f:
        f.write(buffer.getvalue())


def _resolve_repo_path(path_str: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path_str)


def _resolve_device(requested_device: str) -> str:
    normalized = (requested_device or "auto").strip().lower()

    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if normalized == "cpu":
        return "cpu"

    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
        return "cuda"

    if normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError("A specific CUDA device was requested but CUDA is unavailable")
        index_text = normalized.split(":", 1)[1]
        if not index_text.isdigit():
            raise ValueError("Expected device to be one of: auto, cpu, cuda, cuda:N")
        device_index = int(index_text)
        if device_index >= torch.cuda.device_count():
            raise ValueError(
                "Requested CUDA device index %d, but only %d device(s) are visible"
                % (device_index, torch.cuda.device_count())
            )
        return normalized

    raise ValueError("Expected device to be one of: auto, cpu, cuda, cuda:N")


def _build_algorithm(args, replay_buffer: ReplayBuffer, device: str):
    if args.algo == "iql":
        return IQL(
            replay_buffer=replay_buffer,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            gamma=args.gamma,
            device=device,
            tau=args.tau,
            beta=args.beta,
        )
    if args.algo == "cql":
        return CQL(
            replay_buffer=replay_buffer,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            gamma=args.gamma,
            device=device,
            alpha=args.alpha,
        )
    if args.algo == "awac":
        return AWAC(
            replay_buffer=replay_buffer,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            gamma=args.gamma,
            device=device,
            lam=args.lam,
        )
    if args.algo == "grpo":
        return OffPolicyGRPO(
            replay_buffer=replay_buffer,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            gamma=args.gamma,
            device=device,
            clip_ratio=args.clip_ratio,
            kl_coeff=args.kl_coeff,
            n_policy_updates=args.n_policy_updates,
        )
    if args.algo == "td3bc":
        return TD3BC(
            replay_buffer=replay_buffer,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            gamma=args.gamma,
            device=device,
            alpha=args.td3bc_alpha,
            target_noise=args.td3bc_target_noise,
            noise_clip=args.td3bc_noise_clip,
            policy_freq=args.td3bc_policy_freq,
        )
    raise ValueError("Unknown algo: %s" % args.algo)


def main():
    parser = argparse.ArgumentParser(description="Offline RL training")
    parser.add_argument("--data", type=str, required=True, help="Path to trajectory JSONL")
    parser.add_argument("--algo", type=str, choices=["iql", "cql", "awac", "grpo", "td3bc"], default="iql")
    parser.add_argument("--device", type=str, default="cuda", help="Training device: cuda | cuda:N | auto | cpu")
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--state-dim", type=int, default=128, help="State embedding dim")
    parser.add_argument("--action-dim", type=int, default=128, help="Action embedding dim")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden layer dim")
    parser.add_argument("--log-interval", type=int, default=50, help="Log every N steps")
    # IQL specific
    parser.add_argument("--tau", type=float, default=0.7, help="IQL expectile param")
    parser.add_argument("--beta", type=float, default=3.0, help="IQL policy extraction temp")
    # CQL specific
    parser.add_argument("--alpha", type=float, default=1.0, help="CQL regularization coeff")
    # AWAC specific
    parser.add_argument("--lam", type=float, default=1.0, help="AWAC advantage temp")
    # GRPO specific
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="GRPO PPO-style clip ratio")
    parser.add_argument("--kl-coeff", type=float, default=0.01, help="GRPO KL penalty coefficient")
    parser.add_argument("--n-policy-updates", type=int, default=4, help="GRPO updates per sampled batch")
    # TD3+BC specific
    parser.add_argument("--td3bc-alpha", type=float, default=2.5, help="TD3+BC relative Q-weight (Fujimoto 2021 default: 2.5)")
    parser.add_argument("--td3bc-target-noise", type=float, default=0.2, help="TD3+BC target policy smoothing noise std")
    parser.add_argument("--td3bc-noise-clip", type=float, default=0.5, help="TD3+BC target noise clipping range")
    parser.add_argument("--td3bc-policy-freq", type=int, default=2, help="TD3+BC actor update interval (delayed policy update)")
    parser.add_argument("--output", type=str, default=None, help="Save checkpoint path")
    args = parser.parse_args()

    try:
        resolved_device = _resolve_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))

    # Resolve data path
    data_path = _resolve_repo_path(args.data)

    logger.info("Loading trajectories from %s", data_path)
    store = TrajectoryStore(data_path)
    stats = store.stats()
    logger.info("Dataset: %d trajectories, %.1f%% success, avg %.1f steps",
                stats["total_trajectories"], stats["success_rate"] * 100, stats["avg_steps"])

    buf = ReplayBuffer(store=store, max_trajectories=10000)
    loaded = buf.load_from_store()
    logger.info("Loaded %d trajectories into replay buffer", loaded)

    logger.info("Resolved device: requested=%s actual=%s", args.device, resolved_device)
    algo = _build_algorithm(args, buf, resolved_device)

    logger.info("Training %s for %d steps (batch_size=%d)", args.algo.upper(), args.steps, args.batch_size)
    t0 = time.time()
    metrics = algo.train(num_steps=args.steps, batch_size=args.batch_size, log_interval=args.log_interval)
    elapsed = time.time() - t0

    final_loss = sum(m.loss for m in metrics[-args.log_interval:]) / min(args.log_interval, len(metrics))
    logger.info("Training complete in %.1fs. Final avg loss: %.4f", elapsed, final_loss)

    # Evaluate on a few sample states
    sample_batch = buf.sample_transitions(min(5, len(buf)))
    q_values = algo.get_action_values(sample_batch.observation_contexts, sample_batch.actions)
    logger.info("Sample Q-values: %s", q_values.tolist())

    if args.output:
        output_path = _resolve_repo_path(args.output)
        _save_checkpoint(algo, output_path)
        logger.info("Saved checkpoint to %s", output_path)


if __name__ == "__main__":
    main()
