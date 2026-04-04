#!/usr/bin/env python
"""
Train offline RL algorithms (IQL/CQL/AWAC/GRPO) on pre-collected trajectories.

Usage:
    python scripts/train_offline.py --data data/trajectories.jsonl --algo iql --steps 500
    python scripts/train_offline.py --data data/trajectories.jsonl --algo cql --alpha 2.0
    python scripts/train_offline.py --data data/trajectories.jsonl --algo awac --lam 0.5
    python scripts/train_offline.py --data data/trajectories.jsonl --algo grpo --n-policy-updates 2
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offline_rl.data.trajectory_store import TrajectoryStore
from offline_rl.data.replay_buffer import ReplayBuffer
from offline_rl.algorithms.iql import IQL
from offline_rl.algorithms.cql import CQL
from offline_rl.algorithms.awac import AWAC
from offline_rl.algorithms.off_policy_grpo import OffPolicyGRPO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


ALGO_MAP = {"iql": IQL, "cql": CQL, "awac": AWAC, "grpo": OffPolicyGRPO}


def _save_checkpoint(algo, output_path: str) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    buffer = io.BytesIO()
    algo.save(buffer)
    with open(output_path, "wb") as f:
        f.write(buffer.getvalue())


def main():
    parser = argparse.ArgumentParser(description="Offline RL training")
    parser.add_argument("--data", type=str, required=True, help="Path to trajectory JSONL")
    parser.add_argument("--algo", type=str, choices=["iql", "cql", "awac", "grpo"], default="iql")
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
    parser.add_argument("--output", type=str, default=None, help="Save checkpoint path")
    args = parser.parse_args()

    # Resolve data path
    data_path = args.data
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_path)

    logger.info("Loading trajectories from %s", data_path)
    store = TrajectoryStore(data_path)
    stats = store.stats()
    logger.info("Dataset: %d trajectories, %.1f%% success, avg %.1f steps",
                stats["total_trajectories"], stats["success_rate"] * 100, stats["avg_steps"])

    buf = ReplayBuffer(store=store, max_trajectories=10000)
    loaded = buf.load_from_store()
    logger.info("Loaded %d trajectories into replay buffer", loaded)

    if args.algo == "iql":
        algo = IQL(
            replay_buffer=buf,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            gamma=args.gamma,
            device="cpu",
            tau=args.tau,
            beta=args.beta,
        )
    elif args.algo == "cql":
        algo = CQL(
            replay_buffer=buf,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            gamma=args.gamma,
            device="cpu",
            alpha=args.alpha,
        )
    elif args.algo == "awac":
        algo = AWAC(
            replay_buffer=buf,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            gamma=args.gamma,
            device="cpu",
            lam=args.lam,
        )
    elif args.algo == "grpo":
        algo = OffPolicyGRPO(
            replay_buffer=buf,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            gamma=args.gamma,
            device="cpu",
            clip_ratio=args.clip_ratio,
            kl_coeff=args.kl_coeff,
            n_policy_updates=args.n_policy_updates,
        )
    else:
        raise ValueError(f"Unknown algo: {args.algo}")

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
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), output_path)
        _save_checkpoint(algo, output_path)
        logger.info("Saved checkpoint to %s", output_path)


if __name__ == "__main__":
    main()
