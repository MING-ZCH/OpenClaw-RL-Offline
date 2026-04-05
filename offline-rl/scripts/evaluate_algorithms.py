#!/usr/bin/env python
"""
evaluate_algorithms.py: Multi-algorithm offline RL comparison benchmark.

Trains multiple offline RL algorithms on the same dataset for the same
number of steps and records per-algorithm performance metrics, then
prints a comparison table.

Supported algorithms: iql, cql, awac, grpo, td3bc, edac, dt, crr, rwft, oreo, sorl, arpo

Usage:
    python scripts/evaluate_algorithms.py --data data/trajs.jsonl --algos iql cql awac crr oreo
    python scripts/evaluate_algorithms.py --data data/trajs.jsonl --algos all --steps 300
    python scripts/evaluate_algorithms.py --data data/trajs.jsonl --algos iql td3bc oreo --output eval_results.csv

Output columns:
    Algorithm | Steps | Final Loss | Loss Trend | Q Mean | Q Std | Time(s)
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import os
import sys
import time
from typing import List, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offline_rl.data.trajectory_store import TrajectoryStore
from offline_rl.data.replay_buffer import ReplayBuffer
from offline_rl.algorithms.iql import IQL
from offline_rl.algorithms.cql import CQL
from offline_rl.algorithms.awac import AWAC
from offline_rl.algorithms.off_policy_grpo import OffPolicyGRPO
from offline_rl.algorithms.td3bc import TD3BC
from offline_rl.algorithms.edac import EDAC
from offline_rl.algorithms.decision_transformer import DecisionTransformer
from offline_rl.algorithms.crr import CRR
from offline_rl.algorithms.rw_finetuning import RWFineTuning
from offline_rl.algorithms.oreo import OREO
from offline_rl.algorithms.sorl import SORLOffPolicyGRPO
from offline_rl.algorithms.arpo import ARPO

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

ALL_ALGOS = ["iql", "cql", "awac", "grpo", "td3bc", "edac", "dt", "crr", "rwft", "oreo", "sorl", "arpo"]

_ALGO_REFS = {
    "iql": "Kostrikov et al. ICLR 2022",
    "cql": "Kumar et al. NeurIPS 2020",
    "awac": "Nair et al. 2021",
    "grpo": "OpenClaw off-policy GRPO",
    "td3bc": "Fujimoto & Gu NeurIPS 2021",
    "edac": "An et al. NeurIPS 2021",
    "dt": "Chen et al. NeurIPS 2021",
    "crr": "Wang et al. NeurIPS 2020",
    "rwft": "Mukherjee et al. NeurIPS 2025",
    "oreo": "Wang et al. arXiv 2412.16145",
    "sorl": "Li et al. arXiv 2511.20718",
    "arpo": "arXiv 2505.16282 (dvlab-research/ARPO)",
}


def _build_algo(name: str, buf: ReplayBuffer, args: argparse.Namespace, device: str):
    """Build one algorithm instance with shared hyper-params."""
    common = dict(
        replay_buffer=buf,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        device=device,
    )
    if name == "iql":
        return IQL(**common, tau=0.7, beta=3.0)
    if name == "cql":
        return CQL(**common, alpha=1.0)
    if name == "awac":
        return AWAC(**common, lam=1.0)
    if name == "grpo":
        return OffPolicyGRPO(**common, clip_ratio=0.2, kl_coeff=0.01, n_policy_updates=2)
    if name == "td3bc":
        return TD3BC(**common, alpha=2.5, target_noise=0.2, noise_clip=0.5, policy_freq=2)
    if name == "edac":
        return EDAC(**common, n_critics=5, eta=1.0, alpha_init=0.1, auto_alpha=True)
    if name == "dt":
        return DecisionTransformer(**common, context_len=10, nhead=2, num_transformer_layers=1)
    if name == "crr":
        return CRR(**common, beta=1.0, filter_type="exp", mc_samples=8)
    if name == "rwft":
        return RWFineTuning(**common, beta=1.0, reward_norm="softmax", reward_clip=10.0)
    if name == "oreo":
        return OREO(**common, beta=1.0, mc_samples=8, target_update_rate=0.005)
    if name == "sorl":
        return SORLOffPolicyGRPO(
            **common, clip_ratio=0.2, kl_coeff=0.01, n_policy_updates=2,
            clip_norm_threshold=0.2,
        )
    if name == "arpo":
        return ARPO(
            **common, clip_ratio_low=0.2, clip_ratio_high=0.3,
            n_policy_updates=2, arpo_buffer_size=8,
        )
    raise ValueError("Unknown algorithm: %s" % name)


def _compute_loss_trend(losses: List[float]) -> float:
    """Compute average loss slope over the last 25% of training steps.

    Returns:
        slope_per_step (negative = improving, positive = diverging)
    """
    n = len(losses)
    if n < 4:
        return 0.0
    tail = losses[3 * n // 4:]
    if len(tail) < 2:
        return 0.0
    x = list(range(len(tail)))
    mx = sum(x) / len(x)
    my = sum(tail) / len(tail)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, tail))
    den = sum((xi - mx) ** 2 for xi in x)
    return (num / den) if abs(den) > 1e-12 else 0.0


def _safe_q_stats(algo, sample_states: List[str], sample_actions: List[str]):
    """Compute Q-value mean/std, falling back to (nan, nan) on error."""
    try:
        q_vals = algo.get_action_values(sample_states, sample_actions)
        q_np = q_vals.detach().cpu().float()
        q_list = q_np.tolist()
        q_list = [v for v in q_list if math.isfinite(v)]
        if not q_list:
            return float("nan"), float("nan")
        mean = sum(q_list) / len(q_list)
        std = (sum((v - mean) ** 2 for v in q_list) / len(q_list)) ** 0.5
        return mean, std
    except Exception as exc:
        logger.debug("Q-stats error for %s: %s", type(algo).__name__, exc)
        return float("nan"), float("nan")


def _run_algo(
    name: str,
    buf: ReplayBuffer,
    args: argparse.Namespace,
    device: str,
    sample_states: List[str],
    sample_actions: List[str],
) -> dict:
    """Train one algorithm and collect evaluation metrics."""
    needs_traj = name == "dt"

    algo = _build_algo(name, buf, args, device)
    losses = []
    t0 = time.time()

    for step in range(args.steps):
        if needs_traj:
            raw = buf.sample_trajectories(args.batch_size)
            if not raw:
                continue
        else:
            raw = buf.sample_transitions(args.batch_size)

        try:
            m = algo.train_step(raw)
            if math.isfinite(m.loss):
                losses.append(m.loss)
        except Exception as exc:
            logger.warning("[%s] train_step error at step %d: %s", name, step, exc)
            break

    elapsed = time.time() - t0

    if not losses:
        return {
            "name": name,
            "ref": _ALGO_REFS.get(name, ""),
            "steps": 0,
            "final_loss": float("nan"),
            "loss_trend": float("nan"),
            "q_mean": float("nan"),
            "q_std": float("nan"),
            "time_s": elapsed,
            "error": "no metrics collected",
        }

    final_loss = sum(losses[-max(1, len(losses) // 10):]) / max(1, len(losses) // 10)
    trend = _compute_loss_trend(losses)
    q_mean, q_std = _safe_q_stats(algo, sample_states, sample_actions)

    return {
        "name": name,
        "ref": _ALGO_REFS.get(name, ""),
        "steps": len(losses),
        "final_loss": final_loss,
        "loss_trend": trend,
        "q_mean": q_mean,
        "q_std": q_std,
        "time_s": elapsed,
        "error": None,
    }


def _fmt(v, fmt="%.4f"):
    if isinstance(v, float) and not math.isfinite(v):
        return "n/a"
    try:
        return fmt % v
    except Exception:
        return str(v)


def _print_table(results: List[dict], verbose: bool = False) -> None:
    """Print a formatted comparison table to stdout."""
    header = (
        "%-16s %-8s %-12s %-12s %-10s %-10s %-8s"
        % ("Algorithm", "Steps", "Final Loss", "Trend/step", "Q Mean", "Q Std", "Time(s)")
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in results:
        trend_str = _fmt(r["loss_trend"], "%+.5f")
        label = r["name"].upper()
        if r.get("error"):
            row = "%-16s %-8s %-12s %-12s %-10s %-10s %-8s  [ERROR: %s]" % (
                label, "0", "n/a", "n/a", "n/a", "n/a", _fmt(r["time_s"], "%.2f"),
                r["error"],
            )
        else:
            row = "%-16s %-8d %-12s %-12s %-10s %-10s %-8s" % (
                label,
                r["steps"],
                _fmt(r["final_loss"]),
                trend_str,
                _fmt(r["q_mean"]),
                _fmt(r["q_std"]),
                _fmt(r["time_s"], "%.2f"),
            )
        print(row)
        if verbose and r.get("ref"):
            print("    Reference: %s" % r["ref"])
    print(sep)
    # Rank by final loss (lower is better, skip nan)
    ranked = [r for r in results if r.get("error") is None and math.isfinite(r["final_loss"])]
    if ranked:
        ranked_sorted = sorted(ranked, key=lambda r: r["final_loss"])
        print("\nBest final loss: %s (%.4f)" % (
            ranked_sorted[0]["name"].upper(), ranked_sorted[0]["final_loss"]
        ))
        improving = [r for r in ranked if r["loss_trend"] < 0]
        if improving:
            most_improving = sorted(improving, key=lambda r: r["loss_trend"])[0]
            print("Fastest improving: %s (trend %+.5f/step)" % (
                most_improving["name"].upper(), most_improving["loss_trend"]
            ))
    print(sep + "\n")


def _save_csv(results: List[dict], path: str) -> None:
    """Save results to CSV."""
    import csv
    keys = ["name", "ref", "steps", "final_loss", "loss_trend", "q_mean", "q_std", "time_s", "error"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in keys})
    print("Saved CSV to %s" % path)


def _save_markdown(results: List[dict], path: str) -> None:
    """Save results to markdown table."""
    lines = [
        "| Algorithm | Steps | Final Loss | Trend/step | Q Mean | Q Std | Time(s) | Reference |",
        "|-----------|-------|------------|------------|--------|-------|---------|-----------|",
    ]
    for r in results:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["name"].upper(),
            str(r["steps"]),
            _fmt(r["final_loss"]),
            _fmt(r["loss_trend"], "%+.5f"),
            _fmt(r["q_mean"]),
            _fmt(r["q_std"]),
            _fmt(r["time_s"], "%.2f"),
            r.get("ref", ""),
        ))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("Saved Markdown table to %s" % path)


def main():
    parser = argparse.ArgumentParser(description="Offline RL algorithm comparison benchmark")
    parser.add_argument("--data", required=True, help="Path to trajectory JSONL dataset")
    parser.add_argument(
        "--algos", nargs="+", default=["iql", "cql", "awac", "td3bc", "crr", "oreo"],
        help="Algorithms to compare. Use 'all' to run all 11 algorithms.",
    )
    parser.add_argument("--steps", type=int, default=200, help="Training steps per algorithm")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per step")
    parser.add_argument("--state-dim", type=int, default=64, help="State embedding dim")
    parser.add_argument("--action-dim", type=int, default=64, help="Action embedding dim")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer dim")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (shared)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default=None, help="Save results to CSV (.csv) or Markdown (.md)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print references in table")
    args = parser.parse_args()

    # Resolve algos
    if args.algos == ["all"] or args.algos[0] == "all":
        algos = ALL_ALGOS
    else:
        unknown = [a for a in args.algos if a not in ALL_ALGOS]
        if unknown:
            parser.error("Unknown algorithms: %s. Valid: %s" % (unknown, ALL_ALGOS))
        algos = args.algos

    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Resolve data path
    data_path = args.data
    if not os.path.isabs(data_path):
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_path
        )

    print("Loading dataset from %s ..." % data_path)
    store = TrajectoryStore(data_path)
    stats = store.stats()
    print("Dataset: %d trajectories, %.1f%% success" % (
        stats["total_trajectories"], stats["success_rate"] * 100
    ))

    buf = ReplayBuffer(store=store, max_trajectories=10000)
    loaded = buf.load_from_store()

    if loaded == 0:
        print("ERROR: Replay buffer is empty. Check --data path and file contents.")
        sys.exit(1)

    # Collect sample state-action pairs for Q-value reporting
    sample = buf.sample_transitions(min(8, len(buf)))
    sample_states = list(sample.observation_contexts[:8])
    sample_actions = list(sample.actions[:8])

    print("\nAlgorithms to evaluate: %s" % ", ".join(algos))
    print("Steps per algo: %d | Batch size: %d | Device: %s\n" % (
        args.steps, args.batch_size, device
    ))

    results = []
    for i, name in enumerate(algos):
        print("[%d/%d] Training %-16s ..." % (i + 1, len(algos), name.upper()), end=" ", flush=True)
        result = _run_algo(name, buf, args, device, sample_states, sample_actions)
        results.append(result)
        status = ("DONE  final_loss=%.4f  time=%.1fs" % (result["final_loss"], result["time_s"])
                  if not result.get("error")
                  else "ERROR: %s" % result["error"])
        print(status)

    _print_table(results, verbose=args.verbose)

    if args.output:
        if args.output.endswith(".md"):
            _save_markdown(results, args.output)
        else:
            _save_csv(results, args.output)


if __name__ == "__main__":
    main()
