"""
Pre-compute advantage weights from offline RL critic for LLM fine-tuning.

This script trains one of 15 offline RL critics on trajectory data, then
exports per-(trajectory, step) advantage/value weights as a JSON file for use
by the custom ``offline_loss.py`` function during slime training.

All 15 algorithms supported:
    iql     cql     awac    grpo    td3bc   edac
    dt      crr     rwft    oreo    sorl    arpo
    retrospex  webrl  glider

Usage:
    python compute_weights.py --algo iql --data trajectories.jsonl --output weights.json
    python compute_weights.py --algo cql --data trajectories.jsonl --output weights.json --alpha 2.0
    python compute_weights.py --algo td3bc --data trajectories.jsonl --output weights.json
    python compute_weights.py --algo retrospex --data trajectories.jsonl --output weights.json --retrospex-tau 0.7
    python compute_weights.py --algo webrl --data trajectories.jsonl --output weights.json --webrl-alpha-orm 0.5
    python compute_weights.py --algo glider --data trajectories.jsonl --output weights.json --glider-plan-dim 64

Output JSON maps "{trajectory_id}:{step_idx}" → float advantage/value.

Advantage dispatch:
  - Algorithms with ``get_advantages()`` (IQL, AWAC, CRR, EDAC, OREO):
        A(s,a) = Q(s,a) - V(s)   (true IQL advantage)
  - All other algorithms use ``get_action_values()`` as Q/value proxy:
        GRPO, SORL, ARPO, DT, RW-FT  → log-prob or policy score proxy
        CQL, TD3+BC, Retrospex, GLIDER → Q-value
        WebRL                          → ORM probability P(success) ∈ [0, 1]
"""

import argparse
import json
import logging
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OFFLINE_RL_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "offline-rl")
if _OFFLINE_RL_DIR not in sys.path:
    sys.path.insert(0, _OFFLINE_RL_DIR)

from offline_rl.data.trajectory_store import TrajectoryStore
from offline_rl.data.replay_buffer import ReplayBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Algorithms that have a true get_advantages() method (Q - V)
_HAS_ADVANTAGES = {"iql", "awac", "crr", "edac", "oreo"}


def _build_algo(args, buffer, device):
    """
    Construct the correct algorithm instance from parsed args.

    Keeps the same shared kwargs for all algorithms (state_dim, action_dim,
    hidden_dim, lr, gamma, device) and dispatches algo-specific kwargs.
    """
    common = dict(
        replay_buffer=buffer,
        state_dim=256,
        action_dim=256,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        device=device,
    )

    if args.algo == "iql":
        from offline_rl.algorithms.iql import IQL
        return IQL(**common, tau=0.7, beta=args.beta)

    if args.algo == "cql":
        from offline_rl.algorithms.cql import CQL
        return CQL(**common, alpha=args.alpha)

    if args.algo == "awac":
        from offline_rl.algorithms.awac import AWAC
        return AWAC(**common, lam=args.lam)

    if args.algo == "grpo":
        from offline_rl.algorithms.off_policy_grpo import OffPolicyGRPO
        return OffPolicyGRPO(**common, clip_ratio=0.2, kl_coeff=0.01, n_policy_updates=2)

    if args.algo == "td3bc":
        from offline_rl.algorithms.td3bc import TD3BC
        return TD3BC(**common, alpha=args.alpha)

    if args.algo == "edac":
        from offline_rl.algorithms.edac import EDAC
        return EDAC(**common, n_critics=3, eta=1.0, alpha_init=0.1, auto_alpha=True)

    if args.algo == "dt":
        from offline_rl.algorithms.decision_transformer import DecisionTransformer
        return DecisionTransformer(**common, context_len=10, nhead=2, num_transformer_layers=1)

    if args.algo == "crr":
        from offline_rl.algorithms.crr import CRR
        return CRR(**common, beta=args.beta, filter_type="exp", mc_samples=8)

    if args.algo == "rwft":
        from offline_rl.algorithms.rw_finetuning import RWFineTuning
        return RWFineTuning(**common, beta=args.beta, reward_norm="softmax", reward_clip=10.0)

    if args.algo == "oreo":
        from offline_rl.algorithms.oreo import OREO
        return OREO(**common, beta=args.beta, mc_samples=8)

    if args.algo == "sorl":
        from offline_rl.algorithms.sorl import SORLOffPolicyGRPO
        return SORLOffPolicyGRPO(
            **common, clip_ratio=0.2, kl_coeff=0.01, n_policy_updates=2,
            clip_norm_threshold=0.2,
        )

    if args.algo == "arpo":
        from offline_rl.algorithms.arpo import ARPO
        return ARPO(
            **common, clip_ratio_low=0.2, clip_ratio_high=0.3,
            n_policy_updates=2, arpo_buffer_size=8,
        )

    if args.algo == "retrospex":
        from offline_rl.algorithms.retrospex import Retrospex
        return Retrospex(
            **common,
            tau=args.retrospex_tau,
            lambda_scale=args.retrospex_lambda_scale,
        )

    if args.algo == "webrl":
        from offline_rl.algorithms.webrl import WebRL
        return WebRL(
            **common,
            clip_ratio=0.2, kl_coeff=0.01, n_policy_updates=2,
            alpha_orm=args.webrl_alpha_orm,
        )

    if args.algo == "glider":
        from offline_rl.algorithms.glider import GLIDER
        return GLIDER(
            **common,
            plan_dim=args.glider_plan_dim,
            beta=args.glider_beta,
            tau=args.glider_tau,
        )

    raise ValueError("Unknown algorithm: %s" % args.algo)


def _get_advantage(algo, algo_name: str, state: str, action: str) -> float:
    """
    Extract a scalar advantage / value estimate for one (state, action) pair.

    Dispatch:
      - ``get_advantages()`` if the algorithm provides true IQL-style Q-V.
      - ``get_action_values()`` as a Q/ value proxy otherwise.
    """
    if algo_name in _HAS_ADVANTAGES:
        return algo.get_advantages([state], [action]).item()
    return algo.get_action_values([state], [action]).item()


def main():
    parser = argparse.ArgumentParser(description="Compute offline RL advantage weights for slime fine-tuning")
    parser.add_argument("--data", required=True, help="Path to trajectories.jsonl")
    parser.add_argument("--output", required=True, help="Output weights.json path")
    parser.add_argument(
        "--algo",
        choices=[
            "iql", "cql", "awac", "grpo", "td3bc", "edac",
            "dt", "crr", "rwft", "oreo", "sorl", "arpo",
            "retrospex", "webrl", "glider",
        ],
        default="iql",
        help="Offline RL algorithm to train the critic (default: iql)",
    )
    parser.add_argument("--train-steps", type=int, default=500, help="Training steps for critic")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--device", default="cuda", help="torch device (default: cuda)")
    # Shared hyperparams
    parser.add_argument("--beta", type=float, default=3.0,
                        help="Temperature for IQL/CRR/OREO/RW-FT advantage weighting")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="CQL conservative coeff OR TD3+BC Q-weight")
    parser.add_argument("--lam", type=float, default=1.0,
                        help="AWAC advantage temperature")
    # Retrospex-specific
    parser.add_argument("--retrospex-tau", type=float, default=0.7,
                        help="Retrospex IQL expectile parameter (default 0.7)")
    parser.add_argument("--retrospex-lambda-scale", type=float, default=1.0,
                        help="Retrospex Q-critic weight in combined rescoring (default 1.0)")
    # WebRL-specific
    parser.add_argument("--webrl-alpha-orm", type=float, default=0.5,
                        help="WebRL ORM reward mix weight (default 0.5)")
    # GLIDER-specific
    parser.add_argument("--glider-plan-dim", type=int, default=None,
                        help="GLIDER latent plan embedding dimension (defaults to hidden_dim//4)")
    parser.add_argument("--glider-beta", type=float, default=1.0,
                        help="GLIDER AWR advantage temperature")
    parser.add_argument("--glider-tau", type=float, default=0.7,
                        help="GLIDER IQL expectile parameter")
    args = parser.parse_args()

    # Load data
    store = TrajectoryStore(args.data)
    count = store.count()
    if count == 0:
        logger.error("Trajectory store is empty: %s", args.data)
        return

    logger.info("Loaded %d trajectories from %s", count, args.data)

    buffer = ReplayBuffer(store=store, max_trajectories=5000)
    loaded = buffer.load_from_store()
    logger.info("Loaded %d trajectories → %d transitions", loaded, len(buffer))

    # Resolve device
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        args.device = "cpu"

    # Build algorithm
    logger.info("Building %s critic on device=%s", args.algo.upper(), args.device)
    algo = _build_algo(args, buffer, args.device)

    # Train critic
    logger.info("Training %s for %d steps...", args.algo.upper(), args.train_steps)
    metrics = algo.train(num_steps=args.train_steps, batch_size=args.batch_size, log_interval=100)
    final_loss = metrics[-1].loss if metrics else float("nan")
    logger.info("Training complete. Final loss: %.4f", final_loss)

    # Extract advantages for all transitions
    logger.info("Computing advantage weights for all %d trajectories...", count)
    weights = {}

    for traj in store.iter_trajectories():
        for i, step in enumerate(traj.steps):
            # Build running-context state string (same as slime prompt format)
            state = "Task: {}\n".format(traj.instruction)
            for prev in traj.steps[:i]:
                state += "Step {}: {}\n".format(prev.step_idx, prev.action)
            action = step.action

            key = "{}:{}".format(traj.trajectory_id, step.step_idx)
            weights[key] = _get_advantage(algo, args.algo, state, action)

    # Save
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(weights, f, indent=2)

    logger.info("Saved %d advantage weights to %s", len(weights), args.output)

    # Summary
    values = list(weights.values())
    mean_val = sum(values) / len(values)
    logger.info("  Mean advantage: %.4f", mean_val)
if __name__ == "__main__":
    main()
