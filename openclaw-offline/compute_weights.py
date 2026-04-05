"""
Pre-compute advantage weights from offline RL critic for LLM fine-tuning.

This script trains an IQL/CQL/TD3+BC critic on trajectory data, then exports
advantage weights as a JSON file for use with the custom loss function.

Usage:
    python compute_weights.py --algo iql --data trajectories.jsonl --output weights.json
    python compute_weights.py --algo cql --data trajectories.jsonl --output weights.json --alpha 2.0
    python compute_weights.py --algo td3bc --data trajectories.jsonl --output weights.json

The output weights.json maps "{trajectory_id}:{step_idx}" → advantage_value.
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


def main():
    parser = argparse.ArgumentParser(description="Compute offline RL advantage weights")
    parser.add_argument("--data", required=True, help="Path to trajectories.jsonl")
    parser.add_argument("--output", required=True, help="Output weights.json path")
    parser.add_argument("--algo", choices=["iql", "cql", "td3bc"], default="iql", help="Algorithm for critic training")
    parser.add_argument("--train-steps", type=int, default=500, help="Training steps for critic")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--beta", type=float, default=3.0, help="Temperature for IQL advantage weighting")
    parser.add_argument("--alpha", type=float, default=1.0, help="CQL conservative coefficient / TD3+BC Q-weight")
    parser.add_argument("--device", default="cuda", help="torch device (default: cuda)")
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

    # Train critic
    if args.algo == "iql":
        from offline_rl.algorithms.iql import IQL
        algo = IQL(
            replay_buffer=buffer,
            state_dim=256, action_dim=256, hidden_dim=256,
            lr=3e-4, tau=0.7, beta=args.beta,
            device=args.device,
        )
    elif args.algo == "cql":
        from offline_rl.algorithms.cql import CQL
        algo = CQL(
            replay_buffer=buffer,
            state_dim=256, action_dim=256, hidden_dim=256,
            lr=3e-4, alpha=args.alpha,
            device=args.device,
        )
    else:  # td3bc
        from offline_rl.algorithms.td3bc import TD3BC
        algo = TD3BC(
            replay_buffer=buffer,
            state_dim=256, action_dim=256, hidden_dim=256,
            lr=3e-4, alpha=args.alpha,
            device=args.device,
        )

    logger.info("Training %s for %d steps...", args.algo.upper(), args.train_steps)
    metrics = algo.train(num_steps=args.train_steps, batch_size=args.batch_size, log_interval=100)
    final_loss = metrics[-1].loss
    logger.info("Training complete. Final loss: %.4f", final_loss)

    # Extract advantages for all transitions
    logger.info("Computing advantage weights for all transitions...")
    weights = {}

    for traj in store.iter_trajectories():
        for i, step in enumerate(traj.steps):
            state = f"Task: {traj.instruction}\n"
            for prev in traj.steps[:i]:
                state += f"Step {prev.step_idx}: {prev.action}\n"
            action = step.action

            if args.algo == "iql":
                advantage = algo.get_advantages([state], [action]).item()
            else:
                # CQL and TD3+BC use Q-values directly as advantage proxy
                q_value = algo.get_action_values([state], [action]).item()
                advantage = q_value

            key = f"{traj.trajectory_id}:{step.step_idx}"
            weights[key] = advantage

    # Save
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(weights, f, indent=2)

    logger.info("Saved %d advantage weights to %s", len(weights), args.output)

    # Summary
    values = list(weights.values())
    logger.info("  Mean advantage: %.4f", sum(values) / len(values))
    logger.info("  Min: %.4f, Max: %.4f", min(values), max(values))


if __name__ == "__main__":
    main()
