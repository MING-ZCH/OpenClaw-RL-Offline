#!/usr/bin/env python
"""
Collect offline trajectories using the mock environment.

This script demonstrates trajectory collection and storage, which can later
be used for offline RL training with IQL/CQL/AWAC.

Usage:
    python scripts/collect_offline_data.py --output data/trajectories.jsonl --n 200
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offline_rl.data.trajectory_store import TrajectoryStore
from offline_rl.envs.mock_env_server import MockEnvPoolServer, generate_mock_trajectories

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Collect offline trajectories")
    parser.add_argument("--output", type=str, default="data/trajectories.jsonl", help="Output JSONL path")
    parser.add_argument("--n", type=int, default=200, help="Number of trajectories to collect")
    parser.add_argument("--success-rate", type=float, default=0.3, help="Mock agent success rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    logger.info("Collecting %d trajectories with success_rate=%.2f", args.n, args.success_rate)

    server = MockEnvPoolServer()
    trajectories = generate_mock_trajectories(
        server, n_trajectories=args.n, agent_success_rate=args.success_rate
    )

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    store = TrajectoryStore(output_path)
    store.append_batch(trajectories)

    stats = store.stats()
    logger.info("Collection complete!")
    logger.info("  Total trajectories: %d", stats["total_trajectories"])
    logger.info("  Success rate: %.2f", stats["success_rate"])
    logger.info("  Domains: %s", stats["domains"])
    logger.info("  Avg steps: %.1f", stats["avg_steps"])
    logger.info("  Saved to: %s", output_path)


if __name__ == "__main__":
    main()
