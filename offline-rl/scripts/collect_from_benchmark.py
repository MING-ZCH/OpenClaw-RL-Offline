#!/usr/bin/env python
"""
Collect trajectories from any supported benchmark environment.

Supports: mock, osworld, androidworld, webarena, alfworld

Usage:
    python scripts/collect_from_benchmark.py --env osworld --n 50 --output data/os_trajs.jsonl
    python scripts/collect_from_benchmark.py --env androidworld --n 50 --output data/aw_trajs.jsonl
    python scripts/collect_from_benchmark.py --env webarena --n 50 --output data/wa_trajs.jsonl
    python scripts/collect_from_benchmark.py --env alfworld --n 50 --output data/alf_trajs.jsonl
    python scripts/collect_from_benchmark.py --env mock --n 100 --output data/mock_trajs.jsonl
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import uuid
from typing import Any, cast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offline_rl.data.trajectory_store import Step, Trajectory, TrajectoryStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_ENVS = ["mock", "osworld", "androidworld", "webarena", "alfworld"]


def _get_adapter(env_name: str) -> Any:
    if env_name == "osworld":
        from offline_rl.envs.osworld_adapter import MockOSWorldAdapter
        return MockOSWorldAdapter()
    elif env_name == "androidworld":
        from offline_rl.envs.androidworld_adapter import MockAndroidWorldAdapter
        return MockAndroidWorldAdapter()
    elif env_name == "webarena":
        from offline_rl.envs.webarena_adapter import MockWebArenaAdapter
        return MockWebArenaAdapter()
    elif env_name == "alfworld":
        from offline_rl.envs.alfworld_adapter import MockAlfWorldAdapter
        return MockAlfWorldAdapter()
    elif env_name == "mock":
        from offline_rl.envs.mock_env_server import MockEnvPoolServer
        return MockEnvPoolServer()
    else:
        raise ValueError("Unknown env: %s. Options: %s" % (env_name, ", ".join(SUPPORTED_ENVS)))


def _sample_random_action(adapter: Any) -> str:
    benchmark_name = getattr(adapter, "BENCHMARK_NAME", "mock")
    if benchmark_name == "osworld":
        return random.choice([
            "click(540,50)",
            "type('openclaw rl')",
            "hotkey('ctrl','l')",
            "scroll(500,500,'down',3)",
            "wait(1.0)",
        ])
    if benchmark_name == "androidworld":
        return random.choice([
            "open_app('Settings')",
            "click(500, 300)",
            "type('hello')",
            "navigate_back",
            "wait",
        ])
    if benchmark_name == "webarena":
        return random.choice([
            "goto('shopping')",
            "click(200, 300)",
            "type('search query')",
            "scroll(down)",
            "stop",
        ])
    if benchmark_name == "alfworld":
        return random.choice([
            "look",
            "inventory",
            "open fridge",
            "take apple",
            "put apple on table",
            "done",
        ])
    return random.choice([
        "click(100, 200)",
        "type('hello')",
        "scroll(3)",
        "wait",
    ])


def _collect_episode_adapter(adapter: Any, episode_id: str, agent_success_rate: float = 0.3) -> Trajectory:
    """Run one episode with a mock agent using the adapter API."""
    lease = adapter.allocate(episode_id)
    lease_id = lease["lease_id"]

    reset_result = adapter.reset(lease_id)
    task_info = reset_result.get("task", {})
    instruction = task_info.get("instruction", "Unknown task")
    task_id = task_info.get("task_id", "unknown")

    # Get task config for target actions
    configs = adapter.get_task_configs()
    matching = [c for c in configs if c.task_id == task_id]
    target_actions = matching[0].target_actions if matching else []
    max_steps = matching[0].max_steps if matching else 10
    domain = matching[0].domain if matching else adapter.BENCHMARK_NAME

    steps = []
    for step_idx in range(max_steps):
        # Simple mock agent strategy
        if random.random() < agent_success_rate and step_idx < len(target_actions):
            action = target_actions[step_idx]
        else:
            action = _sample_random_action(adapter)

        result = adapter.step(lease_id, action)
        steps.append(Step(
            step_idx=step_idx,
            action=action,
            response=f"I will {action}",
            reward=0.0,
            done=result.get("done", False),
        ))
        if result.get("done", False):
            break

    eval_result = adapter.evaluate(lease_id)
    score = eval_result.get("score", 0.0)
    adapter.close(lease_id)

    return Trajectory(
        trajectory_id=str(uuid.uuid4()),
        domain=domain,
        example_id=task_id,
        instruction=instruction,
        steps=steps,
        outcome_reward=1.0 if score > 0.5 else -1.0,
        eval_score=score,
        num_steps=len(steps),
        status="completed" if score > 0.5 else "failed",
        source=adapter.BENCHMARK_NAME if hasattr(adapter, 'BENCHMARK_NAME') else "mock",
    )


def _collect_episode_mock_server(server: Any, episode_id: str, agent_success_rate: float = 0.3) -> Trajectory:
    """Run one episode with MockEnvPoolServer (different API from adapters)."""
    from offline_rl.envs.mock_env_server import generate_mock_trajectories
    trajs = generate_mock_trajectories(server, n_trajectories=1, agent_success_rate=agent_success_rate)
    return cast(Trajectory, trajs[0])


def main():
    parser = argparse.ArgumentParser(description="Collect trajectories from benchmark")
    parser.add_argument("--env", choices=SUPPORTED_ENVS, default="osworld")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--n", "--num-episodes", dest="n", type=int, default=50, help="Number of trajectories")
    parser.add_argument("--success-rate", type=float, default=0.3, help="Mock agent success rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    logger.info("Collecting %d trajectories from %s", args.n, args.env)

    adapter = _get_adapter(args.env)

    benchmark_name = getattr(adapter, "BENCHMARK_NAME", args.env)
    task_count = len(adapter.get_task_configs()) if hasattr(adapter, "get_task_configs") else 0
    if task_count:
        logger.info("Benchmark: %s (%d tasks available)", benchmark_name, task_count)

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    store = TrajectoryStore(output_path)

    for i in range(args.n):
        if args.env == "mock":
            traj = _collect_episode_mock_server(adapter, f"ep-{i}", args.success_rate)
        else:
            traj = _collect_episode_adapter(adapter, f"ep-{i}", args.success_rate)
        store.append(traj)

        if (i + 1) % 10 == 0:
            logger.info("Collected %d/%d trajectories", i + 1, args.n)

    stats = store.stats()
    logger.info("Collection complete!")
    logger.info("  Benchmark: %s", args.env)
    logger.info("  Total: %d trajectories", stats["total_trajectories"])
    logger.info("  Success rate: %.1f%%", stats["success_rate"] * 100)
    logger.info("  Domains: %s", stats["domains"])
    logger.info("  Avg steps: %.1f", stats["avg_steps"])
    logger.info("  Saved to: %s", output_path)


if __name__ == "__main__":
    main()
