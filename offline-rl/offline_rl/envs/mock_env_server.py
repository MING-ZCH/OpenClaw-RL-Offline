"""
Mock Environment Pool Server for CPU testing.

Simulates gui-rl's env_pool_server HTTP API without requiring actual VMs.
Useful for testing trajectory collection, replay buffer, and algorithm code.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import random
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Minimal 1x1 white PNG for mock screenshots
_MOCK_SCREENSHOT = base64.b64encode(
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f"
    b"\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
).decode("ascii")


class MockTask:
    """Simulated task with simple state machine."""

    def __init__(self, task_config: dict):
        self.instruction = task_config.get("instruction", "Test task")
        self.max_steps = task_config.get("max_steps", 10)
        self.target_actions = task_config.get("target_actions", ["click(100, 200)", "type('hello')"])
        self.current_step = 0
        self.actions_taken: list[str] = []
        self.done = False

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        """Execute action and return (obs, reward, done, info)."""
        self.current_step += 1
        self.actions_taken.append(action)
        self.done = self.current_step >= self.max_steps or action.upper() in ("FAIL", "DONE")
        obs = {"screenshot_b64": _MOCK_SCREENSHOT, "step": self.current_step}
        reward = 0.0
        info = {"action": action, "step": self.current_step}
        return obs, reward, self.done, info

    def evaluate(self) -> float:
        """
        Simple evaluation: success if ≥50% of target actions were taken.
        Returns 0.0 or 1.0.
        """
        if not self.target_actions:
            return float(random.random() > 0.5)
        matches = sum(1 for ta in self.target_actions if any(ta in a for a in self.actions_taken))
        return 1.0 if matches >= len(self.target_actions) * 0.5 else 0.0


class MockEnvPoolServer:
    """
    In-process mock of gui-rl's env_pool_server.

    Implements the same API (allocate/reset/step/evaluate/close) but runs
    entirely in-memory without any VMs or HTTP server.

    Usage:
        server = MockEnvPoolServer(tasks=[...])
        lease = server.allocate("ep1")
        obs = server.reset(lease["lease_id"], task_config)
        obs, reward, done, info = server.step(lease["lease_id"], "click(100,200)")
        score = server.evaluate(lease["lease_id"])
        server.close(lease["lease_id"])
    """

    def __init__(self, tasks: list[dict] | None = None):
        """
        Args:
            tasks: List of task configs. Each has:
                - instruction: str
                - target_actions: list[str] (for evaluation)
                - max_steps: int
        """
        self.tasks = tasks or self._default_tasks()
        self._leases: dict[str, MockTask] = {}

    def _default_tasks(self) -> list[dict]:
        """Generate default mock tasks for testing."""
        return [
            {
                "instruction": "Open the terminal and type 'ls'",
                "target_actions": ["click(50, 750)", "type('ls')"],
                "max_steps": 5,
            },
            {
                "instruction": "Create a new file called test.txt",
                "target_actions": ["click(100, 100)", "type('test.txt')"],
                "max_steps": 5,
            },
            {
                "instruction": "Change the wallpaper to the blue theme",
                "target_actions": ["click(800, 400)", "click(200, 300)"],
                "max_steps": 8,
            },
            {
                "instruction": "Open Firefox and navigate to google.com",
                "target_actions": ["click(50, 50)", "type('google.com')"],
                "max_steps": 5,
            },
            {
                "instruction": "Copy the file data.csv to the Documents folder",
                "target_actions": ["click(300, 200)", "hotkey('ctrl', 'c')", "click(100, 400)", "hotkey('ctrl', 'v')"],
                "max_steps": 10,
            },
        ]

    def allocate(self, episode_id: str) -> dict:
        lease_id = f"mock-{uuid.uuid4().hex[:12]}"
        # Don't create task yet - wait for reset
        self._leases[lease_id] = None  # type: ignore
        return {"ok": True, "lease_id": lease_id}

    def reset(self, lease_id: str, task_config: dict | None = None) -> dict:
        if lease_id not in self._leases:
            return {"ok": False, "error": f"Unknown lease: {lease_id}"}

        if task_config is None:
            task_config = random.choice(self.tasks)

        self._leases[lease_id] = MockTask(task_config)
        return {
            "ok": True,
            "observation": {
                "screenshot_b64": _MOCK_SCREENSHOT,
                "step": 0,
            },
        }

    def get_obs(self, lease_id: str) -> dict:
        task = self._leases.get(lease_id)
        if task is None:
            return {"ok": False, "error": f"No task for lease: {lease_id}"}
        return {
            "ok": True,
            "observation": {"screenshot_b64": _MOCK_SCREENSHOT, "step": task.current_step},
        }

    def step(self, lease_id: str, action: str, sleep_after_execution: float = 0.0) -> dict:
        task = self._leases.get(lease_id)
        if task is None:
            return {"ok": False, "error": f"No task for lease: {lease_id}"}

        obs, reward, done, info = task.step(action)
        return {
            "ok": True,
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info,
        }

    def evaluate(self, lease_id: str) -> dict:
        task = self._leases.get(lease_id)
        if task is None:
            return {"ok": False, "error": f"No task for lease: {lease_id}"}
        score = task.evaluate()
        return {"ok": True, "score": score}

    def heartbeat(self, lease_id: str) -> dict:
        return {"ok": lease_id in self._leases}

    def start_recording(self, lease_id: str) -> dict:
        return {"ok": True}

    def end_recording(self, lease_id: str, out_path: str = "") -> dict:
        return {"ok": True}

    def close(self, lease_id: str) -> dict:
        self._leases.pop(lease_id, None)
        return {"ok": True}


def generate_mock_trajectories(
    server: MockEnvPoolServer,
    n_trajectories: int = 100,
    agent_success_rate: float = 0.3,
) -> list[dict]:
    """
    Generate mock trajectories using the mock server.

    Returns list of trajectory dicts compatible with TrajectoryStore.import format.
    """
    from offline_rl.data.trajectory_store import Step, Trajectory

    trajectories = []
    for i in range(n_trajectories):
        task_config = random.choice(server.tasks)
        lease = server.allocate(f"gen-{i}")
        lease_id = lease["lease_id"]
        server.reset(lease_id, task_config)

        steps = []
        for step_idx in range(task_config.get("max_steps", 5)):
            # Simple mock agent: sometimes picks correct actions
            if random.random() < agent_success_rate and step_idx < len(task_config.get("target_actions", [])):
                action = task_config["target_actions"][step_idx]
            else:
                action = random.choice([
                    f"click({random.randint(0,1920)}, {random.randint(0,1080)})",
                    f"type('{random.choice(['hello','world','test','ls','cd'])}')",
                    "scroll(3)",
                    "hotkey('ctrl', 'c')",
                ])

            result = server.step(lease_id, action)
            steps.append(Step(
                step_idx=step_idx,
                action=action,
                response=f"I will {action}",
                reward=0.0,
                done=result["done"],
            ))
            if result["done"]:
                break

        eval_result = server.evaluate(lease_id)
        score = eval_result["score"]
        server.close(lease_id)

        traj = Trajectory(
            trajectory_id=str(uuid.uuid4()),
            domain="mock",
            example_id=f"task_{i}",
            instruction=task_config["instruction"],
            steps=steps,
            outcome_reward=1.0 if score > 0.5 else -1.0,
            eval_score=score,
            num_steps=len(steps),
            status="completed" if score > 0.5 else "failed",
            source="mock",
        )
        trajectories.append(traj)

    return trajectories
