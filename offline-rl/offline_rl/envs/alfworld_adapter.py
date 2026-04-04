"""
AlfWorld environment adapter.

AlfWorld is a text-based embodied benchmark built on top of ALFRED-style
household tasks. This module provides:
- MockAlfWorldAdapter: CPU-testable mock with representative tasks
- AlfWorldAdapter: optional real adapter scaffold for custom AlfWorld envs
"""

from __future__ import annotations

import importlib.util
import logging
import random
import uuid
from typing import Any, Callable, Dict, List, Optional

from .base_adapter import BaseEnvAdapter, Observation, TaskConfig

logger = logging.getLogger(__name__)


ALFWORLD_ACTIONS = [
    "look",
    "inventory",
    "go",
    "open",
    "close",
    "take",
    "put",
    "heat",
    "cool",
    "clean",
    "use",
    "done",
]


ALFWORLD_TASKS = [
    TaskConfig(
        "alf_kitchen_001",
        "Find the apple, heat it, and place it on the dining table.",
        "kitchen",
        max_steps=12,
        target_actions=["look", "open fridge", "take apple", "heat apple", "put apple on table", "done"],
    ),
    TaskConfig(
        "alf_kitchen_002",
        "Pick up the mug, clean it in the sink, and place it in the cabinet.",
        "kitchen",
        max_steps=12,
        target_actions=["look", "take mug", "clean mug", "open cabinet", "put mug in cabinet", "done"],
    ),
    TaskConfig(
        "alf_kitchen_003",
        "Find the soda can, cool it in the fridge, and put it on the counter.",
        "kitchen",
        max_steps=12,
        target_actions=["look", "take soda", "open fridge", "cool soda", "put soda on counter", "done"],
    ),
    TaskConfig(
        "alf_bedroom_001",
        "Locate the book and place it on the desk.",
        "bedroom",
        max_steps=10,
        target_actions=["look", "take book", "put book on desk", "done"],
    ),
    TaskConfig(
        "alf_livingroom_001",
        "Pick up the remote control and put it on the sofa.",
        "living_room",
        max_steps=10,
        target_actions=["look", "take remote", "put remote on sofa", "done"],
    ),
]


class _MockAlfWorldTask:
    """Simulated AlfWorld task with simple room and inventory state."""

    _ROOM_OBJECTS = {
        "kitchen": ["apple", "mug", "soda", "fridge", "sink", "cabinet", "counter", "table"],
        "bedroom": ["book", "desk", "lamp", "drawer"],
        "living_room": ["remote", "sofa", "table", "lamp"],
    }

    def __init__(self, config: TaskConfig):
        self.config = config
        self.current_step = 0
        self.actions_taken: List[str] = []
        self.done = False
        self.current_room = config.domain
        self.inventory: List[str] = []

    def _build_observation(self, action: str = "look") -> Observation:
        visible = ", ".join(self._ROOM_OBJECTS.get(self.current_room, ["table", "drawer"]))
        inventory = ", ".join(self.inventory) if self.inventory else "nothing"
        text = (
            "You are in the %s. You see %s. "
            "Last action: %s. Inventory: %s."
        ) % (self.current_room.replace("_", " "), visible, action, inventory)
        admissible = [
            "look",
            "inventory",
            "open fridge",
            "open cabinet",
            "take apple",
            "take mug",
            "take soda",
            "take book",
            "take remote",
            "put apple on table",
            "put mug in cabinet",
            "put soda on counter",
            "put book on desk",
            "put remote on sofa",
            "heat apple",
            "clean mug",
            "cool soda",
            "done",
        ]
        return Observation(
            step=self.current_step,
            accessibility_tree="room=%s" % self.current_room,
            extra={
                "text_observation": text,
                "admissible_commands": admissible,
                "inventory": list(self.inventory),
            },
        )

    def step(self, action: str) -> tuple:
        self.current_step += 1
        self.actions_taken.append(action)

        normalized = action.strip().lower()
        if normalized.startswith("go "):
            self.current_room = normalized.split(" ", 1)[1].replace(" ", "_")
        elif normalized.startswith("take "):
            obj = normalized.split(" ", 1)[1]
            if obj not in self.inventory:
                self.inventory.append(obj)
        elif normalized.startswith("put "):
            obj = normalized.split(" ", 1)[1].split(" on ")[0].split(" in ")[0]
            if obj in self.inventory:
                self.inventory.remove(obj)

        self.done = self.current_step >= self.config.max_steps or normalized in {"done", "stop", "terminate"}
        obs = self._build_observation(action)
        return obs, 0.0, self.done, {"action": action, "step": self.current_step}

    def evaluate(self) -> float:
        if not self.config.target_actions:
            return float(random.random() > 0.5)
        matches = sum(
            1 for target in self.config.target_actions
            if any(target in action.lower() for action in self.actions_taken)
        )
        return 1.0 if matches >= len(self.config.target_actions) * 0.5 else 0.0


class MockAlfWorldAdapter(BaseEnvAdapter):
    """CPU-testable mock of AlfWorld with representative household tasks."""

    BENCHMARK_NAME = "alfworld"
    ACTION_TYPES = ALFWORLD_ACTIONS

    def __init__(self, tasks: Optional[List[TaskConfig]] = None):
        self.tasks = tasks or ALFWORLD_TASKS
        self._leases: Dict[str, Optional[_MockAlfWorldTask]] = {}

    def allocate(self, episode_id: str) -> dict:
        lease_id = "alf-%s" % uuid.uuid4().hex[:12]
        self._leases[lease_id] = None
        return {"ok": True, "lease_id": lease_id}

    def reset(self, lease_id: str, task_config: Optional[dict] = None) -> dict:
        if lease_id not in self._leases:
            return {"ok": False, "error": "Unknown lease: %s" % lease_id}

        config = TaskConfig(**task_config) if task_config else random.choice(self.tasks)
        task = _MockAlfWorldTask(config)
        self._leases[lease_id] = task
        obs = task._build_observation("look")
        return {
            "ok": True,
            "observation": obs.to_dict(),
            "task": {"task_id": config.task_id, "instruction": config.instruction},
        }

    def get_obs(self, lease_id: str) -> dict:
        task = self._leases.get(lease_id)
        if task is None:
            return {"ok": False, "error": "No active task for lease: %s" % lease_id}
        return {"ok": True, "observation": task._build_observation().to_dict()}

    def step(self, lease_id: str, action: str, sleep_after: float = 0.0) -> dict:
        task = self._leases.get(lease_id)
        if task is None:
            return {"ok": False, "error": "No active task for lease: %s" % lease_id}
        obs, reward, done, info = task.step(action)
        return {"ok": True, "observation": obs.to_dict(), "reward": reward, "done": done, "info": info}

    def evaluate(self, lease_id: str) -> dict:
        task = self._leases.get(lease_id)
        if task is None:
            return {"ok": False, "error": "No active task for lease: %s" % lease_id}
        return {"ok": True, "score": task.evaluate()}

    def close(self, lease_id: str) -> dict:
        self._leases.pop(lease_id, None)
        return {"ok": True}

    def heartbeat(self, lease_id: str) -> dict:
        return {"ok": lease_id in self._leases}

    def get_task_configs(self) -> List[TaskConfig]:
        return list(self.tasks)


class AlfWorldAdapter(BaseEnvAdapter):
    """
    Optional real AlfWorld adapter scaffold.

    AlfWorld packaging differs across setups, so this adapter accepts a custom
    env_factory callable. The mock adapter should be used for CPU testing.
    """

    BENCHMARK_NAME = "alfworld"
    ACTION_TYPES = ALFWORLD_ACTIONS

    def __init__(self, env_factory: Optional[Callable[[TaskConfig], Any]] = None):
        self.env_factory = env_factory
        self._leases: Dict[str, Dict[str, Any]] = {}
        try:
            spec = importlib.util.find_spec("alfworld")
            if spec is None:
                raise ImportError
            self._has_real_env = True
        except ImportError:
            self._has_real_env = False
            logger.warning(
                "alfworld not installed. Use MockAlfWorldAdapter for CPU testing. "
                "Install with: pip install alfworld"
            )

    def allocate(self, episode_id: str) -> dict:
        if not self._has_real_env and self.env_factory is None:
            return {"ok": False, "error": "alfworld not installed"}
        lease_id = "alf-real-%s" % uuid.uuid4().hex[:12]
        self._leases[lease_id] = {"episode_id": episode_id, "env": None, "config": None, "step": 0}
        return {"ok": True, "lease_id": lease_id}

    def reset(self, lease_id: str, task_config: Optional[dict] = None) -> dict:
        state = self._leases.get(lease_id)
        if state is None:
            return {"ok": False, "error": "Unknown lease: %s" % lease_id}
        if self.env_factory is None:
            return {
                "ok": False,
                "error": "Real AlfWorld integration requires env_factory=... when constructing AlfWorldAdapter",
            }
        config = TaskConfig(**task_config) if task_config else random.choice(ALFWORLD_TASKS)
        env = self.env_factory(config)
        obs = env.reset()
        state.update({"env": env, "config": config, "step": 0})
        observation = Observation(
            step=0,
            extra={"text_observation": str(obs), "admissible_commands": []},
        )
        return {
            "ok": True,
            "observation": observation.to_dict(),
            "task": {"task_id": config.task_id, "instruction": config.instruction},
        }

    def get_obs(self, lease_id: str) -> dict:
        state = self._leases.get(lease_id)
        if state is None or state.get("env") is None:
            return {"ok": False, "error": "No active task for lease: %s" % lease_id}
        return {"ok": True, "observation": Observation(step=state["step"]).to_dict()}

    def step(self, lease_id: str, action: str, sleep_after: float = 0.0) -> dict:
        state = self._leases.get(lease_id)
        if state is None or state.get("env") is None:
            return {"ok": False, "error": "No active task for lease: %s" % lease_id}
        obs, reward, done, info = state["env"].step(action)
        state["step"] += 1
        observation = Observation(
            step=state["step"],
            extra={"text_observation": str(obs), "admissible_commands": info.get("admissible_commands", [])},
        )
        return {"ok": True, "observation": observation.to_dict(), "reward": reward, "done": done, "info": info}

    def evaluate(self, lease_id: str) -> dict:
        state = self._leases.get(lease_id)
        if state is None or state.get("env") is None:
            return {"ok": False, "error": "No active task for lease: %s" % lease_id}
        env = state["env"]
        if hasattr(env, "evaluate"):
            return {"ok": True, "score": float(env.evaluate())}
        return {"ok": True, "score": 0.0}

    def close(self, lease_id: str) -> dict:
        state = self._leases.pop(lease_id, None)
        if state and state.get("env") is not None and hasattr(state["env"], "close"):
            state["env"].close()
        return {"ok": True}

    def get_task_configs(self) -> List[TaskConfig]:
        return list(ALFWORLD_TASKS)