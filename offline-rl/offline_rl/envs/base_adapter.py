"""
Base environment adapter interface.

All environment adapters implement this interface, which mirrors gui-rl's
env_pool_server HTTP API (allocate/reset/step/evaluate/close).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskConfig:
    """Standard task configuration across all benchmarks."""
    task_id: str
    instruction: str
    domain: str  # benchmark-specific domain (e.g., "chrome", "gmail", "shopping")
    max_steps: int = 15
    target_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """Standard observation format returned by all adapters."""
    screenshot_b64: Optional[str] = None  # Base64 PNG screenshot
    step: int = 0
    accessibility_tree: Optional[str] = None  # A11y tree or UI hierarchy
    url: Optional[str] = None  # Current URL (web environments)
    current_app: Optional[str] = None  # Current app (Android)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"step": self.step}
        if self.screenshot_b64:
            d["screenshot_b64"] = self.screenshot_b64
        if self.accessibility_tree:
            d["accessibility_tree"] = self.accessibility_tree
        if self.url:
            d["url"] = self.url
        if self.current_app:
            d["current_app"] = self.current_app
        d.update(self.extra)
        return d


class BaseEnvAdapter(abc.ABC):
    """
    Abstract environment adapter compatible with gui-rl's env_pool_server API.

    Subclasses implement benchmark-specific environment logic. Both mock
    and real implementations share this interface.
    """

    # Benchmark identification
    BENCHMARK_NAME: str = "base"
    ACTION_TYPES: List[str] = []

    @abc.abstractmethod
    def allocate(self, episode_id: str) -> dict:
        """Allocate an environment instance. Returns {ok, lease_id}."""

    @abc.abstractmethod
    def reset(self, lease_id: str, task_config: Optional[dict] = None) -> dict:
        """Reset environment with a task. Returns {ok, observation}."""

    @abc.abstractmethod
    def get_obs(self, lease_id: str) -> dict:
        """Get current observation. Returns {ok, observation}."""

    @abc.abstractmethod
    def step(self, lease_id: str, action: str, sleep_after: float = 0.0) -> dict:
        """Execute action. Returns {ok, observation, reward, done, info}."""

    @abc.abstractmethod
    def evaluate(self, lease_id: str) -> dict:
        """Evaluate task success. Returns {ok, score}."""

    @abc.abstractmethod
    def close(self, lease_id: str) -> dict:
        """Release environment. Returns {ok}."""

    def heartbeat(self, lease_id: str) -> dict:
        """Check if lease is alive."""
        return {"ok": True}

    @abc.abstractmethod
    def get_task_configs(self) -> List[TaskConfig]:
        """Return available task configurations for this benchmark."""

    def get_benchmark_info(self) -> dict:
        """Return benchmark metadata."""
        return {
            "name": self.BENCHMARK_NAME,
            "action_types": self.ACTION_TYPES,
            "num_tasks": len(self.get_task_configs()),
        }
