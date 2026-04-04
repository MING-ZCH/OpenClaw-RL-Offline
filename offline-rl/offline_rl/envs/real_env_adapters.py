"""
Real environment adapters for production use.

These adapters connect to actual OSWorld / AndroidWorld / WebArena environments.
They require the respective packages to be installed and environments to be running.

Adapted to work with the offline-rl data pipeline.

Usage:
    from offline_rl.envs.real_env_adapters import RealOSWorldAdapter
    adapter = RealOSWorldAdapter(task_config, max_steps=30)
    obs = adapter.reset()
    obs, reward, done, info = adapter.step({"type": "click", "coordinate": [100, 200]})
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .process_reward import OSWorldProcessReward, AndroidWorldProcessReward, WebArenaProcessReward

logger = logging.getLogger(__name__)


# ============================================================
# Action conversion utilities
# ============================================================

def convert_action_osworld(action):
    # type: (Dict[str, Any]) -> Dict[str, Any]
    """Convert standardized action dict to OSWorld format."""
    action_type = action.get("type", "wait")

    converters = {
        "click": lambda a: {"action": "click", "coordinate": a.get("coordinate", [0, 0])},
        "right_click": lambda a: {"action": "right_click", "coordinate": a.get("coordinate", [0, 0])},
        "double_click": lambda a: {"action": "double_click", "coordinate": a.get("coordinate", [0, 0])},
        "type": lambda a: {"action": "type", "text": a.get("text", "")},
        "hotkey": lambda a: {"action": "hotkey", "keys": a.get("keys", [])},
        "scroll": lambda a: {
            "action": "scroll",
            "coordinate": a.get("coordinate", [0, 0]),
            "direction": a.get("direction", "down"),
            "amount": a.get("amount", 3),
        },
        "wait": lambda a: {"action": "wait", "duration": a.get("duration", 1.0)},
        "terminate": lambda a: {"action": "terminate", "answer": a.get("answer", "")},
    }

    fn = converters.get(action_type)
    if fn:
        return fn(action)
    return {"action": "wait"}


def convert_action_android(action):
    # type: (Dict[str, Any]) -> Dict[str, Any]
    """Convert standardized action dict to AndroidWorld format."""
    action_type = action.get("type", "wait")

    converters = {
        "click": lambda a: {"action_type": "click", "coordinate": a.get("coordinate", [0, 0])},
        "long_press": lambda a: {"action_type": "long_press", "coordinate": a.get("coordinate", [0, 0]),
                                  "duration": a.get("duration", 1.0)},
        "swipe": lambda a: {"action_type": "swipe",
                            "start_coordinate": a.get("start_coordinate", [0, 0]),
                            "end_coordinate": a.get("end_coordinate", [0, 0]),
                            "duration": a.get("duration", 0.5)},
        "type": lambda a: {"action_type": "type", "text": a.get("text", "")},
        "navigate_back": lambda a: {"action_type": "navigate_back"},
        "navigate_home": lambda a: {"action_type": "navigate_home"},
        "open_app": lambda a: {"action_type": "open_app", "app_name": a.get("app_name", "")},
        "wait": lambda a: {"action_type": "wait", "duration": a.get("duration", 1.0)},
        "terminate": lambda a: {"action_type": "terminate"},
    }

    fn = converters.get(action_type)
    if fn:
        return fn(action)
    return {"action_type": "wait"}


def format_observation_standard(obs, task_description="", task_id="unknown", step=0):
    # type: (Dict[str, Any], str, str, int) -> Dict[str, Any]
    """
    Format raw environment observation to standardized format.

    Handles screenshot encoding (numpy → base64), accessibility tree,
    UI hierarchy, URL, and HTML fields.
    """
    formatted = {
        "task": task_description,
        "task_id": task_id,
        "step": step,
    }

    # Screenshot handling
    if "screenshot" in obs:
        screenshot = obs["screenshot"]
        try:
            import numpy as np
            if isinstance(screenshot, np.ndarray):
                import base64
                from io import BytesIO
                try:
                    from PIL import Image
                    pil_image = Image.fromarray(screenshot)
                    buffered = BytesIO()
                    pil_image.save(buffered, format="PNG")
                    formatted["screenshot"] = base64.b64encode(buffered.getvalue()).decode()
                except ImportError:
                    # PIL not available, store raw shape info
                    formatted["screenshot_shape"] = list(screenshot.shape)
            else:
                formatted["screenshot"] = screenshot
        except ImportError:
            formatted["screenshot"] = str(screenshot)

    # Accessibility tree (OSWorld)
    for key in ("accessibility_tree", "a11y_tree"):
        if key in obs:
            formatted["accessibility_tree"] = obs[key]
            break

    # UI hierarchy (AndroidWorld)
    for key in ("ui_hierarchy", "view_hierarchy"):
        if key in obs:
            formatted["ui_tree"] = obs[key]
            break

    # Web content
    if "url" in obs:
        formatted["url"] = obs["url"]
    if "html" in obs:
        formatted["html"] = obs["html"]

    # Current app (AndroidWorld)
    if "current_app" in obs:
        formatted["current_app"] = obs["current_app"]

    return formatted


# ============================================================
# Real environment adapter base
# ============================================================

class RealEnvAdapterBase:
    """
    Base class for real environment adapters.

    Provides shared functionality: step counting, trajectory recording,
    observation formatting, and process reward computation.
    """

    def __init__(self, task_config, max_steps=30, compute_process_reward=True):
        # type: (Dict[str, Any], int, bool) -> None
        self.task_config = task_config
        self.max_steps = max_steps
        self.compute_process_reward_flag = compute_process_reward

        self.task_description = task_config.get("task_description", "")
        self.task_id = task_config.get("task_id", "unknown")
        self.current_step = 0
        self.current_trajectory = []  # type: List[Dict[str, Any]]
        self.env = None  # subclass sets this

        self.prm = None  # type: Optional[ProcessRewardModel]  # subclass sets

    def _format_obs(self, raw_obs):
        # type: (Dict[str, Any]) -> Dict[str, Any]
        return format_observation_standard(
            raw_obs,
            task_description=self.task_description,
            task_id=self.task_id,
            step=self.current_step,
        )

    def get_trajectory(self):
        # type: () -> List[Dict[str, Any]]
        """Return the recorded trajectory for the current episode."""
        return list(self.current_trajectory)


# ============================================================
# Real OSWorld adapter
# ============================================================

class RealOSWorldAdapter(RealEnvAdapterBase):
    """
    Adapter for real OSWorld environments (Docker VM based).

    Requires: pip install osworld
    """

    def __init__(self, task_config, max_steps=30, compute_process_reward=True):
        super().__init__(task_config, max_steps, compute_process_reward)
        self.prm = OSWorldProcessReward(task_description=self.task_description)
        self.env = self._init_env(task_config)

    def _init_env(self, task_config):
        try:
            from osworld.environment import Environment
            return Environment(
                snapshot_name=task_config.get("snapshot_name", "ubuntu"),
                task_name=task_config.get("task_name", "default"),
            )
        except ImportError:
            raise ImportError(
                "OSWorld is not installed. Install with:\n"
                "  pip install -e git+https://github.com/xlang-ai/OSWorld.git#egg=osworld"
            )

    def reset(self):
        # type: () -> Dict[str, Any]
        raw = self.env.reset()
        self.current_step = 0
        self.current_trajectory = []
        self.prm.reset()
        obs = self._format_obs(raw)
        self.current_trajectory.append(obs)
        return obs

    def step(self, action):
        # type: (Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]
        osworld_action = convert_action_osworld(action)
        raw_obs, env_reward, done, info = self.env.step(osworld_action)
        self.current_step += 1

        if self.current_step >= self.max_steps:
            done = True

        obs = self._format_obs(raw_obs)

        if self.compute_process_reward_flag:
            success = self.env.evaluate() if done else None
            reward = self.prm.compute_reward(obs, action, done, success=success)
        else:
            reward = env_reward

        self.current_trajectory.append({"action": action, "observation": obs, "reward": reward})
        info["step"] = self.current_step
        info["task_id"] = self.task_id
        return obs, reward, done, info

    def evaluate(self):
        # type: () -> bool
        try:
            return self.env.evaluate()
        except Exception as e:
            logger.error("OSWorld evaluation error: %s", e)
            return False


# ============================================================
# Real AndroidWorld adapter
# ============================================================

class RealAndroidWorldAdapter(RealEnvAdapterBase):
    """
    Adapter for real AndroidWorld environments (AVD based).

    Requires: pip install android-world
    """

    def __init__(self, task_config, max_steps=20, avd_name="pixel_6_api_33",
                 compute_process_reward=True):
        super().__init__(task_config, max_steps, compute_process_reward)
        self.avd_name = avd_name
        self.prm = AndroidWorldProcessReward(task_description=self.task_description)
        self.env = self._init_env(task_config)

    def _init_env(self, task_config):
        try:
            from android_world import AndroidWorldEnv
            return AndroidWorldEnv(
                task_name=task_config.get("task_name", "default"),
                avd_name=self.avd_name,
            )
        except ImportError:
            raise ImportError(
                "AndroidWorld is not installed. Install with:\n"
                "  pip install -e git+https://github.com/google-research/android-world.git#egg=android-world"
            )

    def reset(self):
        raw = self.env.reset()
        self.current_step = 0
        self.current_trajectory = []
        self.prm.reset()
        obs = self._format_obs(raw)
        self.current_trajectory.append(obs)
        return obs

    def step(self, action):
        android_action = convert_action_android(action)
        raw_obs, reward, done, info = self.env.step(android_action)
        self.current_step += 1

        if self.current_step >= self.max_steps:
            done = True

        obs = self._format_obs(raw_obs)

        if self.compute_process_reward_flag:
            success = info.get("success", False) if done else None
            reward = self.prm.compute_reward(obs, action, done, success=success)
        else:
            if done:
                reward = 1.0 if info.get("success", False) else 0.0

        self.current_trajectory.append({"action": action, "observation": obs, "reward": reward})
        info["step"] = self.current_step
        info["task_id"] = self.task_id
        return obs, reward, done, info

    def evaluate(self):
        try:
            return self.env.evaluate()
        except Exception as e:
            logger.error("AndroidWorld evaluation error: %s", e)
            return False
