"""
AndroidWorld environment adapter.

AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents
- Paper: https://arxiv.org/abs/2405.14536
- GitHub: https://github.com/google-research/android_world
- 116 programmatic tasks across 20 real Android apps
- Dynamic task generation with programmatic evaluation

This module provides:
- MockAndroidWorldAdapter: CPU-testable mock (no emulator needed)
- AndroidWorldAdapter: Real adapter (requires android_world + Android emulator)
"""

from __future__ import annotations

import base64
import logging
import random
import uuid
from typing import Any, Dict, List, Optional

from .base_adapter import BaseEnvAdapter, Observation, TaskConfig

logger = logging.getLogger(__name__)

# Minimal 1x1 white PNG
_MOCK_SCREENSHOT = base64.b64encode(
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f"
    b"\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
).decode("ascii")

# AndroidWorld action space
ANDROID_ACTIONS = [
    "click",        # click(x, y)
    "long_press",   # long_press(x, y)
    "swipe",        # swipe(x1, y1, x2, y2)
    "type",         # type('text')
    "navigate_back",
    "navigate_home",
    "open_app",     # open_app('AppName')
    "wait",
    "terminate",
]

# Representative AndroidWorld tasks (from 20 app categories)
ANDROIDWORLD_TASKS = [
    # Communication apps
    TaskConfig("aw_sms_001", "Send an SMS to Bob saying 'Meeting at 3pm'", "sms",
               max_steps=10, target_actions=["open_app('Messages')", "click(200,100)", "type('Meeting at 3pm')", "click(900,1800)"]),
    TaskConfig("aw_email_001", "Compose an email to alice@example.com with subject 'Report'", "email",
               max_steps=12, target_actions=["open_app('Gmail')", "click(100,1800)", "type('alice@example.com')", "type('Report')"]),
    TaskConfig("aw_contacts_001", "Add a new contact named 'Charlie' with number '555-1234'", "contacts",
               max_steps=10, target_actions=["open_app('Contacts')", "click(900,100)", "type('Charlie')", "type('555-1234')"]),
    # Productivity apps
    TaskConfig("aw_calendar_001", "Create a calendar event 'Team Standup' for tomorrow at 10am", "calendar",
               max_steps=15, target_actions=["open_app('Calendar')", "click(100,100)", "type('Team Standup')", "click(500,800)"]),
    TaskConfig("aw_clock_001", "Set an alarm for 7:30 AM", "clock",
               max_steps=8, target_actions=["open_app('Clock')", "click(200,200)", "type('7:30')", "click(800,1600)"]),
    TaskConfig("aw_notes_001", "Create a new note with the title 'Shopping List'", "notes",
               max_steps=8, target_actions=["open_app('Notes')", "click(900,1800)", "type('Shopping List')"]),
    # Settings & System
    TaskConfig("aw_settings_001", "Turn on WiFi", "settings",
               max_steps=6, target_actions=["open_app('Settings')", "click(500,300)", "click(900,200)"]),
    TaskConfig("aw_settings_002", "Change display brightness to 50%", "settings",
               max_steps=8, target_actions=["open_app('Settings')", "click(500,500)", "swipe(200,800,500,800)"]),
    TaskConfig("aw_settings_003", "Enable Dark Mode", "settings",
               max_steps=8, target_actions=["open_app('Settings')", "click(500,600)", "click(500,300)"]),
    # Media apps
    TaskConfig("aw_camera_001", "Take a photo using the camera app", "camera",
               max_steps=5, target_actions=["open_app('Camera')", "click(540,1800)"]),
    TaskConfig("aw_gallery_001", "Delete the most recent photo from the gallery", "gallery",
               max_steps=8, target_actions=["open_app('Gallery')", "click(200,200)", "click(900,100)", "click(500,1200)"]),
    # Browser
    TaskConfig("aw_browser_001", "Open Chrome and navigate to wikipedia.org", "browser",
               max_steps=8, target_actions=["open_app('Chrome')", "click(540,100)", "type('wikipedia.org')", "click(540,400)"]),
    TaskConfig("aw_browser_002", "Search for 'machine learning' on Google in Chrome", "browser",
               max_steps=8, target_actions=["open_app('Chrome')", "click(540,100)", "type('machine learning')", "click(540,400)"]),
    # Files
    TaskConfig("aw_files_001", "Create a new folder named 'Documents' in the file manager", "files",
               max_steps=10, target_actions=["open_app('Files')", "click(900,100)", "click(500,400)", "type('Documents')"]),
    # Maps
    TaskConfig("aw_maps_001", "Search for 'coffee shops near me' in Google Maps", "maps",
               max_steps=8, target_actions=["open_app('Maps')", "click(540,100)", "type('coffee shops near me')"]),
]


class _MockAndroidTask:
    """Simulated Android task with simple state machine."""

    def __init__(self, config: TaskConfig):
        self.config = config
        self.current_step = 0
        self.actions_taken: list[str] = []
        self.done = False
        self.current_app: Optional[str] = None

    def step(self, action: str) -> tuple:
        self.current_step += 1
        self.actions_taken.append(action)

        # Parse app opening
        if action.startswith("open_app("):
            app_name = action.split("'")[1] if "'" in action else "Unknown"
            self.current_app = app_name

        self.done = (
            self.current_step >= self.config.max_steps
            or action == "terminate"
        )

        obs = Observation(
            screenshot_b64=_MOCK_SCREENSHOT,
            step=self.current_step,
            current_app=self.current_app,
        )
        return obs, 0.0, self.done, {"action": action, "step": self.current_step}

    def evaluate(self) -> float:
        if not self.config.target_actions:
            return float(random.random() > 0.5)
        matches = sum(
            1 for ta in self.config.target_actions
            if any(ta in a for a in self.actions_taken)
        )
        return 1.0 if matches >= len(self.config.target_actions) * 0.5 else 0.0


class MockAndroidWorldAdapter(BaseEnvAdapter):
    """
    CPU-testable mock of AndroidWorld environment.

    Simulates 116-style AndroidWorld tasks with 15 representative mock tasks
    across communication, productivity, settings, media, browser, and file domains.
    No Android emulator required.

    Usage:
        adapter = MockAndroidWorldAdapter()
        lease = adapter.allocate("ep1")
        adapter.reset(lease["lease_id"])
        result = adapter.step(lease["lease_id"], "open_app('Settings')")
        score = adapter.evaluate(lease["lease_id"])
        adapter.close(lease["lease_id"])
    """

    BENCHMARK_NAME = "androidworld"
    ACTION_TYPES = ANDROID_ACTIONS

    def __init__(self, tasks: Optional[List[TaskConfig]] = None):
        self.tasks = tasks or ANDROIDWORLD_TASKS
        self._leases: Dict[str, Optional[_MockAndroidTask]] = {}

    def allocate(self, episode_id: str) -> dict:
        lease_id = f"aw-{uuid.uuid4().hex[:12]}"
        self._leases[lease_id] = None
        return {"ok": True, "lease_id": lease_id}

    def reset(self, lease_id: str, task_config: Optional[dict] = None) -> dict:
        if lease_id not in self._leases:
            return {"ok": False, "error": f"Unknown lease: {lease_id}"}

        if task_config:
            config = TaskConfig(**task_config)
        else:
            config = random.choice(self.tasks)

        self._leases[lease_id] = _MockAndroidTask(config)
        return {
            "ok": True,
            "observation": Observation(
                screenshot_b64=_MOCK_SCREENSHOT, step=0
            ).to_dict(),
            "task": {"task_id": config.task_id, "instruction": config.instruction},
        }

    def get_obs(self, lease_id: str) -> dict:
        task = self._leases.get(lease_id)
        if task is None:
            return {"ok": False, "error": f"No active task for lease: {lease_id}"}
        return {
            "ok": True,
            "observation": Observation(
                screenshot_b64=_MOCK_SCREENSHOT,
                step=task.current_step,
                current_app=task.current_app,
            ).to_dict(),
        }

    def step(self, lease_id: str, action: str, sleep_after: float = 0.0) -> dict:
        task = self._leases.get(lease_id)
        if task is None:
            return {"ok": False, "error": f"No active task for lease: {lease_id}"}
        obs, reward, done, info = task.step(action)
        return {"ok": True, "observation": obs.to_dict(), "reward": reward, "done": done, "info": info}

    def evaluate(self, lease_id: str) -> dict:
        task = self._leases.get(lease_id)
        if task is None:
            return {"ok": False, "error": f"No active task for lease: {lease_id}"}
        return {"ok": True, "score": task.evaluate()}

    def close(self, lease_id: str) -> dict:
        self._leases.pop(lease_id, None)
        return {"ok": True}

    def heartbeat(self, lease_id: str) -> dict:
        return {"ok": lease_id in self._leases}

    def get_task_configs(self) -> List[TaskConfig]:
        return self.tasks


class AndroidWorldAdapter(BaseEnvAdapter):
    """
    Real AndroidWorld adapter (requires android_world package + Android emulator).

    Wraps google-research/android_world to expose gui-rl-compatible API.
    Falls back to mock if android_world is not installed.

    Requires:
        pip install android_world  (or clone from github)
        Android emulator (AVD) running
    """

    BENCHMARK_NAME = "androidworld"
    ACTION_TYPES = ANDROID_ACTIONS

    def __init__(self, avd_name: str = "pixel_6_api_33", **kwargs):
        self.avd_name = avd_name
        self._env = None
        self._leases: Dict[str, Any] = {}

        try:
            import android_world  # noqa: F401
            self._has_real_env = True
            logger.info("AndroidWorld package found. Using real environment.")
        except ImportError:
            self._has_real_env = False
            logger.warning(
                "android_world not installed. Use MockAndroidWorldAdapter for testing. "
                "Install with: pip install android_world"
            )

    def allocate(self, episode_id: str) -> dict:
        if not self._has_real_env:
            return {"ok": False, "error": "android_world not installed"}
        lease_id = f"aw-real-{uuid.uuid4().hex[:12]}"
        self._leases[lease_id] = {"episode_id": episode_id, "env": None, "step": 0}
        return {"ok": True, "lease_id": lease_id}

    def reset(self, lease_id: str, task_config: Optional[dict] = None) -> dict:
        if not self._has_real_env:
            return {"ok": False, "error": "android_world not installed"}
        if lease_id not in self._leases:
            return {"ok": False, "error": f"Unknown lease: {lease_id}"}

        from android_world import AndroidWorldEnv  # type: ignore

        task_name = task_config.get("task_name", "default") if task_config else "default"
        env = AndroidWorldEnv(task_name=task_name, avd_name=self.avd_name)
        obs = env.reset()
        self._leases[lease_id]["env"] = env
        self._leases[lease_id]["step"] = 0
        return {"ok": True, "observation": {"screenshot_b64": "", "step": 0}}

    def get_obs(self, lease_id: str) -> dict:
        lease = self._leases.get(lease_id)
        if not lease or not lease.get("env"):
            return {"ok": False, "error": "No active environment"}
        return {"ok": True, "observation": {"step": lease["step"]}}

    def step(self, lease_id: str, action: str, sleep_after: float = 0.0) -> dict:
        lease = self._leases.get(lease_id)
        if not lease or not lease.get("env"):
            return {"ok": False, "error": "No active environment"}
        env = lease["env"]
        obs, reward, done, info = env.step(action)
        lease["step"] += 1
        return {"ok": True, "observation": {"step": lease["step"]}, "reward": reward, "done": done, "info": info}

    def evaluate(self, lease_id: str) -> dict:
        lease = self._leases.get(lease_id)
        if not lease or not lease.get("env"):
            return {"ok": False, "error": "No active environment"}
        score = float(lease["env"].evaluate())
        return {"ok": True, "score": score}

    def close(self, lease_id: str) -> dict:
        lease = self._leases.pop(lease_id, None)
        if lease and lease.get("env"):
            try:
                lease["env"].close()
            except Exception:
                pass
        return {"ok": True}

    def get_task_configs(self) -> List[TaskConfig]:
        return ANDROIDWORLD_TASKS
