"""
OSWorld environment adapter.

OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer
Environments.
- Paper: https://arxiv.org/abs/2404.07972
- GitHub: https://github.com/xlang-ai/OSWorld
- 369 computer tasks across Ubuntu desktop apps (Chrome, VS Code, GIMP, etc.)
- Evaluation via programmatic success checking + screenshot comparison

This module provides:
- MockOSWorldAdapter: CPU-testable mock (no VM needed)
- OSWorldAdapter: Real adapter (requires osworld + VM or cloud instances)
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

# OSWorld action space (matches upstream gui-rl)
OSWORLD_ACTIONS = [
    "click",         # click(x, y)
    "right_click",   # right_click(x, y)
    "double_click",  # double_click(x, y)
    "type",          # type('text')
    "hotkey",        # hotkey('ctrl', 'c')
    "scroll",        # scroll(x, y, direction, amount)
    "wait",          # wait(duration)
    "terminate",     # terminate(answer)
]

# Representative OSWorld tasks across desktop application domains
OSWORLD_TASKS = [
    # Browser / Web
    TaskConfig("os_chrome_001", "Open Chrome and search for 'OpenClaw-RL' on Google", "chrome",
               max_steps=15,
               target_actions=["click(540,50)", "type('OpenClaw-RL')", "hotkey('Return')"]),
    TaskConfig("os_chrome_002", "Navigate to github.com and star the OpenClaw-RL repository", "chrome",
               max_steps=20,
               target_actions=["click(540,50)", "type('github.com')", "click(600,400)"]),
    TaskConfig("os_chrome_003", "Download a PDF from arxiv.org about reinforcement learning", "chrome",
               max_steps=20,
               target_actions=["click(540,50)", "type('arxiv.org')", "click(400,300)"]),
    # Office / LibreOffice
    TaskConfig("os_libreoffice_001", "Create a new document and type 'Hello World' as the title", "libreoffice",
               max_steps=10,
               target_actions=["click(400,400)", "type('Hello World')"]),
    TaskConfig("os_libreoffice_002", "Open an existing spreadsheet and sum column A values", "libreoffice",
               max_steps=15,
               target_actions=["click(200,200)", "type('=SUM(A:A)')", "hotkey('Return')"]),
    # File Manager
    TaskConfig("os_files_001", "Copy file 'test.txt' from Desktop to Documents folder", "files",
               max_steps=15,
               target_actions=["right_click(300,200)", "click(350,400)", "click(200,300)"]),
    TaskConfig("os_files_002", "Create a new folder named 'Project' on the Desktop", "files",
               max_steps=10,
               target_actions=["right_click(500,500)", "click(550,350)", "type('Project')"]),
    # Image Editing / GIMP
    TaskConfig("os_gimp_001", "Open GIMP and resize an image to 512x512 pixels", "gimp",
               max_steps=20,
               target_actions=["click(100,50)", "click(150,300)", "type('512')", "type('512')"]),
    # Terminal
    TaskConfig("os_terminal_001", "Open a terminal and list files in the home directory", "terminal",
               max_steps=8,
               target_actions=["hotkey('ctrl','alt','t')", "type('ls ~')", "hotkey('Return')"]),
    TaskConfig("os_terminal_002", "Create a Python virtual environment named 'myenv'", "terminal",
               max_steps=10,
               target_actions=["hotkey('ctrl','alt','t')", "type('python3 -m venv myenv')", "hotkey('Return')"]),
    # VS Code / Code Editor
    TaskConfig("os_vscode_001", "Open VS Code and create a new Python file", "vscode",
               max_steps=12,
               target_actions=["click(100,50)", "click(150,100)", "type('hello.py')"]),
    # System Settings
    TaskConfig("os_settings_001", "Change the desktop wallpaper in system settings", "settings",
               max_steps=15,
               target_actions=["click(800,50)", "click(400,200)", "click(300,400)"]),
    TaskConfig("os_settings_002", "Connect to a WiFi network from the system tray", "settings",
               max_steps=10,
               target_actions=["click(1800,10)", "click(1700,100)", "click(1700,200)"]),
    # Media Player / VLC
    TaskConfig("os_vlc_001", "Open VLC and play a sample video file", "vlc",
               max_steps=8,
               target_actions=["click(100,50)", "click(150,200)", "click(400,300)"]),
    # Multi-app composite task
    TaskConfig("os_composite_001", "Download an image from Chrome, then open it in GIMP and crop it", "composite",
               max_steps=30,
               target_actions=["click(540,50)", "type('sample image')", "right_click(400,400)", "click(450,500)"]),
]


class _MockOSWorldTask:
    """Simulated OSWorld task with simple state machine."""

    def __init__(self, config: TaskConfig):
        self.config = config
        self.current_step = 0
        self.actions_taken: list[str] = []
        self.done = False
        self.active_window: str = "desktop"
        self.url: str = ""

    def step(self, action: str) -> tuple:
        self.current_step += 1
        self.actions_taken.append(action)

        # Simulate window / app state transitions
        action_lower = action.lower()
        if "chrome" in action_lower or "browser" in action_lower:
            self.active_window = "chrome"
        elif "libreoffice" in action_lower or "writer" in action_lower:
            self.active_window = "libreoffice"
        elif "gimp" in action_lower:
            self.active_window = "gimp"
        elif "terminal" in action_lower or "ctrl" in action_lower and "alt" in action_lower:
            self.active_window = "terminal"
        elif "vscode" in action_lower or "code" in action_lower:
            self.active_window = "vscode"
        elif "files" in action_lower or "nautilus" in action_lower:
            self.active_window = "files"
        elif "settings" in action_lower:
            self.active_window = "settings"

        self.done = (
            self.current_step >= self.config.max_steps
            or action.startswith("terminate")
        )

        obs = Observation(
            screenshot_b64=_MOCK_SCREENSHOT,
            step=self.current_step,
            accessibility_tree=f"Window: {self.active_window}, Step: {self.current_step}",
            url=f"https://example.com/{self.active_window}" if self.active_window == "chrome" else None,
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


class MockOSWorldAdapter(BaseEnvAdapter):
    """
    CPU-testable mock of OSWorld environment.

    Simulates 369-style OSWorld tasks with 15 representative mock tasks
    across browser, office, file manager, image editor, terminal, VS Code,
    settings, media player, and composite domains.
    No VM or cloud instance required.

    Usage:
        adapter = MockOSWorldAdapter()
        lease = adapter.allocate("ep1")
        adapter.reset(lease["lease_id"])
        result = adapter.step(lease["lease_id"], "click(540,50)")
        score = adapter.evaluate(lease["lease_id"])
        adapter.close(lease["lease_id"])
    """

    BENCHMARK_NAME = "osworld"
    ACTION_TYPES = OSWORLD_ACTIONS

    def __init__(self, tasks: Optional[List[TaskConfig]] = None):
        self.tasks = tasks or OSWORLD_TASKS
        self._leases: Dict[str, Optional[_MockOSWorldTask]] = {}

    def allocate(self, episode_id: str) -> dict:
        lease_id = f"os-{uuid.uuid4().hex[:12]}"
        self._leases[lease_id] = None
        return {"ok": True, "lease_id": lease_id}

    def reset(self, lease_id: str, task_config: Optional[dict] = None) -> dict:
        if lease_id not in self._leases:
            return {"ok": False, "error": f"Unknown lease: {lease_id}"}
        if task_config:
            config = TaskConfig(**task_config)
        else:
            config = random.choice(self.tasks)
        self._leases[lease_id] = _MockOSWorldTask(config)
        return {
            "ok": True,
            "observation": Observation(
                screenshot_b64=_MOCK_SCREENSHOT,
                step=0,
                accessibility_tree="Desktop ready",
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
                accessibility_tree=f"Window: {task.active_window}",
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
        score = task.evaluate()
        return {"ok": True, "score": score}

    def close(self, lease_id: str) -> dict:
        if lease_id in self._leases:
            del self._leases[lease_id]
        return {"ok": True}

    def get_task_configs(self) -> List[TaskConfig]:
        return list(self.tasks)


class OSWorldAdapter(BaseEnvAdapter):
    """
    Real OSWorld adapter using osworld environment + VM instances.

    Requires:
    - pip install osworld  (or git clone from xlang-ai/OSWorld)
    - VM snapshot configured (Ubuntu desktop)

    Action format strings follow gui-rl convention:
        "click(x, y)", "type('text')", "hotkey('ctrl','c')", etc.

    For development without real VMs, use MockOSWorldAdapter instead.
    """

    BENCHMARK_NAME = "osworld"
    ACTION_TYPES = OSWORLD_ACTIONS

    def __init__(
        self,
        vm_config: Optional[Dict[str, Any]] = None,
        max_steps: int = 30,
        compute_process_reward: bool = False,
    ):
        try:
            from desktop_env.envs.desktop_env import DesktopEnv
        except ImportError:
            raise ImportError(
                "osworld is not installed. Install from: "
                "https://github.com/xlang-ai/OSWorld"
            )
        self._vm_config = vm_config or {}
        self._max_steps = max_steps
        self._compute_process_reward = compute_process_reward
        self._env_cls = DesktopEnv
        self._leases: Dict[str, Any] = {}

    def allocate(self, episode_id: str) -> dict:
        lease_id = f"os-{uuid.uuid4().hex[:12]}"
        env = self._env_cls(**self._vm_config)
        self._leases[lease_id] = {
            "env": env,
            "step": 0,
            "actions": [],
            "done": False,
        }
        return {"ok": True, "lease_id": lease_id}

    def reset(self, lease_id: str, task_config: Optional[dict] = None) -> dict:
        if lease_id not in self._leases:
            return {"ok": False, "error": f"Unknown lease: {lease_id}"}
        state = self._leases[lease_id]
        obs = state["env"].reset(task_config or {})
        state["step"] = 0
        state["actions"] = []
        state["done"] = False
        return {
            "ok": True,
            "observation": self._format_obs(obs, 0),
            "task": task_config or {},
        }

    def get_obs(self, lease_id: str) -> dict:
        if lease_id not in self._leases:
            return {"ok": False, "error": f"Unknown lease: {lease_id}"}
        state = self._leases[lease_id]
        obs = state["env"].observation()
        return {"ok": True, "observation": self._format_obs(obs, state["step"])}

    def step(self, lease_id: str, action: str, sleep_after: float = 0.0) -> dict:
        if lease_id not in self._leases:
            return {"ok": False, "error": f"Unknown lease: {lease_id}"}
        state = self._leases[lease_id]
        parsed = self._parse_action(action)
        obs, reward, done, info = state["env"].step(parsed)
        state["step"] += 1
        state["actions"].append(action)
        if state["step"] >= self._max_steps:
            done = True
        state["done"] = done
        return {
            "ok": True,
            "observation": self._format_obs(obs, state["step"]),
            "reward": reward,
            "done": done,
            "info": info,
        }

    def evaluate(self, lease_id: str) -> dict:
        if lease_id not in self._leases:
            return {"ok": False, "error": f"Unknown lease: {lease_id}"}
        state = self._leases[lease_id]
        try:
            score = float(state["env"].evaluate())
        except Exception as e:
            logger.warning("Evaluation error: %s", e)
            score = 0.0
        return {"ok": True, "score": score}

    def close(self, lease_id: str) -> dict:
        if lease_id in self._leases:
            try:
                self._leases[lease_id]["env"].close()
            except Exception:
                pass
            del self._leases[lease_id]
        return {"ok": True}

    def get_task_configs(self) -> List[TaskConfig]:
        return list(OSWORLD_TASKS)

    @staticmethod
    def _format_obs(obs: Any, step: int) -> dict:
        d = {"step": step}
        if isinstance(obs, dict):
            if "screenshot" in obs:
                import io
                from PIL import Image as PILImage
                img = obs["screenshot"]
                if hasattr(img, "shape"):  # numpy array
                    pil = PILImage.fromarray(img)
                    buf = io.BytesIO()
                    pil.save(buf, format="PNG")
                    d["screenshot_b64"] = base64.b64encode(buf.getvalue()).decode()
                elif isinstance(img, str):
                    d["screenshot_b64"] = img
            if "accessibility_tree" in obs:
                d["accessibility_tree"] = obs["accessibility_tree"]
            if "url" in obs:
                d["url"] = obs["url"]
        return d

    @staticmethod
    def _parse_action(action_str: str) -> dict:
        """Parse action string like 'click(540, 50)' into dict."""
        action_str = action_str.strip()
        if "(" in action_str:
            name = action_str[:action_str.index("(")].strip()
            args_str = action_str[action_str.index("(") + 1:action_str.rindex(")")]
        else:
            name = action_str
            args_str = ""

        result = {"action": name}
        if name in ("click", "right_click", "double_click"):
            parts = [p.strip() for p in args_str.split(",")]
            if len(parts) >= 2:
                result["coordinate"] = [int(parts[0]), int(parts[1])]
        elif name == "type":
            text = args_str.strip().strip("'\"")
            result["text"] = text
        elif name == "hotkey":
            keys = [k.strip().strip("'\"") for k in args_str.split(",")]
            result["keys"] = keys
        elif name == "scroll":
            parts = [p.strip().strip("'\"") for p in args_str.split(",")]
            if len(parts) >= 2:
                result["coordinate"] = [int(parts[0]), int(parts[1])]
            if len(parts) >= 3:
                result["direction"] = parts[2]
            if len(parts) >= 4:
                result["amount"] = int(parts[3])
        elif name == "wait":
            if args_str.strip():
                result["duration"] = float(args_str.strip())
        elif name == "terminate":
            if args_str.strip():
                result["answer"] = args_str.strip().strip("'\"")
        return result
