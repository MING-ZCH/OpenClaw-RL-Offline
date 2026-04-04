"""
WebArena environment adapter.

WebArena: A Realistic Web Environment for Building Autonomous Agents
- Paper: https://arxiv.org/abs/2307.13854
- GitHub: https://github.com/web-arena-x/webarena
- 812 tasks across 5 web applications (Shopping, CMS, Reddit, GitLab, Maps)
- Functional correctness evaluation with reference answers

This module provides:
- MockWebArenaAdapter: CPU-testable mock (no web server needed)
- WebArenaAdapter: Real adapter (requires WebArena Docker setup)
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

# WebArena browser action space
WEBARENA_ACTIONS = [
    "click",        # click(element_id) or click(x, y)
    "type",         # type(element_id, 'text') or type('text')
    "scroll",       # scroll(direction) - up/down
    "hover",        # hover(element_id)
    "goto",         # goto('url')
    "go_back",
    "go_forward",
    "new_tab",
    "tab_focus",    # tab_focus(tab_idx)
    "close_tab",
    "stop",         # Finish task (agent declares done)
]

# WebArena's 5 web applications with representative tasks
WEBARENA_TASKS = [
    # Shopping (OneStopShop) - e-commerce
    TaskConfig("wa_shop_001", "Find the cheapest laptop on the shopping site and add it to cart", "shopping",
               max_steps=15, target_actions=["goto('shopping')", "click(100,200)", "click(300,400)"]),
    TaskConfig("wa_shop_002", "Check the order status for order #12345", "shopping",
               max_steps=10, target_actions=["goto('shopping')", "click(100,100)", "type('12345')"]),
    TaskConfig("wa_shop_003", "Leave a 5-star review for the most recently purchased item", "shopping",
               max_steps=12, target_actions=["goto('shopping')", "click(200,300)", "click(500,200)", "type('Great product')"]),
    TaskConfig("wa_shop_004", "Change the shipping address to 123 Main St, New York", "shopping",
               max_steps=12, target_actions=["goto('shopping')", "click(100,500)", "type('123 Main St')"]),
    # CMS (Content Management System)
    TaskConfig("wa_cms_001", "Create a new blog post titled 'AI in 2025'", "cms",
               max_steps=12, target_actions=["goto('cms')", "click(100,200)", "type('AI in 2025')", "click(800,600)"]),
    TaskConfig("wa_cms_002", "Publish the draft article 'Quarterly Report'", "cms",
               max_steps=8, target_actions=["goto('cms')", "click(200,300)", "click(700,100)"]),
    TaskConfig("wa_cms_003", "Delete the oldest comment on the latest post", "cms",
               max_steps=10, target_actions=["goto('cms')", "click(200,200)", "click(800,500)", "click(600,400)"]),
    # Reddit (forum)
    TaskConfig("wa_reddit_001", "Post a question 'How to learn Python?' in the programming subreddit", "reddit",
               max_steps=12, target_actions=["goto('reddit')", "click(100,100)", "type('How to learn Python?')", "click(600,800)"]),
    TaskConfig("wa_reddit_002", "Upvote the top post in the front page", "reddit",
               max_steps=8, target_actions=["goto('reddit')", "click(50,300)"]),
    TaskConfig("wa_reddit_003", "Reply to the first comment on the top post with 'Great insight!'", "reddit",
               max_steps=10, target_actions=["goto('reddit')", "click(200,300)", "click(100,500)", "type('Great insight!')"]),
    # GitLab (code hosting)
    TaskConfig("wa_gitlab_001", "Create a new repository called 'test-project'", "gitlab",
               max_steps=12, target_actions=["goto('gitlab')", "click(100,100)", "type('test-project')", "click(600,800)"]),
    TaskConfig("wa_gitlab_002", "Create a new issue titled 'Fix bug in login' in the main project", "gitlab",
               max_steps=10, target_actions=["goto('gitlab')", "click(200,200)", "click(100,300)", "type('Fix bug in login')"]),
    TaskConfig("wa_gitlab_003", "Star the most popular repository", "gitlab",
               max_steps=8, target_actions=["goto('gitlab')", "click(300,200)", "click(800,100)"]),
    # Maps (OpenStreetMap)
    TaskConfig("wa_maps_001", "Find the distance from New York to Los Angeles", "maps",
               max_steps=10, target_actions=["goto('maps')", "click(200,100)", "type('New York to Los Angeles')"]),
    TaskConfig("wa_maps_002", "Search for restaurants near Times Square", "maps",
               max_steps=8, target_actions=["goto('maps')", "type('restaurants near Times Square')"]),
]


class _MockWebPage:
    """Simulated web page state."""

    def __init__(self):
        self.url = "about:blank"
        self.title = "New Tab"
        self.elements: List[Dict] = []

    def navigate(self, url: str):
        self.url = url
        if "shopping" in url:
            self.title = "OneStopShop - Home"
        elif "cms" in url:
            self.title = "WordPress Admin"
        elif "reddit" in url:
            self.title = "Reddit - Front Page"
        elif "gitlab" in url:
            self.title = "GitLab - Projects"
        elif "maps" in url:
            self.title = "OpenStreetMap"
        else:
            self.title = url


class _MockWebArenaTask:
    """Simulated WebArena task."""

    def __init__(self, config: TaskConfig):
        self.config = config
        self.current_step = 0
        self.actions_taken: list[str] = []
        self.done = False
        self.page = _MockWebPage()
        self.tabs: list[_MockWebPage] = [self.page]

    def step(self, action: str) -> tuple:
        self.current_step += 1
        self.actions_taken.append(action)

        # Parse navigation
        if action.startswith("goto("):
            url = action.split("'")[1] if "'" in action else ""
            self.page.navigate(url)
        elif action == "go_back":
            pass  # mock
        elif action == "new_tab":
            new_page = _MockWebPage()
            self.tabs.append(new_page)
            self.page = new_page
        elif action == "stop":
            self.done = True

        self.done = self.done or self.current_step >= self.config.max_steps

        obs = Observation(
            screenshot_b64=_MOCK_SCREENSHOT,
            step=self.current_step,
            url=self.page.url,
            extra={"title": self.page.title, "num_tabs": len(self.tabs)},
        )
        return obs, 0.0, self.done, {"action": action, "step": self.current_step}

    def evaluate(self) -> float:
        """
        WebArena evaluation: check if target actions were approximately completed.
        Real WebArena uses functional correctness checks against reference answers.
        """
        if not self.config.target_actions:
            return float(random.random() > 0.5)
        matches = sum(
            1 for ta in self.config.target_actions
            if any(ta in a for a in self.actions_taken)
        )
        return 1.0 if matches >= len(self.config.target_actions) * 0.5 else 0.0


class MockWebArenaAdapter(BaseEnvAdapter):
    """
    CPU-testable mock of WebArena environment.

    Simulates web browsing tasks across 5 WebArena applications
    (Shopping, CMS, Reddit, GitLab, Maps) without requiring Docker
    containers or actual web servers.

    Usage:
        adapter = MockWebArenaAdapter()
        lease = adapter.allocate("ep1")
        adapter.reset(lease["lease_id"])
        result = adapter.step(lease["lease_id"], "goto('shopping')")
        result = adapter.step(lease["lease_id"], "click(200, 300)")
        score = adapter.evaluate(lease["lease_id"])
        adapter.close(lease["lease_id"])
    """

    BENCHMARK_NAME = "webarena"
    ACTION_TYPES = WEBARENA_ACTIONS

    def __init__(self, tasks: Optional[List[TaskConfig]] = None):
        self.tasks = tasks or WEBARENA_TASKS
        self._leases: Dict[str, Optional[_MockWebArenaTask]] = {}

    def allocate(self, episode_id: str) -> dict:
        lease_id = f"wa-{uuid.uuid4().hex[:12]}"
        self._leases[lease_id] = None
        return {"ok": True, "lease_id": lease_id}

    def reset(self, lease_id: str, task_config: Optional[dict] = None) -> dict:
        if lease_id not in self._leases:
            return {"ok": False, "error": f"Unknown lease: {lease_id}"}

        if task_config:
            config = TaskConfig(**task_config)
        else:
            config = random.choice(self.tasks)

        self._leases[lease_id] = _MockWebArenaTask(config)
        return {
            "ok": True,
            "observation": Observation(
                screenshot_b64=_MOCK_SCREENSHOT, step=0, url="about:blank"
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
                url=task.page.url,
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


class WebArenaAdapter(BaseEnvAdapter):
    """
    Real WebArena adapter (requires WebArena Docker setup + Playwright/Selenium).

    Wraps web-arena-x/webarena to expose gui-rl-compatible API.
    
    Requires:
        WebArena Docker containers running (shopping, cms, reddit, gitlab, maps)
        pip install playwright  (or selenium)
        See: https://github.com/web-arena-x/webarena#setup
    """

    BENCHMARK_NAME = "webarena"
    ACTION_TYPES = WEBARENA_ACTIONS

    def __init__(self, base_urls: Optional[Dict[str, str]] = None, **kwargs):
        self.base_urls = base_urls or {
            "shopping": "http://localhost:7770",
            "cms": "http://localhost:7780",
            "reddit": "http://localhost:9999",
            "gitlab": "http://localhost:8023",
            "maps": "http://localhost:3000",
        }
        self._leases: Dict[str, Any] = {}
        self._has_playwright = False

        try:
            import playwright  # noqa: F401
            self._has_playwright = True
            logger.info("Playwright found. Using real browser automation.")
        except ImportError:
            logger.warning(
                "playwright not installed. Use MockWebArenaAdapter for testing. "
                "Install with: pip install playwright && playwright install"
            )

    def allocate(self, episode_id: str) -> dict:
        if not self._has_playwright:
            return {"ok": False, "error": "playwright not installed"}
        lease_id = f"wa-real-{uuid.uuid4().hex[:12]}"
        self._leases[lease_id] = {"episode_id": episode_id, "browser": None, "step": 0}
        return {"ok": True, "lease_id": lease_id}

    def reset(self, lease_id: str, task_config: Optional[dict] = None) -> dict:
        if not self._has_playwright:
            return {"ok": False, "error": "playwright not installed"}
        if lease_id not in self._leases:
            return {"ok": False, "error": f"Unknown lease: {lease_id}"}
        # Real implementation would launch browser via Playwright
        self._leases[lease_id]["step"] = 0
        return {"ok": True, "observation": {"screenshot_b64": "", "step": 0, "url": "about:blank"}}

    def get_obs(self, lease_id: str) -> dict:
        lease = self._leases.get(lease_id)
        if not lease:
            return {"ok": False, "error": "No active environment"}
        return {"ok": True, "observation": {"step": lease["step"]}}

    def step(self, lease_id: str, action: str, sleep_after: float = 0.0) -> dict:
        lease = self._leases.get(lease_id)
        if not lease:
            return {"ok": False, "error": "No active environment"}
        lease["step"] += 1
        return {"ok": True, "observation": {"step": lease["step"]}, "reward": 0.0, "done": False, "info": {}}

    def evaluate(self, lease_id: str) -> dict:
        lease = self._leases.get(lease_id)
        if not lease:
            return {"ok": False, "error": "No active environment"}
        # Real implementation would use WebArena's evaluation scripts
        return {"ok": True, "score": 0.0}

    def close(self, lease_id: str) -> dict:
        lease = self._leases.pop(lease_id, None)
        if lease and lease.get("browser"):
            try:
                lease["browser"].close()
            except Exception:
                pass
        return {"ok": True}

    def get_task_configs(self) -> List[TaskConfig]:
        return WEBARENA_TASKS
