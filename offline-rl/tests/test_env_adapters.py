"""Tests for AndroidWorld, WebArena, and OSWorld environment adapters."""

from __future__ import annotations

import random

from offline_rl.envs.androidworld_adapter import (
    ANDROIDWORLD_TASKS,
    MockAndroidWorldAdapter,
)
from offline_rl.envs.webarena_adapter import (
    WEBARENA_TASKS,
    MockWebArenaAdapter,
)
from offline_rl.envs.osworld_adapter import (
    OSWORLD_TASKS,
    MockOSWorldAdapter,
)
from offline_rl.envs.alfworld_adapter import (
    ALFWORLD_TASKS,
    MockAlfWorldAdapter,
)
from offline_rl.envs.base_adapter import TaskConfig, Observation


# ── Base types ──

class TestTaskConfig:
    def test_defaults(self):
        tc = TaskConfig(task_id="t1", instruction="do X", domain="test")
        assert tc.max_steps == 15
        assert tc.target_actions == []

    def test_custom(self):
        tc = TaskConfig("t2", "do Y", "web", max_steps=20, target_actions=["click(1,2)"])
        assert tc.max_steps == 20
        assert len(tc.target_actions) == 1


class TestObservation:
    def test_to_dict_minimal(self):
        obs = Observation(step=3)
        d = obs.to_dict()
        assert d["step"] == 3
        assert "screenshot_b64" not in d

    def test_to_dict_full(self):
        obs = Observation(screenshot_b64="abc", step=1, url="http://x", current_app="Chrome")
        d = obs.to_dict()
        assert d["screenshot_b64"] == "abc"
        assert d["url"] == "http://x"
        assert d["current_app"] == "Chrome"


# ── AndroidWorld Mock ──

class TestMockAndroidWorldAdapter:
    def test_task_catalog(self):
        adapter = MockAndroidWorldAdapter()
        tasks = adapter.get_task_configs()
        assert len(tasks) == len(ANDROIDWORLD_TASKS)
        domains = {t.domain for t in tasks}
        assert "sms" in domains
        assert "settings" in domains
        assert "browser" in domains

    def test_benchmark_info(self):
        adapter = MockAndroidWorldAdapter()
        info = adapter.get_benchmark_info()
        assert info["name"] == "androidworld"
        assert "click" in info["action_types"]

    def test_full_episode(self):
        adapter = MockAndroidWorldAdapter()
        lease = adapter.allocate("ep1")
        assert lease["ok"]
        lid = lease["lease_id"]
        assert lid.startswith("aw-")

        reset_r = adapter.reset(lid)
        assert reset_r["ok"]
        assert "observation" in reset_r
        assert "task" in reset_r

        # Take some actions
        r1 = adapter.step(lid, "open_app('Settings')")
        assert r1["ok"]
        assert not r1["done"]

        r2 = adapter.step(lid, "click(500, 300)")
        assert r2["ok"]

        obs = adapter.get_obs(lid)
        assert obs["ok"]

        eval_r = adapter.evaluate(lid)
        assert eval_r["ok"]
        assert "score" in eval_r

        adapter.close(lid)
        assert not adapter.heartbeat(lid)["ok"]

    def test_terminate_action(self):
        adapter = MockAndroidWorldAdapter()
        lease = adapter.allocate("ep1")
        lid = lease["lease_id"]
        adapter.reset(lid)
        r = adapter.step(lid, "terminate")
        assert r["done"]

    def test_max_steps_done(self):
        adapter = MockAndroidWorldAdapter(tasks=[
            TaskConfig("short", "test", "test", max_steps=2)
        ])
        lease = adapter.allocate("ep1")
        lid = lease["lease_id"]
        adapter.reset(lid)
        adapter.step(lid, "click(1,1)")
        r = adapter.step(lid, "click(2,2)")
        assert r["done"]

    def test_unknown_lease(self):
        adapter = MockAndroidWorldAdapter()
        r = adapter.reset("bad-lease")
        assert not r["ok"]

    def test_multiple_episodes(self):
        adapter = MockAndroidWorldAdapter()
        leases = []
        for i in range(5):
            l = adapter.allocate(f"ep{i}")
            leases.append(l["lease_id"])
            adapter.reset(l["lease_id"])
        for lid in leases:
            adapter.close(lid)


# ── WebArena Mock ──

class TestMockWebArenaAdapter:
    def test_task_catalog(self):
        adapter = MockWebArenaAdapter()
        tasks = adapter.get_task_configs()
        assert len(tasks) == len(WEBARENA_TASKS)
        domains = {t.domain for t in tasks}
        assert "shopping" in domains
        assert "cms" in domains
        assert "reddit" in domains
        assert "gitlab" in domains
        assert "maps" in domains

    def test_benchmark_info(self):
        adapter = MockWebArenaAdapter()
        info = adapter.get_benchmark_info()
        assert info["name"] == "webarena"
        assert "goto" in info["action_types"]
        assert "scroll" in info["action_types"]

    def test_full_episode(self):
        adapter = MockWebArenaAdapter()
        lease = adapter.allocate("ep1")
        assert lease["ok"]
        lid = lease["lease_id"]
        assert lid.startswith("wa-")

        reset_r = adapter.reset(lid)
        assert reset_r["ok"]
        assert "observation" in reset_r

        r1 = adapter.step(lid, "goto('shopping')")
        assert r1["ok"]
        assert not r1["done"]

        r2 = adapter.step(lid, "click(200, 300)")
        assert r2["ok"]

        obs = adapter.get_obs(lid)
        assert obs["ok"]
        assert "url" in obs["observation"]

        eval_r = adapter.evaluate(lid)
        assert eval_r["ok"]

        adapter.close(lid)

    def test_navigation_updates_url(self):
        adapter = MockWebArenaAdapter()
        lease = adapter.allocate("ep1")
        lid = lease["lease_id"]
        adapter.reset(lid)

        adapter.step(lid, "goto('shopping')")
        obs = adapter.get_obs(lid)
        assert "shopping" in obs["observation"].get("url", "")

    def test_stop_action(self):
        adapter = MockWebArenaAdapter()
        lease = adapter.allocate("ep1")
        lid = lease["lease_id"]
        adapter.reset(lid)
        r = adapter.step(lid, "stop")
        assert r["done"]

    def test_new_tab(self):
        adapter = MockWebArenaAdapter()
        lease = adapter.allocate("ep1")
        lid = lease["lease_id"]
        adapter.reset(lid)
        r = adapter.step(lid, "new_tab")
        assert r["ok"]

    def test_evaluation_with_matching_actions(self):
        random.seed(42)
        adapter = MockWebArenaAdapter(tasks=[
            TaskConfig("t1", "test", "shopping", max_steps=10,
                       target_actions=["goto('shopping')", "click(200, 300)"])
        ])
        lease = adapter.allocate("ep1")
        lid = lease["lease_id"]
        adapter.reset(lid)
        adapter.step(lid, "goto('shopping')")
        adapter.step(lid, "click(200, 300)")
        eval_r = adapter.evaluate(lid)
        assert eval_r["score"] == 1.0


# ── OSWorld Mock ──

class TestMockOSWorldAdapter:
    def test_task_catalog(self):
        adapter = MockOSWorldAdapter()
        tasks = adapter.get_task_configs()
        assert len(tasks) == len(OSWORLD_TASKS)
        domains = {t.domain for t in tasks}
        assert "chrome" in domains
        assert "terminal" in domains
        assert "gimp" in domains
        assert "vscode" in domains
        assert "settings" in domains

    def test_benchmark_info(self):
        adapter = MockOSWorldAdapter()
        info = adapter.get_benchmark_info()
        assert info["name"] == "osworld"
        assert "click" in info["action_types"]
        assert "hotkey" in info["action_types"]
        assert "scroll" in info["action_types"]

    def test_full_episode(self):
        adapter = MockOSWorldAdapter()
        lease = adapter.allocate("ep1")
        assert lease["ok"]
        lid = lease["lease_id"]
        assert lid.startswith("os-")

        reset_r = adapter.reset(lid)
        assert reset_r["ok"]
        assert "observation" in reset_r
        assert "task" in reset_r

        r1 = adapter.step(lid, "click(540, 50)")
        assert r1["ok"]
        assert not r1["done"]

        r2 = adapter.step(lid, "type('hello world')")
        assert r2["ok"]

        obs = adapter.get_obs(lid)
        assert obs["ok"]

        eval_r = adapter.evaluate(lid)
        assert eval_r["ok"]
        assert "score" in eval_r

        adapter.close(lid)

    def test_terminate_action(self):
        adapter = MockOSWorldAdapter()
        lease = adapter.allocate("ep1")
        lid = lease["lease_id"]
        adapter.reset(lid)
        r = adapter.step(lid, "terminate('done')")
        assert r["done"]

    def test_max_steps_done(self):
        adapter = MockOSWorldAdapter(tasks=[
            TaskConfig("short", "test", "test", max_steps=2)
        ])
        lease = adapter.allocate("ep1")
        lid = lease["lease_id"]
        adapter.reset(lid)
        adapter.step(lid, "click(1,1)")
        r = adapter.step(lid, "click(2,2)")
        assert r["done"]

    def test_unknown_lease(self):
        adapter = MockOSWorldAdapter()
        r = adapter.reset("bad-lease")
        assert not r["ok"]

    def test_window_tracking(self):
        adapter = MockOSWorldAdapter()
        lease = adapter.allocate("ep1")
        lid = lease["lease_id"]
        adapter.reset(lid)
        adapter.step(lid, "open Chrome browser")
        obs = adapter.get_obs(lid)
        assert obs["ok"]

    def test_accessibility_tree_in_obs(self):
        adapter = MockOSWorldAdapter()
        lease = adapter.allocate("ep1")
        lid = lease["lease_id"]
        r = adapter.reset(lid)
        assert "accessibility_tree" in r["observation"]

    def test_multiple_episodes(self):
        adapter = MockOSWorldAdapter()
        leases = []
        for i in range(5):
            l = adapter.allocate(f"ep{i}")
            leases.append(l["lease_id"])
            adapter.reset(l["lease_id"])
        for lid in leases:
            adapter.close(lid)

    def test_evaluation_with_matching_actions(self):
        random.seed(42)
        adapter = MockOSWorldAdapter(tasks=[
            TaskConfig("t1", "test", "chrome", max_steps=10,
                       target_actions=["click(540,50)", "type('hello')"])
        ])
        lease = adapter.allocate("ep1")
        lid = lease["lease_id"]
        adapter.reset(lid)
        adapter.step(lid, "click(540,50)")
        adapter.step(lid, "type('hello')")
        eval_r = adapter.evaluate(lid)
        assert eval_r["score"] == 1.0


# ── AlfWorld Mock ──

class TestMockAlfWorldAdapter:
    def test_task_catalog(self):
        adapter = MockAlfWorldAdapter()
        tasks = adapter.get_task_configs()
        assert len(tasks) == len(ALFWORLD_TASKS)
        domains = {t.domain for t in tasks}
        assert "kitchen" in domains
        assert "bedroom" in domains

    def test_benchmark_info(self):
        adapter = MockAlfWorldAdapter()
        info = adapter.get_benchmark_info()
        assert info["name"] == "alfworld"
        assert "look" in info["action_types"]
        assert "done" in info["action_types"]

    def test_full_episode(self):
        adapter = MockAlfWorldAdapter()
        lease = adapter.allocate("ep1")
        assert lease["ok"]
        lid = lease["lease_id"]
        assert lid.startswith("alf-")

        reset_r = adapter.reset(lid)
        assert reset_r["ok"]
        assert "observation" in reset_r
        assert "task" in reset_r
        assert "text_observation" in reset_r["observation"]

        r1 = adapter.step(lid, "look")
        assert r1["ok"]
        assert not r1["done"]

        r2 = adapter.step(lid, "take apple")
        assert r2["ok"]

        obs = adapter.get_obs(lid)
        assert obs["ok"]
        assert "text_observation" in obs["observation"]

        eval_r = adapter.evaluate(lid)
        assert eval_r["ok"]
        assert "score" in eval_r

        adapter.close(lid)

    def test_done_action(self):
        adapter = MockAlfWorldAdapter()
        lease = adapter.allocate("ep1")
        lid = lease["lease_id"]
        adapter.reset(lid)
        r = adapter.step(lid, "done")
        assert r["done"]

    def test_evaluation_with_matching_actions(self):
        random.seed(42)
        adapter = MockAlfWorldAdapter(tasks=[
            TaskConfig("t1", "test", "kitchen", max_steps=10,
                       target_actions=["look", "take apple", "put apple on table"])
        ])
        lease = adapter.allocate("ep1")
        lid = lease["lease_id"]
        adapter.reset(lid)
        adapter.step(lid, "look")
        adapter.step(lid, "take apple")
        adapter.step(lid, "put apple on table")
        eval_r = adapter.evaluate(lid)
        assert eval_r["score"] == 1.0


# ── Cross-adapter compatibility ──

class TestAdapterCompatibility:
    """Verify all mock adapters share the same interface."""

    def test_same_api_methods(self):
        aw = MockAndroidWorldAdapter()
        wa = MockWebArenaAdapter()
        osw = MockOSWorldAdapter()
        alf = MockAlfWorldAdapter()
        required_methods = ["allocate", "reset", "get_obs", "step", "evaluate", "close", "heartbeat", "get_task_configs"]
        for method in required_methods:
            assert hasattr(aw, method), f"AndroidWorld missing {method}"
            assert hasattr(wa, method), f"WebArena missing {method}"
            assert hasattr(osw, method), f"OSWorld missing {method}"
            assert hasattr(alf, method), f"AlfWorld missing {method}"

    def test_interchangeable_in_collection(self):
        """All adapters can be used with the same trajectory collection loop."""
        for adapter in [MockAndroidWorldAdapter(), MockWebArenaAdapter(), MockOSWorldAdapter(), MockAlfWorldAdapter()]:
            lease = adapter.allocate("test")
            lid = lease["lease_id"]
            adapter.reset(lid)
            for _ in range(3):
                action = "look" if adapter.BENCHMARK_NAME == "alfworld" else "click(100, 200)"
                r = adapter.step(lid, action)
                if r["done"]:
                    break
            adapter.evaluate(lid)
            adapter.close(lid)
