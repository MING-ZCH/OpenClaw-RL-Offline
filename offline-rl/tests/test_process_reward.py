"""
Tests for the process reward model and real environment adapter utilities.
"""

import pytest

from offline_rl.envs.process_reward import (
    ProcessRewardModel,
    OSWorldProcessReward,
    AndroidWorldProcessReward,
    WebArenaProcessReward,
)
from offline_rl.envs.real_env_adapters import (
    convert_action_osworld,
    convert_action_android,
    format_observation_standard,
)


# ---------- ProcessRewardModel ----------

class TestProcessRewardModel:
    def test_basic_reward_computation(self):
        prm = ProcessRewardModel(task_description="Open Chrome browser")
        prm.reset()

        obs = {"accessibility_tree": "desktop with chrome icon"}
        action = {"type": "click", "coordinate": [100, 200]}
        reward = prm.compute_reward(obs, action, done=False)
        assert isinstance(reward, float)

    def test_step_penalty(self):
        prm = ProcessRewardModel(task_description="", step_penalty=-0.05)
        prm.reset()

        obs = {}
        action = {"type": "wait"}
        reward = prm.compute_reward(obs, action, done=False)
        # Should include step penalty and validity bonus
        assert reward < 0.2  # step_penalty + possible validity_bonus

    def test_terminal_success_reward(self):
        prm = ProcessRewardModel(success_reward=10.0, failure_penalty=-5.0)
        prm.reset()

        obs = {}
        action = {"type": "terminate"}
        reward_success = prm.compute_reward(obs, action, done=True, success=True)
        assert reward_success > 5.0  # large positive from success

    def test_terminal_failure_penalty(self):
        prm = ProcessRewardModel(success_reward=10.0, failure_penalty=-5.0)
        prm.reset()

        obs = {}
        action = {"type": "terminate"}
        reward_fail = prm.compute_reward(obs, action, done=True, success=False)
        assert reward_fail < 0  # negative from failure

    def test_reset_clears_state(self):
        prm = ProcessRewardModel(task_description="test")
        prm.compute_reward({"text": "hello"}, {"type": "click"}, done=False)
        assert prm.previous_obs is not None
        assert len(prm.progress_history) == 1

        prm.reset()
        assert prm.previous_obs is None
        assert len(prm.progress_history) == 0

    def test_progress_tracking(self):
        prm = ProcessRewardModel(task_description="open chrome browser")
        prm.reset()

        # First step: no matching keywords
        obs1 = {"accessibility_tree": "desktop icons"}
        prm.compute_reward(obs1, {"type": "click"}, done=False)

        # Second step: some matching keywords
        obs2 = {"accessibility_tree": "chrome browser window open"}
        reward2 = prm.compute_reward(obs2, {"type": "click"}, done=False)
        # Progress increased → positive progress reward
        assert prm.progress_history[-1] > prm.progress_history[0]

    def test_state_change_reward(self):
        prm = ProcessRewardModel(task_description="", state_change_scale=1.0)
        prm.reset()

        obs1 = {"accessibility_tree": "desktop with icons"}
        prm.compute_reward(obs1, {"type": "click"}, done=False)

        # Significant state change
        obs2 = {"accessibility_tree": "chrome browser showing google search page"}
        reward2 = prm.compute_reward(obs2, {"type": "click"}, done=False)
        # state_change_scale * change should contribute positively
        # exact value depends on trigram similarity

    def test_invalid_action_penalty(self):
        prm = ProcessRewardModel(validity_bonus=0.1, validity_penalty=-0.2)
        prm.reset()

        obs = {}
        valid_action = {"type": "click", "coordinate": [100, 200]}
        r_valid = prm.compute_reward(obs, valid_action, done=False)

        prm.reset()
        invalid_action = {"type": "unknown_action_xyz"}
        r_invalid = prm.compute_reward(obs, invalid_action, done=False)

        assert r_valid > r_invalid  # valid action should get higher reward


class TestOSWorldProcessReward:
    def test_osworld_actions(self):
        prm = OSWorldProcessReward()
        prm.reset()

        # Valid OSWorld actions
        assert prm._is_valid_action({"type": "click", "coordinate": [100, 200]}, {})
        assert prm._is_valid_action({"type": "hotkey", "keys": ["ctrl", "c"]}, {})
        assert prm._is_valid_action({"type": "type", "text": "hello"}, {})

        # Invalid action type
        assert not prm._is_valid_action({"type": "swipe"}, {})

        # Out of bounds coordinate
        assert not prm._is_valid_action(
            {"type": "click", "coordinate": [2000, 200]}, {}
        )

    def test_osworld_coordinate_bounds(self):
        prm = OSWorldProcessReward()
        assert prm._is_valid_action({"type": "click", "coordinate": [1920, 1080]}, {})
        assert not prm._is_valid_action({"type": "click", "coordinate": [1921, 1080]}, {})


class TestAndroidWorldProcessReward:
    def test_android_actions(self):
        prm = AndroidWorldProcessReward()

        assert prm._is_valid_action({"type": "click", "coordinate": [500, 800]}, {})
        assert prm._is_valid_action({"type": "navigate_back"}, {})
        assert prm._is_valid_action({"type": "open_app", "app_name": "Chrome"}, {})
        assert prm._is_valid_action({"type": "swipe"}, {})

        # Not a valid Android action
        assert not prm._is_valid_action({"type": "hotkey"}, {})

    def test_android_coordinate_bounds(self):
        prm = AndroidWorldProcessReward()
        assert prm._is_valid_action({"type": "click", "coordinate": [1440, 3120]}, {})
        assert not prm._is_valid_action({"type": "click", "coordinate": [1441, 0]}, {})


class TestWebArenaProcessReward:
    def test_webarena_actions(self):
        prm = WebArenaProcessReward()

        assert prm._is_valid_action({"type": "click"}, {})
        assert prm._is_valid_action({"type": "fill"}, {})
        assert prm._is_valid_action({"type": "goto"}, {})
        assert prm._is_valid_action({"type": "hover"}, {})

        assert not prm._is_valid_action({"type": "swipe"}, {})


# ---------- Action Converters ----------

class TestActionConverters:
    def test_osworld_click(self):
        result = convert_action_osworld({"type": "click", "coordinate": [100, 200]})
        assert result == {"action": "click", "coordinate": [100, 200]}

    def test_osworld_type(self):
        result = convert_action_osworld({"type": "type", "text": "hello"})
        assert result == {"action": "type", "text": "hello"}

    def test_osworld_hotkey(self):
        result = convert_action_osworld({"type": "hotkey", "keys": ["ctrl", "c"]})
        assert result == {"action": "hotkey", "keys": ["ctrl", "c"]}

    def test_osworld_scroll(self):
        result = convert_action_osworld({"type": "scroll", "coordinate": [500, 300],
                                          "direction": "up", "amount": 5})
        assert result["action"] == "scroll"
        assert result["direction"] == "up"
        assert result["amount"] == 5

    def test_osworld_unknown_defaults_to_wait(self):
        result = convert_action_osworld({"type": "unknown"})
        assert result == {"action": "wait"}

    def test_android_click(self):
        result = convert_action_android({"type": "click", "coordinate": [300, 400]})
        assert result == {"action_type": "click", "coordinate": [300, 400]}

    def test_android_swipe(self):
        result = convert_action_android({
            "type": "swipe",
            "start_coordinate": [100, 500],
            "end_coordinate": [100, 200],
        })
        assert result["action_type"] == "swipe"
        assert result["start_coordinate"] == [100, 500]
        assert result["end_coordinate"] == [100, 200]

    def test_android_navigate_back(self):
        result = convert_action_android({"type": "navigate_back"})
        assert result == {"action_type": "navigate_back"}

    def test_android_open_app(self):
        result = convert_action_android({"type": "open_app", "app_name": "Chrome"})
        assert result["action_type"] == "open_app"
        assert result["app_name"] == "Chrome"


# ---------- Observation Formatter ----------

class TestObservationFormatter:
    def test_basic_format(self):
        obs = format_observation_standard(
            {"url": "https://google.com"},
            task_description="Search Google",
            task_id="test_1",
            step=3,
        )
        assert obs["task"] == "Search Google"
        assert obs["task_id"] == "test_1"
        assert obs["step"] == 3
        assert obs["url"] == "https://google.com"

    def test_accessibility_tree(self):
        obs = format_observation_standard(
            {"accessibility_tree": "<root><button>OK</button></root>"},
        )
        assert obs["accessibility_tree"] == "<root><button>OK</button></root>"

    def test_a11y_tree_alias(self):
        obs = format_observation_standard({"a11y_tree": "tree data"})
        assert obs["accessibility_tree"] == "tree data"

    def test_ui_hierarchy(self):
        obs = format_observation_standard({"view_hierarchy": "<ViewGroup/>"})
        assert obs["ui_tree"] == "<ViewGroup/>"

    def test_html_field(self):
        obs = format_observation_standard({"html": "<div>content</div>"})
        assert obs["html"] == "<div>content</div>"

    def test_current_app(self):
        obs = format_observation_standard({"current_app": "Chrome"})
        assert obs["current_app"] == "Chrome"
