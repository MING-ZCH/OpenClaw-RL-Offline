"""
Process Reward Model for GUI environment adapters.

Computes fine-grained step-level rewards based on state changes, action validity,
and task progress estimation.

This module provides a common ProcessRewardModel that can be used across
different environment adapters (OSWorld, AndroidWorld, WebArena) to generate
dense reward signals for offline RL training.

References:
    - OpenClaw-RL PRM (Process Reward Model) judge
"""

from typing import Any, Dict, List, Optional


class ProcessRewardModel:
    """
    Compute process (step-level) rewards from observation changes and action quality.

    Reward components:
        1. Task progress change (estimated heuristically)
        2. Action validity check
        3. State change magnitude (encourage meaningful interactions)
        4. Efficiency penalty (discourage idle steps)
        5. Terminal outcome reward

    Usage:
        prm = ProcessRewardModel(task_description="Open Chrome and search...")
        for step in episode:
            reward = prm.compute_reward(obs, action, done)
    """

    def __init__(
        self,
        task_description="",
        progress_scale=2.0,
        validity_bonus=0.1,
        validity_penalty=-0.2,
        state_change_scale=0.5,
        step_penalty=-0.01,
        success_reward=10.0,
        failure_penalty=-5.0,
    ):
        self.task_description = task_description
        self.progress_scale = progress_scale
        self.validity_bonus = validity_bonus
        self.validity_penalty = validity_penalty
        self.state_change_scale = state_change_scale
        self.step_penalty = step_penalty
        self.success_reward = success_reward
        self.failure_penalty = failure_penalty

        self.previous_obs = None  # type: Optional[Dict[str, Any]]
        self.progress_history = []  # type: List[float]

    def reset(self):
        """Reset internal state for a new episode."""
        self.previous_obs = None
        self.progress_history = []

    def compute_reward(self, obs, action, done, success=None):
        # type: (Dict[str, Any], Dict[str, Any], bool, Optional[bool]) -> float
        """
        Compute step-level process reward.

        Args:
            obs: Current observation dict
            action: Action dict with at least a 'type' key
            done: Whether episode ended
            success: Explicit task success flag (used at terminal step)

        Returns:
            Float reward for this step.
        """
        reward = 0.0

        # 1. Task progress reward
        progress = self._estimate_progress(obs)
        if self.progress_history:
            delta = progress - self.progress_history[-1]
            reward += delta * self.progress_scale
        self.progress_history.append(progress)

        # 2. Action validity
        if self._is_valid_action(action, obs):
            reward += self.validity_bonus
        else:
            reward += self.validity_penalty

        # 3. State change reward
        if self.previous_obs is not None:
            change = self._compute_state_change(self.previous_obs, obs)
            reward += change * self.state_change_scale

        # 4. Step penalty (efficiency)
        reward += self.step_penalty

        # 5. Terminal reward
        if done:
            if success is True:
                reward += self.success_reward
            elif success is False:
                reward += self.failure_penalty

        self.previous_obs = obs
        return reward

    # ------------------------------------------------------------------
    # Heuristic helpers — override these for domain-specific behaviour
    # ------------------------------------------------------------------

    def _estimate_progress(self, obs):
        # type: (Dict[str, Any]) -> float
        """
        Estimate task completion progress from observation.

        Default heuristic: count keyword matches between task description
        and observable text (accessibility tree, UI hierarchy, etc.).
        Override for domain-specific progress estimation.
        """
        if not self.task_description:
            return 0.0

        # Collect all text content from the observation
        text_sources = []
        for key in ("accessibility_tree", "ui_tree", "html", "response", "text"):
            if key in obs and isinstance(obs[key], str):
                text_sources.append(obs[key].lower())

        if not text_sources:
            return 0.0

        combined = " ".join(text_sources)
        keywords = self.task_description.lower().split()
        if not keywords:
            return 0.0

        matches = sum(1 for kw in keywords if kw in combined)
        return min(matches / len(keywords), 1.0)

    def _is_valid_action(self, action, obs):
        # type: (Dict[str, Any], Dict[str, Any]) -> bool
        """
        Check basic action validity.

        Validates:
            - Action has a known type
            - Click/tap coordinates are within screen bounds
        """
        action_type = action.get("type", "")

        known_types = {
            # OSWorld actions
            "click", "right_click", "double_click", "type", "hotkey",
            "scroll", "wait", "terminate",
            # AndroidWorld actions
            "long_press", "swipe", "navigate_back", "navigate_home", "open_app",
            # WebArena actions
            "fill", "select", "hover", "goto",
        }

        if action_type not in known_types:
            return False

        # Coordinate bounds check for spatial actions
        if action_type in ("click", "right_click", "double_click", "long_press", "scroll"):
            coord = action.get("coordinate", action.get("coordinates"))
            if coord is not None:
                if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                    x, y = coord[0], coord[1]
                    if not (0 <= x <= 3840 and 0 <= y <= 2160):
                        return False

        return True

    def _compute_state_change(self, prev_obs, curr_obs):
        # type: (Dict[str, Any], Dict[str, Any]) -> float
        """
        Estimate magnitude of meaningful state change between observations.

        Compares text-based observation fields. Returns a value in [0, 1].
        """
        prev_texts = []
        curr_texts = []

        for key in ("accessibility_tree", "ui_tree", "html", "url", "response"):
            if key in prev_obs and isinstance(prev_obs[key], str):
                prev_texts.append(prev_obs[key])
            if key in curr_obs and isinstance(curr_obs[key], str):
                curr_texts.append(curr_obs[key])

        prev_combined = " ".join(prev_texts)
        curr_combined = " ".join(curr_texts)

        if not prev_combined and not curr_combined:
            return 0.0

        # Simple character-level change ratio
        if not prev_combined:
            return 1.0
        if not curr_combined:
            return 0.5

        # Jaccard-like similarity on character trigrams
        def trigrams(s):
            return set(s[i:i+3] for i in range(max(len(s)-2, 1)))

        t_prev = trigrams(prev_combined)
        t_curr = trigrams(curr_combined)

        if not t_prev and not t_curr:
            return 0.0

        intersection = len(t_prev & t_curr)
        union = len(t_prev | t_curr)

        similarity = intersection / max(union, 1)
        return 1.0 - similarity  # higher change = higher reward


class OSWorldProcessReward(ProcessRewardModel):
    """Process reward model tailored for OSWorld desktop tasks."""

    # Valid OSWorld action types
    OSWORLD_ACTIONS = {
        "click", "right_click", "double_click", "type", "hotkey",
        "scroll", "wait", "terminate",
    }

    def _is_valid_action(self, action, obs):
        action_type = action.get("type", "")
        if action_type not in self.OSWORLD_ACTIONS:
            return False
        # Desktop coordinate bounds: typically 1920×1080
        if action_type in ("click", "right_click", "double_click", "scroll"):
            coord = action.get("coordinate")
            if coord and isinstance(coord, (list, tuple)) and len(coord) >= 2:
                if not (0 <= coord[0] <= 1920 and 0 <= coord[1] <= 1080):
                    return False
        return True


class AndroidWorldProcessReward(ProcessRewardModel):
    """Process reward model tailored for AndroidWorld tasks."""

    ANDROID_ACTIONS = {
        "click", "long_press", "swipe", "type",
        "navigate_back", "navigate_home", "open_app",
        "wait", "terminate",
    }

    def _is_valid_action(self, action, obs):
        action_type = action.get("type", "")
        if action_type not in self.ANDROID_ACTIONS:
            return False
        # Android screen bounds: typically 1080×2340 or 1080×1920
        if action_type in ("click", "long_press"):
            coord = action.get("coordinate")
            if coord and isinstance(coord, (list, tuple)) and len(coord) >= 2:
                if not (0 <= coord[0] <= 1440 and 0 <= coord[1] <= 3120):
                    return False
        return True


class WebArenaProcessReward(ProcessRewardModel):
    """Process reward model tailored for WebArena tasks."""

    WEBARENA_ACTIONS = {
        "click", "type", "fill", "select", "hover",
        "scroll", "goto", "wait", "terminate",
    }

    def _is_valid_action(self, action, obs):
        action_type = action.get("type", "")
        return action_type in self.WEBARENA_ACTIONS
