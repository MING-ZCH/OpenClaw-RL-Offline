"""
ARPO: Adaptive Replay Policy Optimization (arXiv 2505.16282).

Extends OffPolicyGRPO with a per-task success replay buffer that is injected
when a training group is all-fail, guaranteeing non-zero reward variance and
non-vanishing GRPO advantages.

Key differences vs standard OffPolicyGRPO / SORL:
  1. ``ARPOSuccessBuffer`` -- FIFO queue of positive-outcome transitions;
     updated every train_step with transitions whose outcome_reward > threshold.
  2. All-fail detection -- if std(outcome_rewards) < 0.05 AND
     mean(outcome_rewards) < 0.2, inject one past success into slot 0.
  3. DAPO asymmetric clipping -- separate lower (clip_ratio_low) and upper
     (clip_ratio_high) PPO clip coefficients; default 0.2 / 0.3.
  4. No KL penalty -- kl_coeff forced to 0, consistent with ARPO paper.

References:
    ARPO: "Adaptive Replay for Group Relative Policy Optimization"
    arXiv 2505.16282, dvlab-research/ARPO (GitHub).
"""

from __future__ import annotations

import logging
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch
from .base import TrainMetrics
from .off_policy_grpo import OffPolicyGRPO

logger = logging.getLogger(__name__)


class ARPOSuccessBuffer:
    """
    FIFO buffer storing transitions with positive outcome reward.

    Implements the replay mechanism described in arXiv 2505.16282:
    ``update_replay_buffer`` stores only successes; ``sample`` returns a
    random element if the buffer is non-empty.

    Args:
        max_size: Maximum number of stored transitions (FIFO eviction).
        success_threshold: Minimum outcome_reward to qualify as a success.
    """

    def __init__(self, max_size: int = 8, success_threshold: float = 0.1):
        self.max_size = max_size
        self.success_threshold = success_threshold
        self._store: List[Tuple[str, str, float, float]] = []  # (obs, act, r, outcome_r)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, batch: TransitionBatch) -> int:
        """
        Add positive-outcome transitions from *batch* to the buffer.

        Returns the number of newly stored transitions.
        """
        added = 0
        for i, outcome_r in enumerate(batch.outcome_rewards):
            if float(outcome_r) > self.success_threshold:
                entry = (
                    batch.observation_contexts[i],
                    batch.actions[i],
                    float(batch.rewards[i]),
                    float(outcome_r),
                )
                self._store.append(entry)
                if len(self._store) > self.max_size:
                    self._store.pop(0)  # FIFO eviction
                added += 1
        return added

    def sample(self) -> Optional[Tuple[str, str, float, float]]:
        """Return a random stored success or None if buffer is empty."""
        if not self._store:
            return None
        return random.choice(self._store)

    def __len__(self) -> int:
        return len(self._store)


class ARPO(OffPolicyGRPO):
    """
    Adaptive Replay Policy Optimization.

    Inherits the full policy network, encoders, and optimizer from
    OffPolicyGRPO.  Overrides ``train_step`` to:

    1. Update the success buffer with current batch transitions.
    2. Check the all-fail condition (std < threshold AND mean < threshold).
    3. If all-fail and buffer non-empty, splice a past success into slot 0.
    4. Recompute GRPO advantages on the (possibly augmented) batch.
    5. Run n_policy_updates DAPO-style update steps (asymmetric clip, no KL).

    Args:
        replay_buffer: Main offline replay buffer (standard offline RL data).
        arpo_buffer_size: Capacity of the per-task success buffer (default 8).
        clip_ratio_low: PPO lower clip epsilon, i.e. ratio >= 1 - clip_ratio_low
            (default 0.2, matching ARPO paper).
        clip_ratio_high: PPO upper clip epsilon, i.e. ratio <= 1 + clip_ratio_high
            (default 0.3, DAPO asymmetric variant).
        all_fail_std_threshold: Reward std below which a group is labelled
            all-fail (default 0.05).
        all_fail_mean_threshold: Reward mean below which a group is labelled
            all-fail (default 0.2).
        success_threshold: Minimum outcome_reward for a transition to be stored
            as a success (default 0.1).
        **kwargs: Forwarded to OffPolicyGRPO (state_dim, action_dim, lr, etc.).
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        arpo_buffer_size: int = 8,
        clip_ratio_low: float = 0.2,
        clip_ratio_high: float = 0.3,
        all_fail_std_threshold: float = 0.05,
        all_fail_mean_threshold: float = 0.2,
        success_threshold: float = 0.1,
        **kwargs,
    ):
        # ARPO removes KL penalty entirely.
        kwargs["kl_coeff"] = 0.0
        # Pass lower clip as base clip_ratio.
        kwargs["clip_ratio"] = clip_ratio_low
        super().__init__(replay_buffer=replay_buffer, **kwargs)

        self.clip_ratio_low = clip_ratio_low
        self.clip_ratio_high = clip_ratio_high
        self.all_fail_std_threshold = all_fail_std_threshold
        self.all_fail_mean_threshold = all_fail_mean_threshold
        self._arpo_success_buf = ARPOSuccessBuffer(
            max_size=arpo_buffer_size,
            success_threshold=success_threshold,
        )

    # ------------------------------------------------------------------
    # Replay injection helpers
    # ------------------------------------------------------------------

    def _is_all_fail_group(self, outcome_rewards: np.ndarray) -> bool:
        """Return True when the group qualifies for replay injection."""
        r_std = float(np.std(outcome_rewards))
        r_mean = float(np.mean(outcome_rewards))
        return (
            r_std < self.all_fail_std_threshold
            and r_mean < self.all_fail_mean_threshold
        )

    def _inject_replay(
        self,
        batch: TransitionBatch,
        outcome_rewards: np.ndarray,
    ) -> Tuple[TransitionBatch, int]:
        """
        Replace slot 0 of *batch* with a sampled success when all-fail.

        Returns the (possibly modified) batch and the injection count (0 or 1).
        """
        if not self._is_all_fail_group(outcome_rewards):
            return batch, 0
        sample = self._arpo_success_buf.sample()
        if sample is None:
            return batch, 0

        pos_obs, pos_act, pos_r, pos_outcome = sample

        # Replace slot 0; keep remaining slots unchanged.
        new_obs = [pos_obs] + list(batch.observation_contexts[1:])
        new_acts = [pos_act] + list(batch.actions[1:])
        new_rewards = np.array([pos_r] + list(batch.rewards[1:]), dtype=np.float32)
        new_outcome_rewards = np.array(
            [pos_outcome] + list(outcome_rewards[1:]), dtype=np.float32
        )

        augmented = TransitionBatch(
            instructions=list(batch.instructions),
            observation_contexts=new_obs,
            actions=new_acts,
            rewards=new_rewards,
            next_observation_contexts=list(batch.next_observation_contexts),
            dones=batch.dones.copy(),
            outcome_rewards=new_outcome_rewards,
            behavior_log_probs=batch.behavior_log_probs.copy(),
        )
        logger.debug("ARPO: injected 1 replay success into all-fail group")
        return augmented, 1

    # ------------------------------------------------------------------
    # DAPO clipped surrogate loss (asymmetric clip, no KL)
    # ------------------------------------------------------------------

    def _dapo_loss(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """
        DAPO-style dual-clip PPO loss with asymmetric epsilon.

        L = -mean( min(r*A, clip(r, 1-eps_low, 1+eps_high)*A) )
        """
        surr_unclipped = ratio * advantages
        surr_clipped = (
            torch.clamp(ratio, 1.0 - self.clip_ratio_low, 1.0 + self.clip_ratio_high)
            * advantages
        )
        return -torch.min(surr_unclipped, surr_clipped).mean()

    # ------------------------------------------------------------------
    # Main training step
    # ------------------------------------------------------------------

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """
        Perform one round of ARPO updates.

        Steps:
            1. Collect outcome_rewards for all-fail detection.
            2. Update success buffer with current batch.
            3. Inject a past success if the group is all-fail.
            4. Compute GRPO group-normalised advantages.
            5. n_policy_updates DAPO gradient steps (no KL, asymmetric clip).
        """
        # --- Outcome rewards (trajectory-level binary signal) ----------
        outcome_rewards = np.asarray(batch.outcome_rewards, dtype=np.float32)

        # --- 1. Update success buffer ---------------------------------
        self._arpo_success_buf.update(batch)

        # --- 2. Replay injection --------------------------------------
        batch, injected = self._inject_replay(batch, outcome_rewards)
        if injected > 0:
            # Refresh after splice.
            outcome_rewards = np.asarray(batch.outcome_rewards, dtype=np.float32)

        # --- 3. GRPO advantages on (augmented) batch ------------------
        rewards_t = torch.as_tensor(outcome_rewards, dtype=torch.float32).to(self.device)
        advantages = self._compute_grpo_advantages(rewards_t)

        # --- 4. n_policy_updates DAPO steps ---------------------------
        total_loss = 0.0
        total_ratio = 0.0

        for _ in range(self.n_policy_updates):
            s, a = self._encode_batch(batch)

            log_probs_current = self.policy(s, a)
            with torch.no_grad():
                log_probs_ref = self.ref_policy(s, a)

            # Behavior log-probs (from replay data or fallback to ref).
            behavior_log_probs, _ = self._get_behavior_log_probs(batch, log_probs_ref)
            ratio = torch.exp(log_probs_current - behavior_log_probs.detach())

            loss = self._dapo_loss(ratio, advantages)

            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(
                list(self.state_encoder.parameters())
                + list(self.action_encoder.parameters())
                + list(self.policy.parameters()),
                self.max_grad_norm or 1.0,
            )
            self.optimizer.step()

            total_loss += loss.item()
            total_ratio += ratio.mean().item()

        avg_loss = total_loss / self.n_policy_updates
        avg_ratio = total_ratio / self.n_policy_updates

        return TrainMetrics(
            loss=avg_loss,
            extra={
                "mean_advantage": advantages.mean().item(),
                "mean_ratio": avg_ratio,
                "replay_injected": float(injected),
                "arpo_buffer_size": float(len(self._arpo_success_buf)),
            },
        )

    def save(self, path: str) -> None:
        """Save model parameters to file."""
        torch.save(
            {
                "state_encoder": self.state_encoder.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "policy": self.policy.state_dict(),
                "ref_policy": self.ref_policy.state_dict(),
            },
            path,
        )
        logger.info("Saved ARPO to %s", path)

    def load(self, path: str) -> None:
        """Load model parameters from file."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.policy.load_state_dict(ckpt["policy"])
        self.ref_policy.load_state_dict(ckpt["ref_policy"])
        logger.info("Loaded ARPO from %s", path)
