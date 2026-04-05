"""
WebRL: Self-Evolving Online Curriculum Reinforcement Learning for LLM Web Agents.

Adapted to the offline/off-policy setting for use with pre-collected trajectory
datasets (no live environment interaction required).

Core algorithmic contributions implemented here:

  1. Outcome-Supervised Reward Model (ORM): a lightweight binary classifier
     ``OutcomeSupervisedRewardModel`` trained on stored (state, action, outcome)
     triples that predicts step-level task-success probability.  Trained jointly
     with the policy via binary cross-entropy on the *trajectory-level* outcome
     label (1 = success, 0 = failure), providing dense per-step supervision.

  2. ORM-Augmented Reward: the dense ORM signal is mixed with the sparse
     trajectory-level outcome reward:
         r_aug(s_t, a_t) = r_outcome + alpha_orm * sigmoid(ORM(s_t, a_t))
     This converts sparse binary rewards into informative per-step signals.

  3. Off-Policy GRPO with Augmented Rewards: standard PPO-clip gradient with
     group-normalised advantage weighting applied to r_aug, using the
     OffPolicyGRPO machinery from the parent class.

  4. Curriculum Difficulty Tracker: classifies each training batch as
     ``easy`` (all-success), ``medium``, or ``hard`` (all-fail) by the mean
     outcome reward, and reports a ``curriculum_difficulty`` scalar metric
     (0.0/0.5/1.0) for downstream curriculum task sampling.

Offline-mode notes vs. original WebRL (Qi et al., ICLR 2025):
  - Original WebRL uses *online* rollouts for curriculum task mutation; this
    implementation works on a *fixed* offline dataset.
  - ORM is trained in-loop on the same batches used for GRPO policy updates.
  - Curriculum selection and task mutation are simplified to a monitoring layer.

References:
    Qi et al., "WebRL: Training LLM Web Agents via Self-Evolving Online
    Curriculum Reinforcement Learning", ICLR 2025 (arXiv 2411.02337).
    Code: https://github.com/THUDM/WebRL
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch
from .base import TrainMetrics
from .off_policy_grpo import OffPolicyGRPO

import torch.nn as nn

logger = logging.getLogger(__name__)


class OutcomeSupervisedRewardModel(nn.Module):
    """
    ORM: lightweight binary classifier for step-level task-success prediction.

    Accepts a (state_emb, action_emb) pair and returns a raw logit whose
    sigmoid is P(task_success | s_t, a_t).  Trained with binary cross-entropy
    using the trajectory-level outcome label (success/failure) as supervision.

    In the WebRL paper the ORM is a full LM-based preference model; here we
    use a shallow MLP over the same token-hash embeddings used by the policy,
    which is sufficient for functional offline training and CPU unit testing.

    Args:
        state_dim: Dimension of state embedding (matches policy's state_dim).
        action_dim: Dimension of action embedding (matches policy's action_dim).
        hidden_dim: Hidden layer size.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state:  (B, state_dim) float tensor.
            action: (B, action_dim) float tensor.
        Returns:
            logits: (B,) raw logit.  Apply sigmoid for probabilities.
        """
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class WebRL(OffPolicyGRPO):
    """
    WebRL: Off-policy GRPO augmented with an Outcome-Supervised Reward Model.

    Extends :class:`OffPolicyGRPO` with:

    * An ``OutcomeSupervisedRewardModel`` (ORM) trained jointly with the
      policy using BCE on binary outcome labels.
    * ORM-augmented reward: ``r_aug = r_outcome + alpha_orm * σ(ORM(s, a))``.
    * A curriculum difficulty classifier reported as a training metric.

    Training procedure (per ``train_step``):
      1. Encode (s, a) embeddings.
      2. Update ORM with BCE loss (detached embeddings so ORM grad does not
         leak into the shared encoders).
      3. Compute ORM probability for each transition (``torch.no_grad``).
      4. Build augmented reward: ``r_aug = r_outcome + alpha_orm * σ(ORM)``.
      5. GRPO advantage normalization on ``r_aug``.
      6. ``n_policy_updates`` PPO-clip gradient steps on policy + encoders.

    Args:
        replay_buffer: Offline dataset of stored trajectories.
        alpha_orm: Weight of the ORM dense reward in the augmented reward.
            ``r_aug = r_outcome + alpha_orm * sigmoid(ORM(s, a))``.
        orm_lr: Learning-rate for the ORM optimizer.  Defaults to the same
            value as the policy learning rate.
        curriculum_easy_thresh: Mean outcome-reward above which a batch is
            classified as "easy" (near-perfect success, low gradient signal).
        curriculum_hard_thresh: Mean outcome-reward below which a batch is
            classified as "hard" (all-fail, no positive signal).
        **kwargs: Forwarded verbatim to :class:`OffPolicyGRPO`.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        alpha_orm: float = 0.5,
        orm_lr: Optional[float] = None,
        curriculum_easy_thresh: float = 0.8,
        curriculum_hard_thresh: float = 0.2,
        **kwargs,
    ):
        super().__init__(replay_buffer=replay_buffer, **kwargs)

        self.alpha_orm = alpha_orm
        self.curriculum_easy_thresh = curriculum_easy_thresh
        self.curriculum_hard_thresh = curriculum_hard_thresh

        # ORM — uses same embedding dimensions as the policy encoders
        self.orm = OutcomeSupervisedRewardModel(
            self.state_dim, self.action_dim, self.hidden_dim
        ).to(self.device)

        effective_orm_lr = orm_lr if orm_lr is not None else self.lr
        self.orm_optimizer = torch.optim.Adam(
            self.orm.parameters(), lr=effective_orm_lr
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_difficulty(self, outcome_array: np.ndarray) -> float:
        """
        Classify batch curriculum difficulty as a float scalar.

        Returns:
            0.0 for easy (all-success), 0.5 for medium, 1.0 for hard.
        """
        mean_r = float(np.mean(outcome_array))
        if mean_r >= self.curriculum_easy_thresh:
            return 0.0
        if mean_r <= self.curriculum_hard_thresh:
            return 1.0
        return 0.5

    def _update_orm(
        self,
        s_detached: torch.Tensor,
        a_detached: torch.Tensor,
        outcome_array: np.ndarray,
    ) -> float:
        """
        One BCE gradient step on the ORM.

        Uses *detached* state/action embeddings so the ORM loss does not
        propagate into the shared state/action encoders (which are separately
        updated by the policy GRPO loss).

        Args:
            s_detached: (B, state_dim) tensor, no gradient.
            a_detached: (B, action_dim) tensor, no gradient.
            outcome_array: (B,) numpy array of outcome rewards.

        Returns:
            orm_loss: scalar float for logging.
        """
        # Positive label = trajectory succeeded
        labels = torch.as_tensor(
            (outcome_array > 0.5).astype(np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        logits = self.orm(s_detached, a_detached)
        orm_loss = F.binary_cross_entropy_with_logits(logits, labels)

        self.orm_optimizer.zero_grad()
        orm_loss.backward()
        clip_grad_norm_(self.orm.parameters(), self.max_grad_norm or 1.0)
        self.orm_optimizer.step()

        return orm_loss.item()

    # ------------------------------------------------------------------
    # Overridden train_step
    # ------------------------------------------------------------------

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """
        WebRL training step: ORM update → reward augmentation → GRPO.

        Returns :class:`TrainMetrics` with extra keys:
          - ``orm_loss``: BCE loss of the ORM on this batch.
          - ``orm_reward_mean``: mean ORM-predicted success probability.
          - ``augmented_reward_mean``: mean of ``r_aug`` used for GRPO.
          - ``curriculum_difficulty``: 0.0 / 0.5 / 1.0.
        """
        outcome_array = np.asarray(batch.outcome_rewards, dtype=np.float32)

        # --- Step 1: encode batch (no grad needed at this stage) ----------
        with torch.no_grad():
            s = self.state_encoder(self._tokenize(batch.observation_contexts))
            a = self.action_encoder(self._tokenize(batch.actions))

        # --- Step 2: update ORM (detached embeddings) ---------------------
        orm_loss_val = self._update_orm(s, a, outcome_array)

        # --- Step 3: ORM-augmented reward ---------------------------------
        with torch.no_grad():
            orm_prob = torch.sigmoid(self.orm(s, a))  # (B,) P(success)
        r_outcome = torch.as_tensor(
            outcome_array, dtype=torch.float32, device=self.device
        )
        r_aug = r_outcome + self.alpha_orm * orm_prob  # (B,)

        # --- Step 4: GRPO advantage normalization on augmented reward ------
        advantages = self._compute_grpo_advantages(r_aug)

        # --- Step 5: n_policy_updates PPO-clip policy gradient steps -------
        total_surr = 0.0
        total_kl = 0.0
        total_ratio = 0.0
        total_cov = 0.0

        for _ in range(self.n_policy_updates):
            # Re-encode with grad for policy update
            s2 = self.state_encoder(self._tokenize(batch.observation_contexts))
            a2 = self.action_encoder(self._tokenize(batch.actions))

            log_probs_cur = self.policy(s2, a2)
            with torch.no_grad():
                log_probs_ref = self.ref_policy(s2, a2)

            beh_lp, cov = self._get_behavior_log_probs(batch, log_probs_ref)
            ratio = torch.exp(log_probs_cur - beh_lp.detach())

            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                * advantages
            )
            surr_loss = -torch.min(surr1, surr2).mean()
            kl_penalty = (log_probs_ref.detach() - log_probs_cur).mean()
            loss = surr_loss + self.kl_coeff * kl_penalty

            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(
                list(self.state_encoder.parameters())
                + list(self.action_encoder.parameters())
                + list(self.policy.parameters()),
                self.max_grad_norm or 1.0,
            )
            self.optimizer.step()

            total_surr += surr_loss.item()
            total_kl += kl_penalty.item()
            total_ratio += ratio.mean().item()
            total_cov += cov

        n = self.n_policy_updates
        avg_surr = total_surr / n
        avg_kl = total_kl / n
        avg_ratio = total_ratio / n

        return TrainMetrics(
            loss=avg_surr + self.kl_coeff * avg_kl,
            extra={
                "surrogate_loss": avg_surr,
                "kl_penalty": avg_kl,
                "mean_advantage": advantages.mean().item(),
                "mean_ratio": avg_ratio,
                "behavior_log_prob_coverage": total_cov / n,
                "orm_loss": orm_loss_val,
                "orm_reward_mean": orm_prob.mean().item(),
                "augmented_reward_mean": r_aug.mean().item(),
                "curriculum_difficulty": self._classify_difficulty(outcome_array),
            },
        )

    # ------------------------------------------------------------------
    # Action scoring (uses ORM success probability as a value estimate)
    # ------------------------------------------------------------------

    def get_action_values(
        self, states: List[str], actions: List[str]
    ) -> torch.Tensor:
        """Return ORM P(success) as a scalar value estimate per action."""
        with torch.no_grad():
            s = self.state_encoder(self._tokenize(states))
            a = self.action_encoder(self._tokenize(actions))
            return torch.sigmoid(self.orm(s, a))  # (B,)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save policy, reference policy, and ORM parameters."""
        torch.save(
            {
                "state_encoder": self.state_encoder.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "policy": self.policy.state_dict(),
                "ref_policy": self.ref_policy.state_dict(),
                "orm": self.orm.state_dict(),
            },
            path,
        )
        logger.info("Saved WebRL to %s", path)

    def load(self, path: str) -> None:
        """Load policy, reference policy, and ORM parameters."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.policy.load_state_dict(ckpt["policy"])
        self.ref_policy.load_state_dict(ckpt["ref_policy"])
        self.orm.load_state_dict(ckpt["orm"])
        logger.info("Loaded WebRL from %s", path)
