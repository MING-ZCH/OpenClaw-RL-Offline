"""
SORL: Stabilized Off-Policy GRPO for Long-Horizon LLM Agents.

Reference: Li et al., "Stabilizing Off-Policy Training for Long-Horizon LLM
Agent via Turn-Level Importance Sampling and Clipping-Triggered Normalization",
arXiv 2511.20718, Nov 2025 (v2 Feb 2026).

Key contributions (both in the optimization loop, zero architecture changes):

1. **Turn-level importance sampling**: In multi-turn agents, each transition
   is one "turn" (observation → action). Using raw per-token IS products
   causes exponential variance blow-up over long episodes.  SORL treats each
   transition as a turn and computes IS at that granularity — which in our
   framework already corresponds to one TransitionBatch row:
       ρ_turn = π_current(a_t | s_t) / π_old(a_t | s_t)
   This is already what OffPolicyGRPO computes; the SORL novelty is the
   *normalization policy* below.

2. **Clipping-triggered normalization (CTN)**: Rather than always normalizing
   advantages to unit variance (standard GRPO), SORL only normalizes when
   the gradient update is "turbulent" — defined as when the clipped IS
   fraction exceeds a threshold θ (default 0.2):
       If  clip_frac(ρ) > θ  →  normalize A_i via (A - μ) / (σ + ε)
       Otherwise             →  use raw A_i  (preserve magnitude signal)
   When the policy is close to the behavior policy (few clips), raw
   advantages carry calibrated magnitude information that normalization
   would discard.  When the policy diverges (many clips), normalization
   prevents gradient explosion.

This algorithm is implemented as a thin subclass of OffPolicyGRPO that
overrides only the training step to inject CTN.

Additional option:
  - use_turn_is_clip: if True, always applies turn-level clipping (the
    same as standard PPO at turn granularity, but no token-level product).
    In our framework this is the default behavior; the flag is kept for
    documentation parity with the paper.

Hyperparameters (in addition to OffPolicyGRPO):
  - clip_norm_threshold θ: fraction of clips that triggers normalization
    (default 0.2; paper range 0.1–0.3).
"""

from __future__ import annotations

import logging

import torch

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch
from .base import TrainMetrics
from .off_policy_grpo import OffPolicyGRPO

logger = logging.getLogger(__name__)


class SORLOffPolicyGRPO(OffPolicyGRPO):
    """
    SORL: OffPolicyGRPO with Clipping-Triggered Normalization (CTN).

    Inherits all behavior from OffPolicyGRPO and overrides `train_step`
    to apply SORL's adaptive advantage normalization.

    The only behavioral change:
    - After computing GRPO advantages and IS ratios, check clip_fraction.
    - Normalize advantages only if clip_fraction exceeds clip_norm_threshold.
    - Otherwise use raw (unnormalized) advantages to preserve magnitude signal.

    This prevents unnecessary variance reduction when the policy is close to
    the behavior policy, while still stabilizing training when divergence
    is detected.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        clip_ratio: float = 0.2,
        kl_coeff: float = 0.01,
        n_policy_updates: int = 4,
        clip_norm_threshold: float = 0.2,
        **kwargs,
    ):
        super().__init__(
            replay_buffer=replay_buffer,
            clip_ratio=clip_ratio,
            kl_coeff=kl_coeff,
            n_policy_updates=n_policy_updates,
            **kwargs,
        )
        self.clip_norm_threshold = clip_norm_threshold

    def _sorl_normalize(
        self,
        advantages: torch.Tensor,
        ratios: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Apply CTN: normalize advantages only if clip_fraction > threshold.

        Args:
            advantages: (B,) raw GRPO advantages (group-relative normalized
                by the base class, BUT in SORL we purposely skip base
                normalization and apply CTN here instead).
            ratios: (B,) IS ratios ρ = π_current / π_old for this batch.

        Returns:
            (advantages_out, clip_fraction)
        """
        with torch.no_grad():
            clip_low = ratios < (1.0 - self.clip_ratio)
            clip_high = ratios > (1.0 + self.clip_ratio)
            clip_frac = (clip_low | clip_high).float().mean().item()

        if clip_frac > self.clip_norm_threshold:
            # High divergence detected — normalize for stability
            adv_out = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            # Policy close to behavior — preserve raw magnitude
            adv_out = advantages

        return adv_out, clip_frac

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """SORL training step with Clipping-Triggered Normalization (CTN).

        Overrides OffPolicyGRPO.train_step to:
        1. Skip the always-on advantage normalization in the base class.
        2. Apply CTN after computing IS ratios on the first sub-step.
        3. Reuse the same (possibly normalized) advantages for all
           n_policy_updates sub-steps.
        """
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32).to(self.device)
        # Raw group-relative advantages (not normalized yet — SORL decides)
        raw_advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # ── Pre-compute IS ratios on first pass (for CTN decision) ────────
        s0, a0 = self._encode_batch(batch)
        with torch.no_grad():
            log_probs_ref0 = self.ref_policy(s0, a0)
        log_probs_cur0 = self.policy(s0.detach(), a0.detach())
        behavior_log_probs0, _ = self._get_behavior_log_probs(batch, log_probs_ref0)
        ratios0 = torch.exp(log_probs_cur0.detach() - behavior_log_probs0.detach())

        # ── SORL CTN: decide whether to normalize advantages ──────────────
        advantages, clip_frac = self._sorl_normalize(raw_advantages, ratios0)

        # ── n_policy_updates gradient steps with frozen advantages ────────
        total_surr_loss = 0.0
        total_kl_loss = 0.0
        total_ratio = 0.0
        total_behavior_coverage = 0.0

        for _ in range(self.n_policy_updates):
            s, a = self._encode_batch(batch)
            log_probs_current = self.policy(s, a)

            with torch.no_grad():
                log_probs_ref = self.ref_policy(s, a)

            behavior_log_probs, coverage = self._get_behavior_log_probs(
                batch, log_probs_ref
            )
            ratio = torch.exp(log_probs_current - behavior_log_probs.detach())

            surr1 = ratio * advantages
            clipped_ratio = torch.clamp(
                ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
            )
            surr2 = clipped_ratio * advantages
            surr_loss = -torch.min(surr1, surr2).mean()

            kl_penalty = (log_probs_ref.detach() - log_probs_current).mean()
            loss = surr_loss + self.kl_coeff * kl_penalty

            self.optimizer.zero_grad()
            loss.backward()
            from torch.nn.utils.clip_grad import clip_grad_norm_
            clip_grad_norm_(
                list(self.state_encoder.parameters())
                + list(self.action_encoder.parameters())
                + list(self.policy.parameters()),
                self.max_grad_norm or 1.0,
            )
            self.optimizer.step()

            total_surr_loss += surr_loss.item()
            total_kl_loss += kl_penalty.item()
            total_ratio += ratio.mean().item()
            total_behavior_coverage += coverage

        n = self.n_policy_updates
        return TrainMetrics(
            loss=total_surr_loss / n + self.kl_coeff * total_kl_loss / n,
            extra={
                "surrogate_loss": total_surr_loss / n,
                "kl_penalty": total_kl_loss / n,
                "mean_advantage": advantages.mean().item(),
                "mean_ratio": total_ratio / n,
                "clip_fraction": clip_frac,
                "ctn_normalized": float(clip_frac > self.clip_norm_threshold),
                "behavior_log_prob_coverage": total_behavior_coverage / n,
            },
        )
