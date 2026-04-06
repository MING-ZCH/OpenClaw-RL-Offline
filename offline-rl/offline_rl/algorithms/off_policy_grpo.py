"""
Off-Policy GRPO (Group Relative Policy Optimization) for offline RL.

Extends the GRPO algorithm used by OpenClaw-RL's gui-rl training pipeline
to the off-policy setting by:
1. Using pre-collected trajectories from a ReplayBuffer
2. Importance sampling to correct for distribution shift
3. Clipped importance weights for stability (PPO-style)

Reference: gui-rl/generate_with_gui.py uses online GRPO with binary rewards.
This offline variant enables training without live environment interaction.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch
from .base import BaseOfflineAlgorithm, StateEncoder, ActionEncoder, TrainMetrics

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """
    Policy network that maps (state, action) to log-probability.

    For LLM agents, this corresponds to the language model's log-prob
    of generating the action text given the state text. This lightweight
    MLP version is for testing; production replaces with LLM forward pass.
    """

    def __init__(self, state_dim: int = 256, action_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Returns log-probability of (state, action)."""
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)  # (batch,)


class OffPolicyGRPO(BaseOfflineAlgorithm):
    """
    Off-Policy Group Relative Policy Optimization.

    Algorithm outline:
        1. Sample a batch of transitions from the replay buffer
        2. Compute GRPO advantages via group-relative normalization:
           A_i = (R_i - mean(R)) / (std(R) + eps)
        3. Compute importance sampling ratio:
           rho = pi_current(a|s) / pi_old(a|s)
        4. Clipped surrogate objective (PPO-style):
           L = -min(rho * A, clip(rho, 1-eps, 1+eps) * A)
        5. KL penalty against reference (frozen) policy:
           L_total = L_surr + beta * KL(pi_current || pi_ref)

    Parameters:
        replay_buffer: Buffer with pre-collected transitions
        clip_ratio: PPO-style clipping parameter (default 0.2)
        kl_coeff: KL penalty coefficient (default 0.01)
        n_policy_updates: Number of gradient steps per train_step call
        **kwargs: Forwarded to BaseOfflineAlgorithm
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        clip_ratio: float = 0.2,
        kl_coeff: float = 0.01,
        n_policy_updates: int = 4,
        state_dim: int = 256,
        action_dim: int = 256,
        hidden_dim: int = 256,
        lr: float = 1e-4,
        gamma: float = 0.99,
        device: str = "cpu",
        max_token_len: int = 128,
    ):
        super().__init__(
            replay_buffer=replay_buffer,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            device=device,
            max_token_len=max_token_len,
        )
        self.clip_ratio = clip_ratio
        self.kl_coeff = kl_coeff
        self.n_policy_updates = n_policy_updates

        # Encoders
        self.state_encoder = StateEncoder(self._vocab_size, 256, state_dim).to(self.device)
        self.action_encoder = ActionEncoder(self._vocab_size, 256, action_dim).to(self.device)

        # Current policy network
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # Reference policy network (frozen, for KL penalty)
        self.ref_policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.ref_policy.load_state_dict(self.policy.state_dict())
        for p in self.ref_policy.parameters():
            p.requires_grad = False

        # Optimizer
        params = (
            list(self.state_encoder.parameters())
            + list(self.action_encoder.parameters())
            + list(self.policy.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def _encode_batch(self, batch: TransitionBatch):
        """Encode states and actions only (GRPO doesn't need next_states)."""
        state_ids = self._tokenize(batch.observation_contexts)
        action_ids = self._tokenize(batch.actions)
        s = self.state_encoder(state_ids)
        a = self.action_encoder(action_ids)
        return s, a

    def _compute_grpo_advantages(
        self,
        rewards: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Group-relative advantage normalization (GRPO).

        When group_ids is provided, normalizes within each group:
            A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
        Otherwise falls back to batch-level normalization.
        """
        if group_ids is None:
            mean_r = rewards.mean()
            std_r = rewards.std() + 1e-8
            return (rewards - mean_r) / std_r

        # Intra-group normalization (correct GRPO)
        advantages = torch.zeros_like(rewards)
        unique_groups = group_ids.unique()
        for gid in unique_groups:
            mask = group_ids == gid
            group_rewards = rewards[mask]
            mean_r = group_rewards.mean()
            std_r = group_rewards.std() + 1e-8
            advantages[mask] = (group_rewards - mean_r) / std_r
        return advantages

    def _get_behavior_log_probs(
        self,
        batch: TransitionBatch,
        log_probs_ref: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        behavior_log_probs = torch.as_tensor(
            batch.behavior_log_probs,
            dtype=torch.float32,
            device=self.device,
        )
        valid_mask = torch.isfinite(behavior_log_probs)
        coverage = valid_mask.float().mean().item()

        if coverage <= 0.0:
            return log_probs_ref.detach(), 0.0
        if coverage >= 1.0:
            return behavior_log_probs, 1.0

        fallback = torch.where(valid_mask, behavior_log_probs, log_probs_ref.detach())
        return fallback, coverage

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """
        Perform one round of off-policy GRPO updates.

        Multiple gradient steps with the same batch (n_policy_updates).
        """
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32).to(self.device)
        group_ids = None
        if batch.group_ids is not None:
            group_ids = torch.as_tensor(batch.group_ids, dtype=torch.long).to(self.device)
        advantages = self._compute_grpo_advantages(rewards, group_ids=group_ids)

        total_surr_loss = 0.0
        total_kl_loss = 0.0
        total_ratio = 0.0
        total_behavior_coverage = 0.0

        for _ in range(self.n_policy_updates):
            s, a = self._encode_batch(batch)

            # Current policy log-probs
            log_probs_current = self.policy(s, a)

            # Reference policy log-probs (frozen)
            with torch.no_grad():
                log_probs_ref = self.ref_policy(s, a)

            # Prefer replayed behavior-policy log-probs when available.
            # Fall back to reference-policy log-probs for legacy datasets.
            behavior_log_probs, coverage = self._get_behavior_log_probs(batch, log_probs_ref)
            ratio = torch.exp(log_probs_current - behavior_log_probs.detach())

            # Clipped surrogate objective
            surr1 = ratio * advantages
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            surr2 = clipped_ratio * advantages
            surr_loss = -torch.min(surr1, surr2).mean()

            # KL penalty
            kl_penalty = (log_probs_ref.detach() - log_probs_current).mean()

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

            total_surr_loss += surr_loss.item()
            total_kl_loss += kl_penalty.item()
            total_ratio += ratio.mean().item()
            total_behavior_coverage += coverage

        avg_surr = total_surr_loss / self.n_policy_updates
        avg_kl = total_kl_loss / self.n_policy_updates
        avg_ratio = total_ratio / self.n_policy_updates
        avg_behavior_coverage = total_behavior_coverage / self.n_policy_updates

        return TrainMetrics(
            loss=avg_surr + self.kl_coeff * avg_kl,
            extra={
                "surrogate_loss": avg_surr,
                "kl_penalty": avg_kl,
                "mean_advantage": advantages.mean().item(),
                "mean_ratio": avg_ratio,
                "behavior_log_prob_coverage": avg_behavior_coverage,
                "behavior_fallback_fraction": 1.0 - avg_behavior_coverage,
            },
        )

    def get_action_values(
        self, states: list[str], actions: list[str]
    ) -> torch.Tensor:
        """Compute policy log-probs for state-action pairs."""
        state_ids = self._tokenize(states)
        action_ids = self._tokenize(actions)
        with torch.no_grad():
            s = self.state_encoder(state_ids)
            a = self.action_encoder(action_ids)
            return self.policy(s, a)

    def train(
        self,
        num_steps: int = 1000,
        batch_size: int = 64,
        n_groups: int = 8,
        log_interval: int = 100,
    ) -> list[TrainMetrics]:
        """Run offline GRPO training with group-relative sampling.

        Uses ``sample_transition_groups`` to ensure advantages are computed
        within instruction groups.  Falls back to ``sample_transitions`` when
        group sampling is unavailable (e.g. single-instruction datasets).

        Args:
            num_steps: Number of training iterations.
            batch_size: Batch size for fallback uniform sampling.
            n_groups: Number of instruction groups per step.
            log_interval: Steps between log messages.
        """
        all_metrics: list[TrainMetrics] = []
        for step in range(num_steps):
            try:
                batch = self.replay_buffer.sample_transition_groups(
                    n_groups=n_groups, min_group_size=2,
                )
            except (ValueError, AttributeError):
                batch = self.replay_buffer.sample_transitions(batch_size)
            metrics = self.train_step(batch)
            all_metrics.append(metrics)

            if (step + 1) % log_interval == 0:
                avg_loss = sum(m.loss for m in all_metrics[-log_interval:]) / log_interval
                logger.info("Step %d/%d: avg_loss=%.4f", step + 1, num_steps, avg_loss)

        return all_metrics

    def update_reference_policy(self):
        """Copy current policy to reference policy (periodic refresh)."""
        self.ref_policy.load_state_dict(self.policy.state_dict())
        for p in self.ref_policy.parameters():
            p.requires_grad = False
        logger.info("Updated reference policy from current policy")

    def save(self, path: str) -> None:
        """Save model parameters to file."""
        torch.save({
            "state_encoder": self.state_encoder.state_dict(),
            "action_encoder": self.action_encoder.state_dict(),
            "policy": self.policy.state_dict(),
            "ref_policy": self.ref_policy.state_dict(),
        }, path)
        logger.info("Saved OffPolicyGRPO to %s", path)

    def load(self, path: str) -> None:
        """Load model parameters from file."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.policy.load_state_dict(ckpt["policy"])
        self.ref_policy.load_state_dict(ckpt["ref_policy"])
        logger.info("Loaded OffPolicyGRPO from %s", path)
