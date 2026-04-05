"""
RW-FT: Reward-Weighted Fine-Tuning for offline RL.

Reference: Mukherjee et al., "Reward-Weighted Fine-Tuning: Offline RL as
Weighted Supervised Learning for LLM Agents", NeurIPS 2025 (arXiv:2506.06964)

Key idea: The simplest offline RL algorithm for LLM agents — just weight BC
loss by (normalized) trajectory-level rewards. No critic, no Q-network, no
value function required:
    L = -E_τ [ w(R_τ) · log π(a_t | s_t) ]
    w(R_τ) = softmax(R_τ / beta)  (normalized across batch trajectories)

In our continuous-action embedding setting, log π(a|s) is replaced by
-‖μ(s) - a_data‖² (MSE in action embedding space), so:
    L = E_τ [ w(R_τ) · ‖μ(s_t) - a_t‖² ]

This recovers REINFORCE with baseline (self-normalized IS) in the batch-
offline limit. Outperforms DPO and supervised fine-tuning on multi-turn
agent benchmarks (WebShop, multi-turn QA) at minimal implementation cost.

The trajectory-level reward `outcome_reward` is read directly from the
TransitionBatch (stored per-transition from the trajectory outcome).
No need for trajectory-level batching — each transition already carries
its parent trajectory's outcome reward.

Hyperparameters:
  - beta: temperature for softmax reward normalization.
          Lower β → more selective (only high-reward trajectories matter).
          Higher β → closer to uniform BC.
  - reward_norm: 'softmax' (default, normalized to sum=1) or 'exp'
                 (unnormalized, can diverge without clip).
  - reward_clip: clip weight range to [1/reward_clip, reward_clip] for stability.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch
from .base import BaseOfflineAlgorithm, StateEncoder, ActionEncoder, TrainMetrics
from .td3bc import DeterministicActor  # Re-use the same 3-layer MLP actor

logger = logging.getLogger(__name__)


class RWFineTuning(BaseOfflineAlgorithm):
    """
    RW-FT: Reward-Weighted Fine-Tuning for offline LLM agent training.

    Algorithm:
        1. For each transition (s, a, R_τ) in the batch, compute a weight
           w_i = softmax(R_τ^{(i)} / beta) · B   (where B = batch size)
           This gives high-reward trajectories more influence.
        2. Update the deterministic actor via weighted MSE:
           L = mean( w_i · ‖μ_θ(s_i) - a_i‖² )
        3. Update the state encoder jointly (gradient flows through μ_θ(s)).

    No Q-network, no value function, no replay buffer beyond the dataset.
    This is the simplest possible offline RL algorithm: reward-weighted BC.

    Key hyperparameters:
    - beta: softmax temperature (default 1.0).
    - reward_norm: 'softmax' (default, numerically stable) or 'exp'.
    - reward_clip: max weight multiplier for numerical stability (default 10.0).
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        state_dim: int = 256,
        action_dim: int = 256,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        beta: float = 1.0,
        reward_norm: str = "softmax",
        reward_clip: float = 10.0,
        device: str = "cpu",
        **kwargs,
    ):
        if reward_norm not in ("softmax", "exp"):
            raise ValueError(
                "reward_norm must be 'softmax' or 'exp', got: %s" % reward_norm
            )
        super().__init__(
            replay_buffer=replay_buffer,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            device=device,
            **kwargs,
        )
        self.beta = beta
        self.reward_norm = reward_norm
        self.reward_clip = reward_clip
        self._total_steps = 0

        # State encoder (no action encoder needed — BC sees action as target)
        self.state_encoder = StateEncoder(
            vocab_size=self._vocab_size, hidden_dim=state_dim
        ).to(self.device)
        self.action_encoder = ActionEncoder(
            vocab_size=self._vocab_size, hidden_dim=action_dim
        ).to(self.device)

        # Deterministic actor: state → action embedding prediction
        self.actor = DeterministicActor(state_dim, action_dim, hidden_dim).to(
            self.device
        )

        # Single optimizer for both encoder and actor
        self.optimizer = torch.optim.Adam(
            list(self.state_encoder.parameters())
            + list(self.action_encoder.parameters())
            + list(self.actor.parameters()),
            lr=lr,
        )

    def _compute_weights(self, outcome_rewards: torch.Tensor) -> torch.Tensor:
        """Compute normalized reward weights for the batch.

        Args:
            outcome_rewards: (B,) tensor of trajectory-level rewards.

        Returns:
            (B,) weight tensor (positive, mean ≈ 1.0).
        """
        if self.reward_norm == "softmax":
            # Numerically stable softmax-based normalization
            w = F.softmax(outcome_rewards / self.beta, dim=0) * outcome_rewards.size(0)
        else:
            # Unnormalized exp weighting (may need reward_clip)
            w = (outcome_rewards / self.beta).exp()
            w = w / (w.mean() + 1e-8)  # normalize mean to 1

        # Clip extreme weights for numerical stability
        return w.clamp(min=1.0 / self.reward_clip, max=self.reward_clip)

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """Reward-weighted BC training step."""
        self._total_steps += 1

        s_tokens = self._tokenize(batch.observation_contexts)
        a_tokens = self._tokenize(batch.actions)

        states = self.state_encoder(s_tokens)
        actions_target = self.action_encoder(a_tokens).detach()  # BC target

        # Outcome rewards (trajectory-level signal, per transition)
        outcome_rewards = torch.as_tensor(
            batch.outcome_rewards, dtype=torch.float32
        ).to(self.device)

        # Reward weights: higher reward → more weight
        weights = self._compute_weights(outcome_rewards)  # (B,)

        # Weighted MSE loss
        pred_actions = self.actor(states)                         # (B, action_dim)
        mse = F.mse_loss(pred_actions, actions_target, reduction="none").mean(dim=-1)
        weighted_loss = (weights * mse).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        if self.max_grad_norm is not None:
            clip_grad_norm_(
                list(self.actor.parameters()) + list(self.state_encoder.parameters()),
                self.max_grad_norm,
            )
        self.optimizer.step()

        return TrainMetrics(
            loss=weighted_loss.item(),
            extra={
                "bc_loss": mse.mean().item(),
                "reward_weight_mean": weights.mean().item(),
                "reward_mean": outcome_rewards.mean().item(),
            },
        )

    def get_action_values(
        self, states: list[str], actions: list[str]
    ) -> torch.Tensor:
        """
        Proxy Q-value: negative BC reconstruction loss.

        RW-FT has no Q-function.  Returns -MSE(μ(s), a_data) as a measure
        of how well the trained policy would produce the given action.
        """
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        with torch.no_grad():
            s_emb = self.state_encoder(s_tokens)
            a_emb = self.action_encoder(a_tokens)
            pred = self.actor(s_emb)
            mse = F.mse_loss(pred, a_emb, reduction="none").mean(dim=-1)
        return -mse  # higher = more likely under policy

    def save(self, path) -> None:
        torch.save(
            {
                "state_encoder": self.state_encoder.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "actor": self.actor.state_dict(),
                "total_steps": self._total_steps,
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info("Saved RWFineTuning to %s", path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.actor.load_state_dict(ckpt["actor"])
        self._total_steps = ckpt.get("total_steps", 0)
        logger.info("Loaded RWFineTuning from %s", path)
