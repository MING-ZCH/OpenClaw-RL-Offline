"""
VEM (Value Environment Model) for offline RL.

Two-stage algorithm that trains a value environment model to predict
state-action values from offline data, then uses it as a reward signal
to optimize the policy via advantage-weighted regression (AWR).

Stage 1: Pretrain a VEM network (MLP) that maps (state, action) → scalar
          value prediction, trained with MSE on ground-truth returns.
Stage 2: Freeze the VEM and use it as the advantage signal for AWR-based
          policy optimization.

In this implementation, both stages are combined into a single train_step
for joint training (VEM parameters are updated with detached encoder
outputs while the policy update flows gradients through the encoders).

Reference:
    Song et al., "VEM: Environment-Free Exploration for Training GUI Agents",
    Microsoft Research 2025 (arXiv:2502.18906)
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch
from .base import BaseOfflineAlgorithm, StateEncoder, ActionEncoder, TextEncoder, TrainMetrics
from .off_policy_grpo import PolicyNetwork

logger = logging.getLogger(__name__)
_LOG_EPS = 1e-8


class ValueEnvironmentModel(nn.Module):
    """MLP that predicts state-action value from encoded representations."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: (B, state_dim)
            actions: (B, action_dim)
        Returns:
            (B,) predicted values
        """
        x = torch.cat([states, actions], dim=-1)
        return self.net(x).squeeze(-1)


class VEM(BaseOfflineAlgorithm):
    """
    Value Environment Model (VEM) for offline RL.

    Algorithm outline:
        1. Encode observations and actions via text encoders.
        2. Update VEM: predict outcome rewards from detached encoder
           outputs using MSE loss.
        3. Compute advantages from frozen VEM predictions, clamp and
           exponentiate with temperature beta for AWR weighting.
        4. Update policy + encoders: advantage-weighted regression loss
           L_AWR = -sum(weights * log_probs).

    Parameters:
        replay_buffer: Buffer with pre-collected transitions.
        vem_lr: Learning rate for the VEM network.
        policy_lr: Learning rate for the policy network.
        beta: AWR temperature for advantage weighting.
        alpha_awr: Weight of AWR loss term (for logging the combined metric).
        vem_pretrain_steps: Stored for reference; VEM pretraining is done
            externally in production.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        state_dim: int = 256,
        action_dim: int = 256,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
        max_token_len: int = 128,
        vem_lr: float = 3e-4,
        policy_lr: float = 3e-4,
        beta: float = 1.0,
        alpha_awr: float = 1.0,
        vem_pretrain_steps: int = 0,
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
        self.beta = beta
        self.alpha_awr = alpha_awr
        self.vem_pretrain_steps = vem_pretrain_steps

        # Encoders
        self.state_encoder = StateEncoder(self._vocab_size, 256, state_dim).to(self.device)
        self.action_encoder = ActionEncoder(self._vocab_size, 256, action_dim).to(self.device)

        # VEM value network
        self.vem = ValueEnvironmentModel(state_dim, action_dim, hidden_dim).to(self.device)

        # Policy network
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # Separate optimizers
        self.encoder_optimizer = torch.optim.Adam(
            list(self.state_encoder.parameters()) + list(self.action_encoder.parameters()),
            lr=lr,
        )
        self.vem_optimizer = torch.optim.Adam(self.vem.parameters(), lr=vem_lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """
        Joint VEM + AWR training step.

        The VEM is updated on detached encoder outputs (no encoder gradient).
        The policy and encoders are updated via AWR with advantages derived
        from the freshly-updated VEM.
        """
        # Encode
        s_tokens = self._tokenize(batch.observation_contexts)
        a_tokens = self._tokenize(batch.actions)
        states = self.state_encoder(s_tokens)
        actions = self.action_encoder(a_tokens)

        outcome_rewards = torch.as_tensor(
            batch.outcome_rewards, dtype=torch.float32,
        ).to(self.device)

        # --- VEM update (detached encoder outputs) ---
        self.vem_optimizer.zero_grad()
        vem_pred = self.vem(states.detach(), actions.detach())
        vem_loss = F.mse_loss(vem_pred, outcome_rewards)
        vem_loss.backward()
        self.vem_optimizer.step()

        # --- AWR update (encoder + policy) ---
        with torch.no_grad():
            vem_values = self.vem(states.detach(), actions.detach())
            advantages = vem_values - vem_values.mean()
            advantages = advantages.clamp(-5.0, 5.0)
            weights = torch.exp(self.beta * advantages)
            weights = weights / weights.sum().clamp(min=_LOG_EPS)

        self.encoder_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        log_probs = self.policy(states, actions)
        awr_loss = -(weights * log_probs).sum()
        awr_loss.backward()
        self.encoder_optimizer.step()
        self.policy_optimizer.step()

        return TrainMetrics(
            loss=vem_loss.item() + self.alpha_awr * awr_loss.item(),
            extra={
                "vem_loss": vem_loss.item(),
                "awr_loss": awr_loss.item(),
                "mean_advantage": advantages.mean().item(),
                "mean_vem_pred": vem_pred.mean().item(),
            },
        )

    def get_action_values(
        self, states: list[str], actions: list[str],
    ) -> torch.Tensor:
        """Return VEM-predicted values for state-action pairs."""
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        with torch.no_grad():
            s_enc = self.state_encoder(s_tokens)
            a_enc = self.action_encoder(a_tokens)
            return self.vem(s_enc, a_enc)

    def save(self, path: str) -> None:
        """Save model parameters to file."""
        torch.save(
            {
                "state_encoder": self.state_encoder.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "vem": self.vem.state_dict(),
                "policy": self.policy.state_dict(),
            },
            path,
        )
        logger.info("Saved VEM to %s", path)

    def load(self, path: str) -> None:
        """Load model parameters from file."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.vem.load_state_dict(ckpt["vem"])
        self.policy.load_state_dict(ckpt["policy"])
        logger.info("Loaded VEM from %s", path)
