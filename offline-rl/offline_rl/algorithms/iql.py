"""
Implicit Q-Learning (IQL) for LLM-based agents.

Reference: Kostrikov et al., "Offline Reinforcement Learning with Implicit
Q-Learning", ICLR 2022 (arXiv:2110.06169)

Key idea: Learn V(s) via expectile regression on Q(s,a), avoiding
querying Q for out-of-distribution actions entirely. Extract policy
via advantage-weighted regression.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch
from .base import BaseOfflineAlgorithm, StateEncoder, ActionEncoder, TrainMetrics, TextEncoder

logger = logging.getLogger(__name__)


class QNetwork(nn.Module):
    """Q(s, a) network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim)
            action: (batch, action_dim)
        Returns:
            (batch, 1) Q-values
        """
        return self.net(torch.cat([state, action], dim=-1))


class VNetwork(nn.Module):
    """V(s) value network."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class IQL(BaseOfflineAlgorithm):
    """
    Implicit Q-Learning for LLM agent offline training.

    Three-phase update:
    1. Update V via expectile regression: V ← argmin E_τ[L_τ(Q(s,a) - V(s))]
    2. Update Q via standard Bellman: Q ← r + γ(1-d)V(s')
    3. Extract policy via advantage-weighted BC: π ← argmax E[exp(β·A)·log π(a|s)]

    Key hyperparameters:
    - tau: expectile parameter (>0.5 biases V toward upper quantiles of Q)
    - beta: temperature for policy extraction (higher = more selective)
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        state_dim: int = 256,
        action_dim: int = 256,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.7,
        beta: float = 3.0,
        target_update_rate: float = 0.005,
        device: str = "cpu",
        **kwargs,
    ):
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
        self.tau = tau
        self.beta = beta
        self.target_update_rate = target_update_rate

        # Networks
        self.state_encoder = StateEncoder(
            vocab_size=self._vocab_size, hidden_dim=state_dim
        ).to(self.device)
        self.action_encoder = ActionEncoder(
            vocab_size=self._vocab_size, hidden_dim=action_dim
        ).to(self.device)

        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.v = VNetwork(state_dim, hidden_dim).to(self.device)

        # Optimizers
        encoder_params = list(self.state_encoder.parameters()) + list(self.action_encoder.parameters())
        self.encoder_optimizer = torch.optim.Adam(encoder_params, lr=lr)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.v_optimizer = torch.optim.Adam(self.v.parameters(), lr=lr)

    def _expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        """Asymmetric squared loss for expectile regression."""
        weight = torch.where(diff > 0, self.tau, 1.0 - self.tau)
        return (weight * diff.pow(2)).mean()

    def _update_v(self, states: torch.Tensor, actions: torch.Tensor) -> float:
        """Update V via expectile regression on min(Q1, Q2)."""
        with torch.no_grad():
            q1_val = self.q1_target(states, actions)
            q2_val = self.q2_target(states, actions)
            q_val = torch.min(q1_val, q2_val)

        v_val = self.v(states)
        v_loss = self._expectile_loss(q_val - v_val)

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        return v_loss.item()

    def _update_q(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        """Update Q via Bellman backup using V(s') as target."""
        with torch.no_grad():
            target = rewards.unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * self.v(next_states)

        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        q_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)

        self.q_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        self.encoder_optimizer.step()

        return q_loss.item()

    def _soft_update_targets(self):
        """Polyak averaging for target networks."""
        self._soft_update_target_pair(self.q1, self.q1_target, self.target_update_rate)
        self._soft_update_target_pair(self.q2, self.q2_target, self.target_update_rate)

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """Single IQL training step."""
        states, actions, next_states, rewards, dones = self._encode_batch(batch)

        # 1. Update V via expectile regression
        v_loss = self._update_v(states.detach(), actions.detach())

        # 2. Update Q via Bellman
        q_loss = self._update_q(states, actions, rewards, next_states, dones)

        # 3. Soft update targets
        self._soft_update_targets()

        total_loss = v_loss + q_loss
        return TrainMetrics(
            loss=total_loss,
            extra={"v_loss": v_loss, "q_loss": q_loss},
        )

    def get_action_values(self, states: list[str], actions: list[str]) -> torch.Tensor:
        """Compute Q-values for state-action pairs."""
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        with torch.no_grad():
            s_enc = self.state_encoder(s_tokens)
            a_enc = self.action_encoder(a_tokens)
            q1 = self.q1(s_enc, a_enc)
            q2 = self.q2(s_enc, a_enc)
            return torch.min(q1, q2).squeeze(-1)

    def get_advantages(self, states: list[str], actions: list[str]) -> torch.Tensor:
        """Compute advantages A(s,a) = Q(s,a) - V(s) for policy extraction."""
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        with torch.no_grad():
            s_enc = self.state_encoder(s_tokens)
            a_enc = self.action_encoder(a_tokens)
            q1 = self.q1(s_enc, a_enc)
            q2 = self.q2(s_enc, a_enc)
            q_min = torch.min(q1, q2)
            v = self.v(s_enc)
            return (q_min - v).squeeze(-1)

    def get_policy_weights(self, states: list[str], actions: list[str]) -> torch.Tensor:
        """
        Compute advantage-weighted policy extraction weights.
        These weights can be used to re-weight the LLM's log-probabilities
        during fine-tuning: loss = -Σ w_i * log π(a_i | s_i)
        """
        advantages = self.get_advantages(states, actions)
        weights = torch.exp(self.beta * advantages)
        weights = torch.clamp(weights, max=100.0)  # Stability
        return weights / weights.sum() * len(weights)  # Normalize

    def save(self, path: str) -> None:
        torch.save({
            "state_encoder": self.state_encoder.state_dict(),
            "action_encoder": self.action_encoder.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "v": self.v.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        self.v.load_state_dict(ckpt["v"])
