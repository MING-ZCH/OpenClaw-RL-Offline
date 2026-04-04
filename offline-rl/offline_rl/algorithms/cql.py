"""
Conservative Q-Learning (CQL) for LLM-based agents.

Reference: Kumar et al., "Conservative Q-Learning for Offline Reinforcement
Learning", NeurIPS 2020 (arXiv:2006.04779)

Key idea: Penalize Q-values for OOD actions by adding a regularizer that
pushes down Q for randomly sampled actions while pushing up Q for
actions in the dataset.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch
from .base import BaseOfflineAlgorithm, StateEncoder, ActionEncoder, TrainMetrics
from .iql import QNetwork

logger = logging.getLogger(__name__)


class CQL(BaseOfflineAlgorithm):
    """
    Conservative Q-Learning for LLM agent offline training.

    Adds a CQL regularizer to standard SAC-style Q-learning:
    L_CQL = α * (log Σ_a exp(Q(s,a)) - E_{a~D}[Q(s,a)])

    This provides a provable lower bound on Q-values, preventing
    overestimation of OOD (out-of-distribution) actions.

    Key hyperparameters:
    - alpha: CQL regularization coefficient (higher = more conservative)
    - n_random_actions: number of random actions for CQL logsumexp estimate
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        state_dim: int = 256,
        action_dim: int = 256,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        alpha: float = 1.0,
        n_random_actions: int = 10,
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
        self.alpha = alpha
        self.n_random_actions = n_random_actions
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

        # Optimizers
        all_params = (
            list(self.state_encoder.parameters())
            + list(self.action_encoder.parameters())
            + list(self.q1.parameters())
            + list(self.q2.parameters())
        )
        self.optimizer = torch.optim.Adam(all_params, lr=lr)

    def _cql_regularizer(self, states: torch.Tensor, data_actions: torch.Tensor) -> torch.Tensor:
        """
        CQL conservative penalty:
        L_CQL = logsumexp(Q(s, a_random)) - E[Q(s, a_data)]
        """
        batch_size = states.shape[0]

        # Generate random action embeddings
        random_actions = torch.randn(
            batch_size, self.n_random_actions, self.action_dim, device=self.device
        )

        # Q for random actions: (batch, n_random)
        states_expanded = states.unsqueeze(1).expand(-1, self.n_random_actions, -1)
        q1_random = self.q1(
            states_expanded.reshape(-1, self.state_dim),
            random_actions.reshape(-1, self.action_dim),
        ).reshape(batch_size, self.n_random_actions)
        q2_random = self.q2(
            states_expanded.reshape(-1, self.state_dim),
            random_actions.reshape(-1, self.action_dim),
        ).reshape(batch_size, self.n_random_actions)

        # Q for data actions
        q1_data = self.q1(states, data_actions)  # (batch, 1)
        q2_data = self.q2(states, data_actions)

        # CQL loss = logsumexp(Q_random) - E[Q_data]
        cql1 = torch.logsumexp(q1_random, dim=1).mean() - q1_data.mean()
        cql2 = torch.logsumexp(q2_random, dim=1).mean() - q2_data.mean()

        return self.alpha * (cql1 + cql2) / 2.0

    def _bellman_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Standard Bellman backup loss."""
        with torch.no_grad():
            # Use min of target Q-networks for next state
            # For simplicity, we use the same action for next state
            # In practice, this would use the policy network
            next_q1 = self.q1_target(next_states, actions)  # Simplified: reuse actions
            next_q2 = self.q2_target(next_states, actions)
            next_q = torch.min(next_q1, next_q2)
            target = rewards.unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * next_q

        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)

        return F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)

    def _soft_update_targets(self):
        self._soft_update_target_pair(self.q1, self.q1_target, self.target_update_rate)
        self._soft_update_target_pair(self.q2, self.q2_target, self.target_update_rate)

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """Single CQL training step."""
        states, actions, next_states, rewards, dones = self._encode_batch(batch)

        # Bellman loss
        td_loss = self._bellman_loss(states, actions, rewards, next_states, dones)

        # CQL conservative regularizer
        cql_loss = self._cql_regularizer(states.detach(), actions.detach())

        total_loss = td_loss + cql_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self._soft_update_targets()

        return TrainMetrics(
            loss=total_loss.item(),
            extra={"td_loss": td_loss.item(), "cql_loss": cql_loss.item()},
        )

    def get_action_values(self, states: list[str], actions: list[str]) -> torch.Tensor:
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        with torch.no_grad():
            s_enc = self.state_encoder(s_tokens)
            a_enc = self.action_encoder(a_tokens)
            q1 = self.q1(s_enc, a_enc)
            q2 = self.q2(s_enc, a_enc)
            return torch.min(q1, q2).squeeze(-1)

    def save(self, path: str) -> None:
        torch.save({
            "state_encoder": self.state_encoder.state_dict(),
            "action_encoder": self.action_encoder.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
