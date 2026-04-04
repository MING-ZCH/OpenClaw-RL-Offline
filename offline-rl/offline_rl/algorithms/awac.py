"""
Advantage Weighted Actor-Critic (AWAC) for LLM-based agents.

Reference: Nair et al., "AWAC: Accelerating Online Reinforcement Learning
with Offline Datasets", 2020 (arXiv:2006.09359)

Key idea: Use advantage-weighted behavioral cloning that seamlessly
transitions from offline to online training without explicit behavior
policy estimation.
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


class PolicyNetwork(nn.Module):
    """
    Policy head that maps (state) → action logits.

    For LLM agents, this would be the LLM's generation head.
    This lightweight version outputs action embeddings for testing.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns predicted action embedding."""
        return self.net(state)

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute log-probability of action given state.
        Uses Gaussian assumption: log p(a|s) ∝ -||a - μ(s)||²
        """
        predicted = self.forward(state)
        # Simplified Gaussian log-prob (unit variance)
        return -0.5 * (action - predicted).pow(2).sum(dim=-1)


class AWAC(BaseOfflineAlgorithm):
    """
    AWAC: Advantage Weighted Actor-Critic for LLM agents.

    Two-phase update:
    1. Critic update: Standard TD learning for Q(s,a)
    2. Actor update: Advantage-weighted behavioral cloning
       π* = argmax_π E[exp(A(s,a)/λ) · log π(a|s)]

    The advantage weighting naturally filters out bad actions from the
    offline dataset without requiring explicit behavior policy estimation.

    Key advantage over IQL/CQL: Seamless offline→online transition.
    When online data becomes available, simply add it to the buffer.

    Key hyperparameters:
    - lam: temperature for advantage weighting (lower = more selective)
    - max_weight: clipping threshold for stability
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        state_dim: int = 256,
        action_dim: int = 256,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 1.0,
        max_weight: float = 100.0,
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
        self.lam = lam
        self.max_weight = max_weight
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

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # Optimizers
        encoder_params = list(self.state_encoder.parameters()) + list(self.action_encoder.parameters())
        self.encoder_optimizer = torch.optim.Adam(encoder_params, lr=lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.actor_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        """Standard TD loss for critic."""
        with torch.no_grad():
            # Use policy to get next actions
            next_actions_pred = self.policy(next_states)
            next_q1 = self.q1_target(next_states, next_actions_pred)
            next_q2 = self.q2_target(next_states, next_actions_pred)
            next_q = torch.min(next_q1, next_q2)
            target = rewards.unsqueeze(-1) + self.gamma * (1 - dones.unsqueeze(-1)) * next_q

        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        critic_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)

        self.critic_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.encoder_optimizer.step()

        return critic_loss.item()

    def _update_actor(self, states: torch.Tensor, actions: torch.Tensor) -> float:
        """Advantage-weighted behavioral cloning."""
        with torch.no_grad():
            q1_val = self.q1(states, actions)
            q2_val = self.q2(states, actions)
            q_val = torch.min(q1_val, q2_val).squeeze(-1)

            # Compute value baseline from policy's predicted actions
            policy_actions = self.policy(states)
            v1 = self.q1(states, policy_actions)
            v2 = self.q2(states, policy_actions)
            v_val = torch.min(v1, v2).squeeze(-1)

            advantage = q_val - v_val
            weights = torch.exp(advantage / self.lam)
            weights = torch.clamp(weights, max=self.max_weight)

        log_probs = self.policy.log_prob(states, actions)
        actor_loss = -(weights * log_probs).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def _soft_update_targets(self):
        self._soft_update_target_pair(self.q1, self.q1_target, self.target_update_rate)
        self._soft_update_target_pair(self.q2, self.q2_target, self.target_update_rate)

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """Single AWAC training step."""
        states, actions, next_states, rewards, dones = self._encode_batch(batch)

        # 1. Update critic
        critic_loss = self._update_critic(states, actions, rewards, next_states, dones)

        # 2. Update actor with advantage-weighted BC
        actor_loss = self._update_actor(states.detach(), actions.detach())

        # 3. Soft update targets
        self._soft_update_targets()

        total_loss = critic_loss + actor_loss
        return TrainMetrics(
            loss=total_loss,
            extra={"critic_loss": critic_loss, "actor_loss": actor_loss},
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

    def predict_actions(self, states: list[str]) -> torch.Tensor:
        """Predict action embeddings from states (for online use)."""
        s_tokens = self._tokenize(states)
        with torch.no_grad():
            s_enc = self.state_encoder(s_tokens)
            return self.policy(s_enc)

    def save(self, path: str) -> None:
        torch.save({
            "state_encoder": self.state_encoder.state_dict(),
            "action_encoder": self.action_encoder.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "policy": self.policy.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        self.policy.load_state_dict(ckpt["policy"])
