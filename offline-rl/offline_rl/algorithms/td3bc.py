"""
TD3+BC (TD3 with Behavior Cloning) for LLM-based agents.

Reference: Fujimoto & Gu, "A Minimalist Approach to Offline Reinforcement
Learning", NeurIPS 2021 (arXiv:2106.06860)

Key idea: Add a simple BC regularization term to the TD3 policy loss:
    π ← argmax [ λ Q(s, μ(s)) - (a_data - μ(s))^2 ]
where λ = α / (1/N * Σ|Q(s_i, a_i)|) normalizes the Q gradient magnitude
relative to the BC gradient, preventing Q over-optimization.

Compared to IQL or CQL, TD3+BC does not require a separate value network.
The BC term alone is sufficient to keep the policy close to the data manifold,
while Q-learning guides improvement.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch
from .base import BaseOfflineAlgorithm, StateEncoder, ActionEncoder, TrainMetrics
from .iql import QNetwork

logger = logging.getLogger(__name__)


class DeterministicActor(nn.Module):
    """
    Deterministic policy network: state → action embedding.

    TD3+BC uses a deterministic actor (not stochastic) so the BC loss
    is simply the L2 distance between the actor output and the dataset action.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Bounded output for stability
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns deterministic action embedding for each state."""
        return self.net(state)


class TD3BC(BaseOfflineAlgorithm):
    """
    TD3+BC: TD3 with Behavior Cloning for offline LLM agent training.

    Training phases (each step):
    1. Critic update: standard TD3 Bellman with twin Q-networks and
       target policy smoothing.
    2. Actor update (every ``policy_freq`` steps): combined BC + Q loss
       normalized by current Q-value magnitude.

    Policy loss:
        L_actor = -λ * mean(Q(s, μ(s))) + mean(||a_data - μ(s)||²)
        λ = alpha / (1/N * Σ|Q(s_i, a_i)|)

    Key hyperparameters:
    - alpha: relative weight of Q objective vs BC term (default 2.5).
      Higher alpha means more weight on Q improvement; lower alpha stays
      closer to behavioral cloning.
    - target_noise: std of smoothing noise added to target actor actions.
    - noise_clip: clipping range for target noise.
    - policy_freq: actor update interval (default 2 as in vanilla TD3).
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        state_dim: int = 256,
        action_dim: int = 256,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        alpha: float = 2.5,
        target_update_rate: float = 0.005,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_freq: int = 2,
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
        self.target_update_rate = target_update_rate
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self._total_steps = 0

        # Text encoders
        self.state_encoder = StateEncoder(
            vocab_size=self._vocab_size, hidden_dim=state_dim
        ).to(self.device)
        self.action_encoder = ActionEncoder(
            vocab_size=self._vocab_size, hidden_dim=action_dim
        ).to(self.device)

        # Twin Q-networks + targets
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        # Deterministic actor + target
        self.actor = DeterministicActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        # Optimizers
        # Encoder is updated jointly with the actor (not the critic) so the
        # critic backward does not free the encoder graph before actor backward.
        encoder_params = list(self.state_encoder.parameters()) + list(self.action_encoder.parameters())
        self.encoder_optimizer = torch.optim.Adam(encoder_params, lr=lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # Freeze target parameters (updated via Polyak averaging)
        for p in list(self.q1_target.parameters()) + list(self.q2_target.parameters()) + list(self.actor_target.parameters()):
            p.requires_grad = False

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """One combined critic + (delayed) actor update step."""
        self._total_steps += 1

        states, actions, next_states, rewards, dones = self._encode_batch(batch)

        # ------------------------------------------------------------------ #
        # Critic update — use detached encodings so the encoder graph is
        # preserved for the actor backward that follows this step.
        # ------------------------------------------------------------------ #
        states_d = states.detach()
        actions_d = actions.detach()
        next_states_d = next_states.detach()

        with torch.no_grad():
            # Target policy smoothing: add clipped Gaussian noise
            raw_next_action = self.actor_target(next_states_d)
            noise = torch.randn_like(raw_next_action) * self.target_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (raw_next_action + noise).clamp(-1.0, 1.0)

            target_q1 = self.q1_target(next_states_d, next_action).squeeze(-1)
            target_q2 = self.q2_target(next_states_d, next_action).squeeze(-1)
            target_q = torch.min(target_q1, target_q2)
            td_target = rewards + self.gamma * (1.0 - dones) * target_q

        current_q1 = self.q1(states_d, actions_d).squeeze(-1)
        current_q2 = self.q2(states_d, actions_d).squeeze(-1)
        critic_loss = F.mse_loss(current_q1, td_target) + F.mse_loss(current_q2, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.max_grad_norm is not None:
            clip_grad_norm_(
                list(self.q1.parameters()) + list(self.q2.parameters()),
                self.max_grad_norm,
            )
        self.critic_optimizer.step()

        actor_loss_val = 0.0
        lam_val = 0.0

        # ------------------------------------------------------------------ #
        # Delayed actor update — encode fresh so gradients flow to encoder.
        # ------------------------------------------------------------------ #
        if self._total_steps % self.policy_freq == 0:
            pred_actions = self.actor(states)  # states retains encoder graph

            # Q values for normalization (detach to avoid critic graph)
            with torch.no_grad():
                q_norms = self.q1(states_d, actions_d).squeeze(-1)
            lam = self.alpha / (q_norms.abs().mean() + 1e-8)
            lam_val = lam.item()

            # Q-value for actor gradient (through actor output)
            q_vals = self.q1(states_d, pred_actions).squeeze(-1)
            bc_loss = F.mse_loss(pred_actions, actions_d)
            actor_loss = -lam * q_vals.mean() + bc_loss

            self.encoder_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.max_grad_norm is not None:
                clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            self.encoder_optimizer.step()
            actor_loss_val = actor_loss.item()

            # Polyak update: actor target
            self._soft_update_target_pair(self.actor, self.actor_target, self.target_update_rate)

        # Polyak update: critic targets
        self._soft_update_target_pair(self.q1, self.q1_target, self.target_update_rate)
        self._soft_update_target_pair(self.q2, self.q2_target, self.target_update_rate)

        return TrainMetrics(
            loss=critic_loss.item(),
            extra={
                "actor_loss": actor_loss_val,
                "lam": lam_val,
                "q_mean": current_q1.mean().item(),
            },
        )

    def get_action_values(
        self, states: list[str], actions: list[str]
    ) -> torch.Tensor:
        """Compute Q1(s, a) for state-action pairs (used for advantage weighting)."""
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        with torch.no_grad():
            s_vecs = self.state_encoder(s_tokens)
            a_vecs = self.action_encoder(a_tokens)
            q1_vals = self.q1(s_vecs, a_vecs).squeeze(-1)
        return q1_vals

    def get_actor_action(self, states: list[str]) -> torch.Tensor:
        """Return actor's deterministic action embeddings for given states."""
        s_tokens = self._tokenize(states)
        with torch.no_grad():
            s_vecs = self.state_encoder(s_tokens)
            return self.actor(s_vecs)

    def save(self, path: str) -> None:
        """Save all networks and optimizers."""
        torch.save(
            {
                "state_encoder": self.state_encoder.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "alpha": self.alpha,
                "total_steps": self._total_steps,
            },
            path,
        )
        logger.info("Saved TD3+BC to %s", path)

    def load(self, path: str) -> None:
        """Load all networks and optimizers."""
        ckpt = torch.load(path, map_location=self.device)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self._total_steps = ckpt.get("total_steps", 0)
        logger.info("Loaded TD3+BC from %s", path)
