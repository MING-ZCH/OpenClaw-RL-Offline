"""
CRR: Critic Regularized Regression for offline RL.

Reference: Wang et al., "Critic Regularized Regression", NeurIPS 2020
(arXiv:2006.15134)

Key idea: Combine Q-learning with advantage-weighted behavioral cloning.
The policy is updated to maximize:
    L_π = E[ f(A(s,a)) · log π(a|s) ]
where A(s,a) = Q(s,a) - V(s) is the advantage and f is a filter function.

The advantage baseline V(s) is estimated by Monte-Carlo averaging Q over
K actions sampled from the current policy, avoiding a separate value network:
    V(s) ≈ (1/K) · Σ_{a'~π(·|s)} Q(s, a')

Filter choices:
  - exp (default): f_exp(A) = exp(A / beta)  [advantage-weighted regression]
  - binary:        f_bin(A) = I[A ≥ 0]       [filter out negative advantages]
  - softmax:       f_soft(A) = softmax(A / beta) over the batch

Compared to:
- AWAC (Nair et al. 2021): CRR uses MC-estimated V instead of a learned V.
- IQL (Kostrikov et al. 2022): CRR updates Q via standard TD, not expectile.
- CQL: No explicit out-of-distribution penalty; relies on conservative Q-target.
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
from .edac import GaussianActor

logger = logging.getLogger(__name__)


class CRR(BaseOfflineAlgorithm):
    """
    CRR: Critic Regularized Regression for offline LLM agent training.

    Training phases each step:
    1. Q update: standard Bellman using MC-estimated V(s') as target.
         V(s') ≈ (1/K) · Σ_{a~π(s')} Q_target(s', a)
         Q_target = r + γ (1-d) V(s')
    2. Policy update: advantage-weighted BC on dataset actions.
         A(s,a) = Q(s,a_data) - V(s)   using MC V(s)
         L_π = mean[ f(A(s,a)) · ‖μ(s) - a_data‖² ]
         where f is exp/binary/softmax filter

    Key hyperparameters:
    - beta: temperature for exp/softmax filter (lower = more selective).
    - filter: 'exp' (default), 'binary', or 'softmax'.
    - mc_samples: K actions sampled per state for MC V-estimate (default 8).
    - target_update_rate: Polyak averaging coefficient (default 0.005).
    """

    _VALID_FILTERS = ("exp", "binary", "softmax")

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        state_dim: int = 256,
        action_dim: int = 256,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        beta: float = 1.0,
        filter_type: str = "exp",
        mc_samples: int = 8,
        target_update_rate: float = 0.005,
        device: str = "cpu",
        **kwargs,
    ):
        if filter_type not in self._VALID_FILTERS:
            raise ValueError(
                "filter_type must be one of %s, got: %s"
                % (self._VALID_FILTERS, filter_type)
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
        self.filter_type = filter_type
        self.mc_samples = mc_samples
        self.target_update_rate = target_update_rate
        self._total_steps = 0

        # Text encoders
        self.state_encoder = StateEncoder(
            vocab_size=self._vocab_size, hidden_dim=state_dim
        ).to(self.device)
        self.action_encoder = ActionEncoder(
            vocab_size=self._vocab_size, hidden_dim=action_dim
        ).to(self.device)

        # Twin Q-networks + frozen targets
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for p in list(self.q1_target.parameters()) + list(self.q2_target.parameters()):
            p.requires_grad = False

        # Stochastic actor (used for MC V-estimation and policy update)
        self.actor = GaussianActor(state_dim, action_dim, hidden_dim).to(self.device)

        # Optimizers
        encoder_params = (
            list(self.state_encoder.parameters())
            + list(self.action_encoder.parameters())
        )
        self.encoder_optimizer = torch.optim.Adam(encoder_params, lr=lr)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def _mc_v(
        self,
        states_d: torch.Tensor,
        q1_net: nn.Module,
        q2_net: nn.Module,
    ) -> torch.Tensor:
        """Monte-Carlo V(s) estimate.

        Sample K actions from the current actor and average min(Q1, Q2).
        All operations run under no_grad since this is a target/baseline.

        Args:
            states_d: (B, state_dim) detached state embeddings.

        Returns:
            (B,) estimated V-values.
        """
        B = states_d.size(0)
        with torch.no_grad():
            # Expand states for K samples: (B*K, state_dim)
            states_rep = states_d.unsqueeze(1).expand(B, self.mc_samples, -1)
            states_rep = states_rep.reshape(B * self.mc_samples, -1)
            sampled_a, _, _ = self.actor.sample(states_rep)     # (B*K, action_dim)
            q1_mc = q1_net(states_rep, sampled_a).squeeze(-1)   # (B*K,)
            q2_mc = q2_net(states_rep, sampled_a).squeeze(-1)
            q_mc = torch.min(q1_mc, q2_mc)
            v_est = q_mc.view(B, self.mc_samples).mean(dim=1)   # (B,)
        return v_est

    def _apply_filter(self, advantages: torch.Tensor) -> torch.Tensor:
        """Apply the filter function f to advantages."""
        if self.filter_type == "exp":
            return (advantages / self.beta).exp().clamp(max=100.0)  # stability clip
        if self.filter_type == "binary":
            return (advantages >= 0).float()
        # softmax over batch
        return F.softmax(advantages / self.beta, dim=0) * advantages.size(0)

    def _soft_update_targets(self) -> None:
        self._soft_update_target_pair(self.q1, self.q1_target, self.target_update_rate)
        self._soft_update_target_pair(self.q2, self.q2_target, self.target_update_rate)

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """One CRR training step (Q-update + advantage-weighted policy update)."""
        self._total_steps += 1

        # Tokenize upfront for potential re-encoding
        s_tokens = self._tokenize(batch.observation_contexts)
        a_tokens = self._tokenize(batch.actions)
        ns_tokens = self._tokenize(batch.next_observation_contexts)

        states = self.state_encoder(s_tokens)
        actions = self.action_encoder(a_tokens)
        next_states = self.state_encoder(ns_tokens)

        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32).to(self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32).to(self.device)

        states_d = states.detach()
        actions_d = actions.detach()
        next_states_d = next_states.detach()

        # ── 1. Q-update via Bellman (MC V-target) ─────────────────────────
        v_next = self._mc_v(next_states_d, self.q1_target, self.q2_target)
        td_target = rewards + self.gamma * (1.0 - dones) * v_next

        q1_pred = self.q1(states_d, actions_d).squeeze(-1)
        q2_pred = self.q2(states_d, actions_d).squeeze(-1)
        q_loss = F.mse_loss(q1_pred, td_target) + F.mse_loss(q2_pred, td_target)

        self.q_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        q_loss.backward()
        if self.max_grad_norm is not None:
            clip_grad_norm_(
                list(self.q1.parameters()) + list(self.q2.parameters()),
                self.max_grad_norm,
            )
        self.q_optimizer.step()
        self.encoder_optimizer.step()

        # ── 2. Compute advantage A(s, a_data) = Q(s,a) - V(s) ────────────
        with torch.no_grad():
            q1_val = self.q1_target(states_d, actions_d).squeeze(-1)
            q2_val = self.q2_target(states_d, actions_d).squeeze(-1)
            q_data = torch.min(q1_val, q2_val)
        v_curr = self._mc_v(states_d, self.q1_target, self.q2_target)
        advantages = (q_data - v_curr).detach()

        # ── 3. Advantage-weighted policy update ───────────────────────────
        # Zero grads and re-encode fresh for clean encoder backward
        self.actor_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()

        fresh_states = self.state_encoder(s_tokens)        # fresh graph
        mu, _ = self.actor(fresh_states)                   # deterministic mean

        # BC loss: MSE between predicted mu and dataset action in embed space
        bc_loss = F.mse_loss(mu, actions_d, reduction="none").mean(dim=-1)  # (B,)
        weights = self._apply_filter(advantages)
        actor_loss = (weights * bc_loss).mean()

        actor_loss.backward()
        if self.max_grad_norm is not None:
            clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.encoder_optimizer.step()

        # ── 4. Soft update targets ────────────────────────────────────────
        self._soft_update_targets()

        return TrainMetrics(
            loss=q_loss.item() + actor_loss.item(),
            extra={
                "q_loss": q_loss.item(),
                "actor_loss": actor_loss.item(),
                "advantage_mean": advantages.mean().item(),
                "weight_mean": weights.mean().item(),
            },
        )

    def get_action_values(
        self, states: list[str], actions: list[str]
    ) -> torch.Tensor:
        """Min Q-value for state-action pairs (critic evaluation)."""
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        with torch.no_grad():
            s_emb = self.state_encoder(s_tokens)
            a_emb = self.action_encoder(a_tokens)
            q1 = self.q1(s_emb, a_emb).squeeze(-1)
            q2 = self.q2(s_emb, a_emb).squeeze(-1)
        return torch.min(q1, q2)

    def get_advantages(
        self, states: list[str], actions: list[str]
    ) -> torch.Tensor:
        """Advantage A(s,a) = Q(s,a) - V(s) for offline weighting."""
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        with torch.no_grad():
            s_emb = self.state_encoder(s_tokens)
            a_emb = self.action_encoder(a_tokens)
            q1_val = self.q1(s_emb, a_emb).squeeze(-1)
            q2_val = self.q2(s_emb, a_emb).squeeze(-1)
            q_data = torch.min(q1_val, q2_val)
        v_curr = self._mc_v(s_emb, self.q1, self.q2)
        return (q_data - v_curr).detach()

    def save(self, path) -> None:
        torch.save(
            {
                "state_encoder": self.state_encoder.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
                "actor": self.actor.state_dict(),
                "total_steps": self._total_steps,
                "q_optimizer": self.q_optimizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
            },
            path,
        )
        logger.info("Saved CRR to %s", path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        self.actor.load_state_dict(ckpt["actor"])
        self._total_steps = ckpt.get("total_steps", 0)
        logger.info("Loaded CRR from %s", path)
