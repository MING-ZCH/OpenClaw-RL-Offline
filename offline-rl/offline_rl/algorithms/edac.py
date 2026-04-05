"""
EDAC: Ensemble Diversified Actor-Critic for offline RL.

Reference: An et al., "Uncertainty-Based Offline Reinforcement Learning
with Diversified Q-Ensemble", NeurIPS 2021 (arXiv:2110.01548)

Key idea: Maintain N ensemble Q-networks to quantify out-of-distribution
uncertainty. Pessimistic value estimate under uncertainty:
    Q_pessimistic(s,a) = Q_min(s,a) - eta * Q_std(s,a)
SAC-style stochastic actor with entropy regularization and auto-tuned
temperature alpha to balance exploration vs. conservatism.

Compared to CQL (which adds an explicit penalty term), EDAC uses
implicit uncertainty quantification via ensemble disagreement, which
is more principled and often more stable.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch
from .base import BaseOfflineAlgorithm, StateEncoder, ActionEncoder, TrainMetrics
from .iql import QNetwork

logger = logging.getLogger(__name__)


class GaussianActor(nn.Module):
    """Stochastic Gaussian policy: state embedding → (mu, log_sigma).

    Uses reparameterized sampling to allow actor gradients to flow
    through the sampled action back to the policy parameters and encoder.
    No squashing (tanh) is applied — action lives in the embedding space
    which is already bounded by the action_encoder architecture.
    """

    LOG_STD_MIN: float = -5.0
    LOG_STD_MAX: float = 2.0

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor):
        """Return (mu, log_std) for the Gaussian policy at the given state."""
        h = self.net(state)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, state: torch.Tensor):
        """Reparameterized Gaussian sample.

        Returns:
            action:   (batch, action_dim) sampled action embedding.
            log_prob: (batch,) log prob under current policy.
            mu:       (batch, action_dim) deterministic mean.
        """
        mu, log_std = self(state)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        action = mu + std * eps
        # Gaussian log-prob: sum over action dims
        log_prob = (
            -0.5 * (eps.pow(2) + 2.0 * log_std + math.log(2.0 * math.pi))
        ).sum(dim=-1)
        return action, log_prob, mu


class EDAC(BaseOfflineAlgorithm):
    """
    EDAC: Ensemble Diversified Actor-Critic for offline LLM agent training.

    Training loop each step:
    1. Compute pessimistic TD target:
         y = r + gamma*(1-d) * [ Q_min_target(s',a') - eta*Q_std_target(s',a')
                                  - alpha * log_pi(a'|s') ]
    2. Update all N Q-networks via MSE loss against y.
    3. Actor update (SAC objective):
         L_actor = alpha * log_pi(a|s) - mean_Q_ensemble(s, a)
    4. Auto-tune log_alpha to match target entropy:
         L_alpha = -log_alpha * (log_pi + target_entropy)
    5. Polyak-average all N target Q-networks.

    Key hyperparameters:
    - n_critics: ensemble size (default 10 from paper; 3-5 for CPU tests).
    - eta: uncertainty penalty weight on Q_std (0 = min-only, 1 = SAC-N).
    - auto_alpha: whether to auto-tune SAC temperature (default True).
    - target_entropy: entropy regularization target (default -action_dim).
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        state_dim: int = 256,
        action_dim: int = 256,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        n_critics: int = 10,
        eta: float = 1.0,
        alpha_init: float = 0.1,
        auto_alpha: bool = True,
        target_entropy: Optional[float] = None,
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
        self.n_critics = n_critics
        self.eta = eta
        self.auto_alpha = auto_alpha
        self.target_update_rate = target_update_rate
        self._total_steps = 0
        # Default target entropy heuristic from the SAC paper: -dim(A)
        self._target_entropy = (
            target_entropy if target_entropy is not None else -float(action_dim)
        )

        # Text encoders (shared between critic and actor)
        self.state_encoder = StateEncoder(
            vocab_size=self._vocab_size, hidden_dim=state_dim
        ).to(self.device)
        self.action_encoder = ActionEncoder(
            vocab_size=self._vocab_size, hidden_dim=action_dim
        ).to(self.device)

        # N ensemble Q-networks + frozen target copies
        self.q_ensemble = nn.ModuleList(
            [
                QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
                for _ in range(n_critics)
            ]
        )
        self.q_targets = nn.ModuleList(
            [copy.deepcopy(q) for q in self.q_ensemble]
        )
        for qt in self.q_targets:
            for p in qt.parameters():
                p.requires_grad = False

        # Stochastic actor with reparameterization
        self.actor = GaussianActor(state_dim, action_dim, hidden_dim).to(self.device)

        # SAC temperature (log-space for unconstrained optimization)
        log_alpha_init = math.log(max(alpha_init, 1e-8))
        self.log_alpha = nn.Parameter(
            torch.tensor(log_alpha_init, device=self.device)
        )

        # Optimizers
        encoder_params = (
            list(self.state_encoder.parameters())
            + list(self.action_encoder.parameters())
        )
        self.encoder_optimizer = torch.optim.Adam(encoder_params, lr=lr)
        self.q_optimizer = torch.optim.Adam(
            [p for q in self.q_ensemble for p in q.parameters()], lr=lr
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self) -> torch.Tensor:
        """Current SAC temperature (always > 0 via exp)."""
        return self.log_alpha.exp()

    def _soft_update_targets(self) -> None:
        for q, qt in zip(self.q_ensemble, self.q_targets):
            self._soft_update_target_pair(q, qt, self.target_update_rate)

    def _ensemble_q(
        self,
        nets: nn.ModuleList,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Q-values from a ModuleList of Q-nets.

        Returns:
            (n_critics, batch) float tensor.
        """
        return torch.stack(
            [net(states, actions).squeeze(-1) for net in nets], dim=0
        )

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """One EDAC training step (critic + actor + alpha)."""
        self._total_steps += 1

        # Tokenize upfront for potential re-encoding in actor section
        s_tokens = self._tokenize(batch.observation_contexts)
        a_tokens = self._tokenize(batch.actions)
        ns_tokens = self._tokenize(batch.next_observation_contexts)

        states = self.state_encoder(s_tokens)
        actions = self.action_encoder(a_tokens)
        next_states = self.state_encoder(ns_tokens)

        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32).to(self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32).to(self.device)

        # Detached copies — critic backward must NOT touch encoder graph
        states_d = states.detach()
        actions_d = actions.detach()
        next_states_d = next_states.detach()
        alpha = self.alpha.detach()

        # ── 1. Pessimistic Bellman target ──────────────────────────────────
        with torch.no_grad():
            next_a, next_log_pi, _ = self.actor.sample(next_states_d)
            next_q_all = self._ensemble_q(self.q_targets, next_states_d, next_a)
            next_q_min = next_q_all.min(dim=0).values

            if self.eta > 0.0 and self.n_critics > 1:
                next_q_std = next_q_all.std(dim=0)
                next_q_pen = next_q_min - self.eta * next_q_std
            else:
                next_q_pen = next_q_min

            target_q = (
                rewards
                + self.gamma * (1.0 - dones) * (next_q_pen - alpha * next_log_pi)
            )

        # ── 2. Critic update (all detached → no encoder gradient) ─────────
        q_preds = self._ensemble_q(self.q_ensemble, states_d, actions_d)  # (N, B)
        q_loss = sum(
            F.mse_loss(q_preds[i], target_q) for i in range(self.n_critics)
        )

        self.q_optimizer.zero_grad()
        q_loss.backward()
        if self.max_grad_norm is not None:
            clip_grad_norm_(list(self.q_ensemble.parameters()), self.max_grad_norm)
        self.q_optimizer.step()  # Q params modified in-place here

        # ── 3. Actor update ────────────────────────────────────────────────
        # Re-encode states to create a fresh computation graph that is NOT
        # entangled with any previous backward pass. Then zero gradients BEFORE
        # the forward pass so no stale .grad accumulation from the critic step.
        self.actor_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()

        fresh_states = self.state_encoder(s_tokens)   # fresh graph for encoder grads
        sampled_a, log_pi, _ = self.actor.sample(fresh_states)

        # Temporarily freeze Q ensemble so actor_loss.backward() only updates
        # actor + encoder parameters (gradient through sampled_a still flows).
        for q in self.q_ensemble:
            for p in q.parameters():
                p.requires_grad_(False)

        q_pi_all = self._ensemble_q(self.q_ensemble, states_d, sampled_a)  # (N, B)
        q_pi_mean = q_pi_all.mean(dim=0)
        actor_loss = (alpha * log_pi - q_pi_mean).mean()

        actor_loss.backward()

        # Restore Q ensemble gradient tracking for next critic update
        for q in self.q_ensemble:
            for p in q.parameters():
                p.requires_grad_(True)

        if self.max_grad_norm is not None:
            clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.encoder_optimizer.step()

        # ── 4. Alpha update ────────────────────────────────────────────────
        alpha_loss_val = 0.0
        if self.auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_pi.detach() + self._target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_val = alpha_loss.item()

        # ── 5. Soft update target Q-networks ──────────────────────────────
        self._soft_update_targets()

        return TrainMetrics(
            loss=q_loss.item() + actor_loss.item(),
            extra={
                "q_loss": q_loss.item(),
                "actor_loss": actor_loss.item(),
                "alpha": self.alpha.item(),
                "alpha_loss": alpha_loss_val,
            },
        )

    def get_action_values(
        self, states: list[str], actions: list[str]
    ) -> torch.Tensor:
        """Mean ensemble Q-value for state-action pairs (no grad)."""
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        with torch.no_grad():
            s_emb = self.state_encoder(s_tokens)
            a_emb = self.action_encoder(a_tokens)
            q_all = self._ensemble_q(self.q_ensemble, s_emb, a_emb)  # (N, B)
        return q_all.mean(dim=0)  # (B,)

    def get_advantages(
        self, states: list[str], actions: list[str]
    ) -> torch.Tensor:
        """
        Uncertainty-penalized advantage estimate for offline weighting.

        Returns:
            (B,) tensor of Q_mean - eta * Q_std values (higher = better).
        """
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        with torch.no_grad():
            s_emb = self.state_encoder(s_tokens)
            a_emb = self.action_encoder(a_tokens)
            q_all = self._ensemble_q(self.q_ensemble, s_emb, a_emb)  # (N, B)
            q_mean = q_all.mean(dim=0)
            if self.eta > 0.0 and self.n_critics > 1:
                q_std = q_all.std(dim=0)
                return q_mean - self.eta * q_std
            return q_mean

    def save(self, path) -> None:
        torch.save(
            {
                "state_encoder": self.state_encoder.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "q_ensemble": [q.state_dict() for q in self.q_ensemble],
                "q_targets": [qt.state_dict() for qt in self.q_targets],
                "actor": self.actor.state_dict(),
                "log_alpha": self.log_alpha.data,
                "total_steps": self._total_steps,
                "q_optimizer": self.q_optimizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
            },
            path,
        )
        logger.info("Saved EDAC to %s", path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        for i, q in enumerate(self.q_ensemble):
            q.load_state_dict(ckpt["q_ensemble"][i])
        for i, qt in enumerate(self.q_targets):
            qt.load_state_dict(ckpt["q_targets"][i])
        self.actor.load_state_dict(ckpt["actor"])
        self.log_alpha.data.copy_(ckpt["log_alpha"])
        self._total_steps = ckpt.get("total_steps", 0)
        logger.info("Loaded EDAC from %s", path)
