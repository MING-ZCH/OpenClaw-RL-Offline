"""
OREO: Offline RL for LLM Multi-Step Reasoning (soft Bellman offline RL).

Reference: Wang et al., "Offline Reinforcement Learning for LLM Multi-Step
Reasoning", arXiv 2412.16145, Dec 2024.

Key idea: Treat the LLM agent's trajectory as a MaxEnt MDP and jointly
optimize a policy and a value function via the soft Bellman equation on
offline data. Unlike DPO, OREO does not require paired preference data.
Unlike uniform-reward fine-tuning (RW-FT), OREO assigns per-step credit
through a learned value function.

Algorithm (discrete-time MaxEnt offline RL):
1. **Soft Bellman backup** (Q-update):
        V_soft(s) = β · log Σ_k exp[Q(s, a_k) / β]    # MC log-mean-exp
        Q_target  = r + γ · (1−d) · V_soft(s')
        L_Q = E[ (Q(s,a) − Q_target)² ]

2. **Entropy-regularized policy update** (actor):
        A(s,a) = Q(s,a) − V_soft(s)          # soft advantage
        L_π = E[ exp(A/β) · ‖μ(s) − a_data‖² ]   # advantage-weighted BC

The soft V via log-mean-exp is the key difference from CRR (arithmetic mean)
or IQL (expectile regression):
  - soft V > arithmetic mean V  (Jensen's inequality, giving a more optimistic
    but entropy-consistent baseline)
  - No separate V-network required (V is implicitly defined by Q + policy)

Compared to existing algorithms in this codebase:
  | Property          | OREO     | CRR  | EDAC   | IQL   | RW-FT |
  |-------------------|----------|------|--------|-------|-------|
  | Value function    | Soft Q+V | Q    | Q ens. | Q+V   | None  |
  | V estimate        | logsumexp| mean | mean   | expt. | N/A   |
  | Anti-overestimate | MaxEnt   | KL   | Ensemble|Expect| N/A  |
  | Critic required   | 1 Q      | 2 Q  | N Q    | 3 net | None  |

Hyperparameters:
  - beta: MaxEnt temperature. Lower β → more conservative (sharper policy).
    Typical range: 0.1–2.0; OREO paper uses 1.0.
  - mc_samples: K actions sampled for soft V estimate (default 8).
  - target_update_rate: Polyak averaging for Q_target (default 0.005).
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch
from .base import BaseOfflineAlgorithm, StateEncoder, ActionEncoder, TrainMetrics
from .iql import QNetwork
from .edac import GaussianActor

logger = logging.getLogger(__name__)


class OREO(BaseOfflineAlgorithm):
    """
    OREO: Soft Bellman offline RL for LLM agent trajectories.

    Jointly trains a Q-critic and a stochastic actor using the MaxEnt
    soft Bellman framework on offline (static) trajectory datasets.

    Training each step:
    1. Q-update:
         V_soft(s') = β·log Σ_k exp[Q_target(s', a_k)/β]  (MC, K samples)
         Q_target   = r + γ·(1-d)·V_soft(s')
         L_Q = MSE(Q(s, a_data), Q_target)
    2. Policy update (advantage-weighted BC):
         A(s, a) = Q_target(s, a_data) − V_soft(s)
         L_π = mean[ clip(exp(A/β), max=100) · ‖μ(s) − a_data‖² ]

    Key props:
    - Single Q-network (no twin; entropy regularization prevents divergence)
    - Soft V estimate via log-mean-exp (MaxEnt consistent; more exploratory
      than CRR's arithmetic mean; more principled than IQL's expectile)
    - Autograd fix: re-encode states fresh for actor backward
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
        mc_samples: int = 8,
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
        self.beta = beta
        self.mc_samples = mc_samples
        self.target_update_rate = target_update_rate
        self._total_steps = 0

        self.state_encoder = StateEncoder(
            vocab_size=self._vocab_size, hidden_dim=state_dim
        ).to(self.device)
        self.action_encoder = ActionEncoder(
            vocab_size=self._vocab_size, hidden_dim=action_dim
        ).to(self.device)

        # Single Q-network (+ frozen target)
        self.q = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q_target = copy.deepcopy(self.q)
        for p in self.q_target.parameters():
            p.requires_grad = False

        # Stochastic actor for policy sampling and update
        self.actor = GaussianActor(state_dim, action_dim, hidden_dim).to(self.device)

        encoder_params = (
            list(self.state_encoder.parameters())
            + list(self.action_encoder.parameters())
        )
        self.encoder_optimizer = torch.optim.Adam(encoder_params, lr=lr)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def _soft_v(self, states_d: torch.Tensor) -> torch.Tensor:
        """Soft V(s) = β · log Σ_k exp(Q_target(s, a_k) / β) − β·log K.

        Uses MC approximation: sum over K policy samples.  This is the
        log-mean-exp (or, equivalently, the entropy-consistent value function
        under the MaxEnt optimal policy π*(a|s) ∝ exp(Q*(s,a)/β)).

        Args:
            states_d: (B, state_dim) **detached** state embeddings.

        Returns:
            (B,) soft value estimates.
        """
        B = states_d.size(0)
        K = self.mc_samples
        with torch.no_grad():
            states_rep = (
                states_d.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
            )
            # Sample actions from current actor
            a_k, _, _ = self.actor.sample(states_rep)   # (B*K, D)
            q_vals = self.q_target(states_rep, a_k).squeeze(-1)  # (B*K,)
            q_vals = q_vals.view(B, K)                  # (B, K)
            # log_mean_exp = logsumexp - log(K)
            v_soft = self.beta * (
                torch.logsumexp(q_vals / self.beta, dim=1)
                - torch.log(torch.tensor(float(K), device=self.device))
            )
        return v_soft  # (B,)

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """One OREO training step: Q soft-Bellman update + actor update."""
        self._total_steps += 1

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

        # ── 1. Soft Bellman Q-update ──────────────────────────────────────
        # V_soft(s') = β · log_mean_exp(Q_target(s', a_k)/β)
        v_next_soft = self._soft_v(next_states_d)
        q_target_val = (
            rewards + self.gamma * (1.0 - dones) * v_next_soft
        ).detach()

        q_pred = self.q(states_d, actions_d).squeeze(-1)
        q_loss = F.mse_loss(q_pred, q_target_val)

        self.q_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        q_loss.backward()
        if self.max_grad_norm is not None:
            clip_grad_norm_(self.q.parameters(), self.max_grad_norm)
        self.q_optimizer.step()
        self.encoder_optimizer.step()

        # ── 2. Soft advantage computation ────────────────────────────────
        with torch.no_grad():
            q_data = self.q_target(states_d, actions_d).squeeze(-1)
        v_curr_soft = self._soft_v(states_d)
        soft_adv = (q_data - v_curr_soft).detach()          # (B,)

        # ── 3. Advantage-weighted policy update ──────────────────────────
        # Re-encode fresh for clean backward pass through encoder
        self.actor_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()

        fresh_states = self.state_encoder(s_tokens)
        mu, _ = self.actor(fresh_states)                     # deterministic mean

        bc_loss = F.mse_loss(mu, actions_d, reduction="none").mean(dim=-1)   # (B,)
        weight = (soft_adv / self.beta).exp().clamp(max=100.0)
        actor_loss = (weight * bc_loss).mean()

        actor_loss.backward()
        if self.max_grad_norm is not None:
            clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.encoder_optimizer.step()

        # ── 4. Soft target update ─────────────────────────────────────────
        self._soft_update_target_pair(self.q, self.q_target, self.target_update_rate)

        return TrainMetrics(
            loss=q_loss.item() + actor_loss.item(),
            extra={
                "q_loss": q_loss.item(),
                "actor_loss": actor_loss.item(),
                "soft_v_mean": v_curr_soft.mean().item(),
                "soft_adv_mean": soft_adv.mean().item(),
            },
        )

    def get_action_values(
        self, states: list[str], actions: list[str]
    ) -> torch.Tensor:
        """Soft Q-value for state-action pairs."""
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        with torch.no_grad():
            s_emb = self.state_encoder(s_tokens)
            a_emb = self.action_encoder(a_tokens)
            return self.q(s_emb, a_emb).squeeze(-1)

    def get_advantages(
        self, states: list[str], actions: list[str]
    ) -> torch.Tensor:
        """Soft advantage A(s,a) = Q(s,a) − V_soft(s)."""
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        with torch.no_grad():
            s_emb = self.state_encoder(s_tokens)
            a_emb = self.action_encoder(a_tokens)
            q_val = self.q(s_emb, a_emb).squeeze(-1)
        v_soft = self._soft_v(s_emb.detach())
        return (q_val - v_soft).detach()

    def save(self, path) -> None:
        torch.save(
            {
                "state_encoder": self.state_encoder.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "q": self.q.state_dict(),
                "q_target": self.q_target.state_dict(),
                "actor": self.actor.state_dict(),
                "total_steps": self._total_steps,
                "q_optimizer": self.q_optimizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
            },
            path,
        )
        logger.info("Saved OREO to %s", path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.q.load_state_dict(ckpt["q"])
        self.q_target.load_state_dict(ckpt["q_target"])
        self.actor.load_state_dict(ckpt["actor"])
        self._total_steps = ckpt.get("total_steps", 0)
        logger.info("Loaded OREO from %s", path)
