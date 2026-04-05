"""
GLIDER: Hierarchical Offline RL for LLM Agents via Latent Plan Decomposition.

Decomposes the long-horizon agent decision problem into two abstraction levels:

  High-level planner
  ------------------
  * ``PlanEncoder``: a shallow MLP that maps a state embedding onto a compact
    *plan embedding* g ∈ R^{plan_dim}, which captures *what the agent intends
    to accomplish* in the near future.
  * ``HighLevelValueNet``: IQL-style expectile value function V_H(s) trained
    on *sparsely-labelled* trajectory-level outcome rewards.  This teaches V_H
    which states are "near success" even without step-level annotations.
  * High-level policy loss: advantage-weighted regression (AWR) that pushes
    the plan encoder toward plan embeddings that correlate with high-return
    states,  loss = E[ exp(A_H / β) · ||g(s) − g_target||² ] where
    g_target = plan_encoder(s') (next-step plan as a proxy target).

  Low-level executor
  ------------------
  * ``LowLevelQNet``: IQL Q-function on the plan-augmented state (s ‖ g).
    Learns to predict step-level value conditioned on both the observation
    and the high-level plan, sharply reducing effective horizon.
  * ``LowLevelValueNet``: companion V function for the low-level IQL update.
  * Low-level actor: advantage-weighted BC that regresses the action embedding
    toward stored expert actions, weighted by exp(A_L / β).

  Training order per ``train_step``:
    1. Encode s, a, s' (with grad for LL encoder training).
    2. High-level: update V_H + plan_encoder with combined AWR loss.
    3. Low-level critic: IQL Q + V Bellman update on (s‖g, a, s'‖g').
    4. Low-level actor: advantage-weighted BC regression.
    5. Polyak soft-update of LL Q target networks.

Differences from the GLIDER paper vs. this implementation:
  - GLIDER originally generates natural language sub-goal descriptions via an
    LLM; here *plan embeddings* are latent continuous vectors of dimension
    ``plan_dim`` — no LLM text generation needed.
  - Full offline mode: no online rollouts; trains entirely on the replay buffer.

References:
    Hu et al., "GLIDER: Offline Hierarchical Reinforcement Learning for
    Multi-Step LLM Agent Planning", ICML 2025 (arXiv 2505.19761).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch
from .base import BaseOfflineAlgorithm, StateEncoder, ActionEncoder, TrainMetrics
from .iql import QNetwork, VNetwork

logger = logging.getLogger(__name__)


class PlanEncoder(nn.Module):
    """
    Maps a state embedding to a compact latent plan embedding.

    Captures high-level task intent as a continuous vector, analogous to a
    sub-goal description in the GLIDER paper.

    Args:
        state_dim: Dimension of the input state embedding.
        plan_dim: Dimension of the output plan embedding.
    """

    def __init__(self, state_dim: int, plan_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, plan_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim) state embedding.
        Returns:
            plan: (B, plan_dim) plan embedding.
        """
        return self.net(state)


class GLIDER(BaseOfflineAlgorithm):
    """
    Hierarchical offline RL for LLM agents: plan encoder + IQL critics.

    Two-level decomposition:
      - High level: ``PlanEncoder`` + ``HighLevelValueNet`` (IQL expectile).
      - Low level:  ``LowLevelQNet`` + ``LowLevelValueNet`` + actor (IQL).

    Args:
        replay_buffer: Offline dataset of stored trajectories.
        plan_dim: Dimension of the latent plan embedding.  Defaults to
            ``hidden_dim // 4`` (a small bottleneck representation).
        beta: Temperature for advantage-weighted regression (AWR).  Larger
            values are more selective; smaller values approach BC.
        tau: Expectile parameter for IQL V-function training (default 0.7).
        target_update_rate: Polyak averaging rate for LL Q-target networks.
        **kwargs: Forwarded to :class:`BaseOfflineAlgorithm`.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        plan_dim: Optional[int] = None,
        beta: float = 1.0,
        tau: float = 0.7,
        target_update_rate: float = 0.005,
        state_dim: int = 256,
        action_dim: int = 256,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
        max_token_len: int = 128,
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
            max_token_len=max_token_len,
        )
        self.plan_dim = plan_dim if plan_dim is not None else max(hidden_dim // 4, 16)
        self.beta = beta
        self.tau = tau
        self.target_update_rate = target_update_rate

        # Shared text encoders
        self.state_encoder = StateEncoder(
            self._vocab_size, 256, state_dim
        ).to(self.device)
        self.action_encoder = ActionEncoder(
            self._vocab_size, 256, action_dim
        ).to(self.device)

        # -----------------------------------------------------------
        # High-level: plan encoder + value function
        # -----------------------------------------------------------
        self.plan_encoder = PlanEncoder(state_dim, self.plan_dim).to(self.device)
        self.hl_value = VNetwork(state_dim, hidden_dim).to(self.device)

        # -----------------------------------------------------------
        # Low-level: IQL critics + actor conditioned on plan
        # -----------------------------------------------------------
        ll_state_dim = state_dim + self.plan_dim  # plan-augmented state dim

        self.ll_q1 = QNetwork(ll_state_dim, action_dim, hidden_dim).to(self.device)
        self.ll_q2 = QNetwork(ll_state_dim, action_dim, hidden_dim).to(self.device)
        self.ll_q1_tgt = QNetwork(ll_state_dim, action_dim, hidden_dim).to(self.device)
        self.ll_q2_tgt = QNetwork(ll_state_dim, action_dim, hidden_dim).to(self.device)
        self.ll_q1_tgt.load_state_dict(self.ll_q1.state_dict())
        self.ll_q2_tgt.load_state_dict(self.ll_q2.state_dict())
        for p in list(self.ll_q1_tgt.parameters()) + list(self.ll_q2_tgt.parameters()):
            p.requires_grad = False

        self.ll_value = VNetwork(ll_state_dim, hidden_dim).to(self.device)

        # Low-level actor: (s ‖ g) → action embedding (regressed against stored a)
        self.ll_actor = nn.Sequential(
            nn.Linear(ll_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        ).to(self.device)

        # -----------------------------------------------------------
        # Optimizers
        # -----------------------------------------------------------
        # High-level: plan_encoder + hl_value
        self.hl_optimizer = torch.optim.Adam(
            list(self.plan_encoder.parameters()) + list(self.hl_value.parameters()),
            lr=lr,
        )
        # Low-level critic: state_encoder + action_encoder + ll_q1/q2 + ll_value
        self.ll_critic_optimizer = torch.optim.Adam(
            list(self.state_encoder.parameters())
            + list(self.action_encoder.parameters())
            + list(self.ll_q1.parameters())
            + list(self.ll_q2.parameters())
            + list(self.ll_value.parameters()),
            lr=lr,
        )
        # Low-level actor
        self.ll_actor_optimizer = torch.optim.Adam(
            self.ll_actor.parameters(), lr=lr
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        """Asymmetric L2 (expectile regression)."""
        weight = torch.where(diff > 0, self.tau, 1.0 - self.tau)
        return (weight * diff.pow(2)).mean()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """
        One GLIDER training step:

          1. Encode states and actions (gradients enabled for LL critic).
          2. High-level V + plan AWR update.
          3. Low-level IQL critic (Q + V) update.
          4. Low-level actor AWR update.
          5. Polyak soft-update of LL Q targets.
        """
        rewards = torch.as_tensor(
            batch.rewards, dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            batch.dones, dtype=torch.float32, device=self.device
        )
        outcome_rewards = torch.as_tensor(
            batch.outcome_rewards, dtype=torch.float32, device=self.device
        )

        # --- Encode ---------------------------------------------------
        s = self.state_encoder(self._tokenize(batch.observation_contexts))
        a = self.action_encoder(self._tokenize(batch.actions))
        s_next = self.state_encoder(
            self._tokenize(batch.next_observation_contexts)
        )

        # === Step 2: High-level V + plan AWR update ===================
        # Use detached s so HL grad does not flow through state_encoder.
        s_det = s.detach()
        s_next_det = s_next.detach()

        # High-level plan embeddings (with grad for plan_encoder update)
        g = self.plan_encoder(s_det)  # (B, plan_dim) — graph through plan_encoder

        # HL value V_H(s): expectile regression on outcome_rewards
        hl_v = self.hl_value(s_det).squeeze(-1)  # (B,)
        hl_v_loss = self._expectile_loss(outcome_rewards - hl_v)

        # HL plan AWR: A_H = outcome_r - V_H(s).detach()
        a_hl = (outcome_rewards - hl_v.detach()).clamp(min=-10.0, max=10.0)
        weights_hl = torch.exp(a_hl / self.beta).clamp(max=20.0)

        with torch.no_grad():
            # g_target: plan embedding for s' as proxy regression target
            g_target = self.plan_encoder(s_next_det)

        # plan_loss: advantage-weighted MSE between g(s) and g(s')
        plan_loss = (
            weights_hl * (g - g_target.detach()).pow(2).mean(dim=-1)
        ).mean()

        hl_total_loss = hl_v_loss + plan_loss

        self.hl_optimizer.zero_grad()
        hl_total_loss.backward()
        clip_grad_norm_(
            list(self.plan_encoder.parameters()) + list(self.hl_value.parameters()),
            self.max_grad_norm or 1.0,
        )
        self.hl_optimizer.step()

        # === Step 3: Low-level IQL critic update ======================
        # Plan-augmented state: cat([s, g.detach()], dim=-1)
        g_det = g.detach()
        with torch.no_grad():
            g_next = self.plan_encoder(s_next_det)
        s_aug = torch.cat([s, g_det], dim=-1)          # (B, state_dim + plan_dim)
        s_next_aug = torch.cat([s_next, g_next], dim=-1)

        # Q Bellman target: r + γ(1-d)·V_L(s'_aug)
        with torch.no_grad():
            ll_v_next = self.ll_value(s_next_aug.detach()).squeeze(-1)
            bellman_tgt = rewards + self.gamma * (1.0 - dones) * ll_v_next

        ll_q1_pred = self.ll_q1(s_aug, a).squeeze(-1)
        ll_q2_pred = self.ll_q2(s_aug, a).squeeze(-1)
        ll_q_loss = (
            F.mse_loss(ll_q1_pred, bellman_tgt)
            + F.mse_loss(ll_q2_pred, bellman_tgt)
        )

        # V IQL expectile regression on frozen Q targets
        with torch.no_grad():
            ll_q1_t = self.ll_q1_tgt(s_aug.detach(), a.detach()).squeeze(-1)
            ll_q2_t = self.ll_q2_tgt(s_aug.detach(), a.detach()).squeeze(-1)
            ll_q_min_t = torch.min(ll_q1_t, ll_q2_t)

        ll_v = self.ll_value(s_aug.detach()).squeeze(-1)
        ll_v_loss = self._expectile_loss(ll_q_min_t - ll_v)

        ll_critic_total = ll_q_loss + ll_v_loss
        self.ll_critic_optimizer.zero_grad()
        ll_critic_total.backward()
        clip_grad_norm_(
            list(self.state_encoder.parameters())
            + list(self.action_encoder.parameters())
            + list(self.ll_q1.parameters())
            + list(self.ll_q2.parameters())
            + list(self.ll_value.parameters()),
            self.max_grad_norm or 1.0,
        )
        self.ll_critic_optimizer.step()

        # === Step 4: Low-level actor AWR update =======================
        with torch.no_grad():
            s_aug_det = torch.cat([s.detach(), g_det], dim=-1)
            ll_q1_a = self.ll_q1(s_aug_det, a.detach()).squeeze(-1)
            ll_q2_a = self.ll_q2(s_aug_det, a.detach()).squeeze(-1)
            ll_q_a = torch.min(ll_q1_a, ll_q2_a)
            ll_v_a = self.ll_value(s_aug_det).squeeze(-1)
            a_ll = (ll_q_a - ll_v_a).clamp(min=-10.0, max=10.0)
            weights_ll = torch.exp(a_ll / self.beta).clamp(max=20.0)

        # Actor regresses its output toward the stored action embedding
        a_pred = self.ll_actor(s_aug_det)  # (B, action_dim)
        actor_loss = (
            weights_ll.unsqueeze(-1) * (a_pred - a.detach()).pow(2)
        ).mean()

        self.ll_actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.ll_actor.parameters(), self.max_grad_norm or 1.0)
        self.ll_actor_optimizer.step()

        # === Step 5: Polyak soft-update LL Q targets ==================
        self._soft_update_target_pair(self.ll_q1, self.ll_q1_tgt, self.target_update_rate)
        self._soft_update_target_pair(self.ll_q2, self.ll_q2_tgt, self.target_update_rate)

        total_loss = (
            hl_v_loss.item()
            + plan_loss.item()
            + ll_critic_total.item()
            + actor_loss.item()
        )

        return TrainMetrics(
            loss=total_loss,
            extra={
                "hl_v_loss": hl_v_loss.item(),
                "plan_loss": plan_loss.item(),
                "ll_q_loss": ll_q_loss.item(),
                "ll_v_loss": ll_v_loss.item(),
                "ll_actor_loss": actor_loss.item(),
                "plan_emb_norm": g_det.norm(dim=-1).mean().item(),
            },
        )

    # ------------------------------------------------------------------
    # Inference: return LL Q-values conditioned on plan embeddings
    # ------------------------------------------------------------------

    def get_action_values(
        self, states: List[str], actions: List[str]
    ) -> torch.Tensor:
        """
        Return low-level Q-values conditioned on the current plan embedding.

        The plan embedding is produced by the plan encoder on-the-fly from
        each state's embedding, giving Q(s ‖ g(s), a) as the value estimate.
        """
        with torch.no_grad():
            s = self.state_encoder(self._tokenize(states))
            a = self.action_encoder(self._tokenize(actions))
            g = self.plan_encoder(s)
            s_aug = torch.cat([s, g], dim=-1)
            q1 = self.ll_q1(s_aug, a).squeeze(-1)
            q2 = self.ll_q2(s_aug, a).squeeze(-1)
            return torch.min(q1, q2)  # conservative Q-estimate

    def get_plan_embeddings(self, states: List[str]) -> torch.Tensor:
        """Return plan embeddings for a list of observation strings."""
        with torch.no_grad():
            s = self.state_encoder(self._tokenize(states))
            return self.plan_encoder(s)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save all model parameters."""
        torch.save(
            {
                "state_encoder": self.state_encoder.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "plan_encoder": self.plan_encoder.state_dict(),
                "hl_value": self.hl_value.state_dict(),
                "ll_q1": self.ll_q1.state_dict(),
                "ll_q2": self.ll_q2.state_dict(),
                "ll_q1_tgt": self.ll_q1_tgt.state_dict(),
                "ll_q2_tgt": self.ll_q2_tgt.state_dict(),
                "ll_value": self.ll_value.state_dict(),
                "ll_actor": self.ll_actor.state_dict(),
            },
            path,
        )
        logger.info("Saved GLIDER to %s", path)

    def load(self, path: str) -> None:
        """Load all model parameters."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.plan_encoder.load_state_dict(ckpt["plan_encoder"])
        self.hl_value.load_state_dict(ckpt["hl_value"])
        self.ll_q1.load_state_dict(ckpt["ll_q1"])
        self.ll_q2.load_state_dict(ckpt["ll_q2"])
        self.ll_q1_tgt.load_state_dict(ckpt["ll_q1_tgt"])
        self.ll_q2_tgt.load_state_dict(ckpt["ll_q2_tgt"])
        self.ll_value.load_state_dict(ckpt["ll_value"])
        self.ll_actor.load_state_dict(ckpt["ll_actor"])
        logger.info("Loaded GLIDER from %s", path)
