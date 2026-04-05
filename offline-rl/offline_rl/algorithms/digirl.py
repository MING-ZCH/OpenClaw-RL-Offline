"""
DigiRL: Doubly-Robust Offline RL for LLM Device-Control Agents.

Reference:
    Bai et al., "DigiRL: Training In-The-Wild Device-Control Agents with
    Autonomous Reinforcement Learning", arXiv:2406.11896 (2024)

Key ideas:
    1. **BCE value functions** instead of MSE: V_step(s) and V_instruct(c) are
       trained as binary classifiers predicting success probability.
    2. **Doubly-robust step advantage**:  blends Monte-Carlo trajectory return
       with TD residual using a mixing weight lambda^(H-h).
    3. **Instruction-level curriculum**: V_instruct(c) estimates per-task
       difficulty; used to focus training on near-frontier instructions.
    4. **Hard-filter AWR**: only transitions with A_step > 1/H contribute to
       the actor loss (filter, not soft-weight).

Compared to existing algorithms in this library:
    - AWAC: soft exponential weighting; DigiRL uses hard filtering.
    - ArCHer: MSE-based IQL critics; DigiRL uses BCE for value learning.
    - GRPO: group-relative normalisation; DigiRL uses doubly-robust advantage.
    - No existing algorithm uses BCE value training or doubly-robust advantage.

Implementation in the discrete-embedding codebase:
    - Inherits BaseOfflineAlgorithm; builds own encoders and networks.
    - StepValueNet: V_step(s, c) ∈ (0, 1) via sigmoid — predicts per-step success.
    - InstructValueNet: V_instruct(c) ∈ (0, 1) — predicts per-instruction success rate.
    - PolicyNetwork reused from off_policy_grpo.
    - Actor uses hard-filtered AWR: loss = -E[log π(a|s)] for transitions where
      A_step(s,a) > threshold.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch
from .base import BaseOfflineAlgorithm, StateEncoder, ActionEncoder, TrainMetrics
from .off_policy_grpo import PolicyNetwork

logger = logging.getLogger(__name__)


class StepValueNet(nn.Module):
    """
    V_step(s) ∈ (0, 1): predicts success probability for current state.

    Trained via binary cross-entropy against the trajectory outcome (binary reward).
    """

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
        """Returns raw logit (apply sigmoid externally for probability)."""
        return self.net(state)


class InstructValueNet(nn.Module):
    """
    V_instruct(s) ∈ (0, 1): predicts success rate for the instruction/task context.

    Uses the same state encoding (which includes the task prefix) to estimate
    overall task difficulty.  Serves as a curriculum signal.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns raw logit."""
        return self.net(state)


class DigiRL(BaseOfflineAlgorithm):
    """
    DigiRL offline RL with doubly-robust advantage and BCE value functions.

    Parameters:
        lam: λ for doubly-robust advantage mixing (0 = pure TD, 1 = pure MC).
        adv_threshold: Hard-filter threshold for AWR actor update.
            DigiRL paper uses 1/H; we default to 0.1.
        actor_lr: Learning rate for the policy network.
        max_grad_norm: Gradient clipping threshold.
        n_actor_updates: Actor gradient steps per train_step call.
        n_value_updates: Value gradient steps per train_step call.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        state_dim: int = 256,
        action_dim: int = 256,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.5,
        adv_threshold: float = 0.1,
        actor_lr: Optional[float] = None,
        max_grad_norm: float = 1.0,
        n_actor_updates: int = 1,
        n_value_updates: int = 1,
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
            max_grad_norm=max_grad_norm,
            **kwargs,
        )
        self.lam = lam
        self.adv_threshold = adv_threshold
        self.n_actor_updates = n_actor_updates
        self.n_value_updates = n_value_updates
        actor_lr = actor_lr or lr

        # Encoders
        self.state_encoder = StateEncoder(
            vocab_size=self._vocab_size, hidden_dim=state_dim
        ).to(self.device)
        self.action_encoder = ActionEncoder(
            vocab_size=self._vocab_size, hidden_dim=action_dim
        ).to(self.device)

        # Value networks — BCE-trained
        self.v_step = StepValueNet(state_dim, hidden_dim).to(self.device)
        self.v_instruct = InstructValueNet(state_dim, hidden_dim).to(self.device)

        # Policy network (actor)
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # Reference policy (frozen, for log-ratio metrics)
        self.ref_policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.ref_policy.load_state_dict(self.policy.state_dict())
        for p in self.ref_policy.parameters():
            p.requires_grad = False

        # Optimizers
        encoder_params = (
            list(self.state_encoder.parameters())
            + list(self.action_encoder.parameters())
        )
        self.encoder_optimizer = torch.optim.Adam(encoder_params, lr=lr)
        self.value_optimizer = torch.optim.Adam(
            list(self.v_step.parameters()) + list(self.v_instruct.parameters()),
            lr=lr,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=actor_lr
        )

    # ------------------------------------------------------------------
    # Value network updates (BCE)
    # ------------------------------------------------------------------

    def _update_values(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor,
    ) -> dict:
        """
        Update V_step and V_instruct via binary cross-entropy.

        Both predict the trajectory outcome (binary success).
        """
        # Labels: use reward as binary target (clamp to [0, 1])
        targets = rewards.clamp(0.0, 1.0).unsqueeze(-1)  # (B, 1)

        total_v_loss = torch.tensor(0.0, device=self.device)

        for _ in range(self.n_value_updates):
            # V_step
            v_step_logit = self.v_step(states.detach())       # (B, 1)
            v_step_loss = F.binary_cross_entropy_with_logits(
                v_step_logit, targets
            )

            # V_instruct — operates on same state embedding (includes task prefix)
            v_inst_logit = self.v_instruct(states.detach())   # (B, 1)
            v_inst_loss = F.binary_cross_entropy_with_logits(
                v_inst_logit, targets
            )

            v_loss = v_step_loss + v_inst_loss

            self.value_optimizer.zero_grad()
            v_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    list(self.v_step.parameters())
                    + list(self.v_instruct.parameters()),
                    self.max_grad_norm,
                )
            self.value_optimizer.step()
            total_v_loss = total_v_loss + v_loss.detach()

        avg_v_loss = total_v_loss.item() / self.n_value_updates
        return {
            "v_step_loss": v_step_loss.item(),
            "v_instruct_loss": v_inst_loss.item(),
        }

    # ------------------------------------------------------------------
    # Doubly-robust advantage computation
    # ------------------------------------------------------------------

    def _compute_dr_advantage(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Doubly-robust step-level advantage (Eq. 4.3 in paper).

        A_step(s_h, a_h) = λ^(H-h) * r_outcome
                         + (1 - λ^(H-h) * r_outcome) * (V_step(s_{h+1}) + r - V_step(s_h))

        Since we don't track H or h in the flat replay buffer, we use a fixed
        λ for the mixing weight: A = λ * r + (1 - λ) * (V(s') + r - V(s)).
        """
        with torch.no_grad():
            v_s = torch.sigmoid(self.v_step(states)).squeeze(-1)      # (B,)
            v_ns = torch.sigmoid(self.v_step(next_states)).squeeze(-1)  # (B,)

        # MC component: pure reward
        mc_return = rewards  # (B,)

        # TD component: V(s') + r - V(s)
        td_advantage = v_ns + rewards - v_s  # (B,)

        # Doubly-robust mix
        advantage = self.lam * mc_return + (1 - self.lam) * td_advantage  # (B,)

        return advantage

    # ------------------------------------------------------------------
    # Actor update (hard-filter AWR)
    # ------------------------------------------------------------------

    def _update_actor(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
    ) -> dict:
        """
        Hard-filter AWR: only train on (s, a) where advantage > threshold.
        """
        total_actor_loss = torch.tensor(0.0, device=self.device)
        n_kept = 0

        for _ in range(self.n_actor_updates):
            # hard filter
            mask = (advantages > self.adv_threshold).float()  # (B,)
            if mask.sum() < 1.0:
                # nothing to train — skip
                continue

            log_pi = self.policy(states.detach(), actions.detach())  # (B,)
            actor_loss = -(mask * log_pi).sum() / mask.sum()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm,
                )
            self.actor_optimizer.step()

            total_actor_loss = total_actor_loss + actor_loss.detach()
            n_kept += int(mask.sum().item())

        avg_actor_loss = total_actor_loss.item() / max(self.n_actor_updates, 1)
        return {
            "actor_loss": avg_actor_loss,
            "actor_n_kept": float(n_kept) / max(self.n_actor_updates, 1),
        }

    # ------------------------------------------------------------------
    # train_step
    # ------------------------------------------------------------------

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """
        DigiRL train step:
        1. Encode batch
        2. Update V_step + V_instruct via BCE
        3. Compute doubly-robust advantages
        4. Update actor via hard-filter AWR
        """
        states, actions, next_states, rewards, dones = self._encode_batch(batch)

        # 1. Update value functions
        v_metrics = self._update_values(states, rewards)

        # 2. Compute advantages (detached)
        advantages = self._compute_dr_advantage(states, next_states, rewards)

        # 3. Update actor
        actor_metrics = self._update_actor(states, actions, advantages)

        # 4. Curriculum difficulty metric
        with torch.no_grad():
            difficulty = torch.sigmoid(self.v_instruct(states)).mean().item()

        total_loss = (
            v_metrics.get("v_step_loss", 0.0)
            + v_metrics.get("v_instruct_loss", 0.0)
            + actor_metrics.get("actor_loss", 0.0)
        )

        extra = {}
        extra.update(v_metrics)
        extra.update(actor_metrics)
        extra["curriculum_difficulty"] = difficulty
        extra["mean_advantage"] = advantages.mean().item()

        return TrainMetrics(loss=total_loss, extra=extra)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_action_values(
        self, states: list, actions: list
    ) -> torch.Tensor:
        """
        Return V_step(s) as the value proxy for advantage weighting.

        V_step is trained to predict P(success|state) ∈ (0, 1).
        """
        s_tokens = self._tokenize(states)
        with torch.no_grad():
            s_enc = self.state_encoder(s_tokens)
            return torch.sigmoid(self.v_step(s_enc)).squeeze(-1)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_encoder": self.state_encoder.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "v_step": self.v_step.state_dict(),
                "v_instruct": self.v_instruct.state_dict(),
                "policy": self.policy.state_dict(),
                "ref_policy": self.ref_policy.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.v_step.load_state_dict(ckpt["v_step"])
        self.v_instruct.load_state_dict(ckpt["v_instruct"])
        self.policy.load_state_dict(ckpt["policy"])
        if "ref_policy" in ckpt:
            self.ref_policy.load_state_dict(ckpt["ref_policy"])
