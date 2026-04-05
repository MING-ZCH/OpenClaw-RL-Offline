"""
BCQ: Batch-Constrained Q-learning for offline RL.

Reference:
    Fujimoto et al., "Off-Policy Deep Reinforcement Learning without
    Exploration", ICML 2019 (arXiv:1812.02900)

Key idea:
    Unconstrained Q-learning on offline data suffers from *extrapolation error*:
    the Q-function is queried at out-of-distribution (OOD) (state, action) pairs
    via the Bellman backup, causing overestimation and divergence.  BCQ addresses
    this by *constraining the policy to stay close to the behavior distribution*
    from which the offline data was collected.

    In the original continuous-action BCQ, a conditional VAE G_ω(s) generates
    action candidates, and a perturbation network ξ refines them.  In our
    discrete-embedding setup, we simplify by using a lightweight Behavior-Cloning
    MLP (BCNet) that predicts the mean action embedding:

        π_BCQ(s) ≈ argmax_{a near BC(s)}  Q(s, a)

    The BCNet is trained jointly via MSE on dataset actions, preventing the Q
    function from exploiting OOD actions during Bellman backup.

Algorithm implemented here:
    1. **Q update** (Bellman): Q_target = r + γ(1−d)·V(s')  [IQL-style V backup]
    2. **V update** (expectile): V ← argmin_V E_τ[L_τ(Q_tgt − V)]
    3. **BC update**: BC(s) ≈ a_data  via MSE(BC(s), a_data)
    4. **Polyak** on Q targets

    Policy extraction (inference): `get_action_values(s, a)` computes Q(s,a)
    with the constraint that the Q values are only meaningful near BC(s).
    `get_constrained_action(s)` returns BC(s) as the safe action embedding.

Compared to TD3BC in this library:
    - TD3BC: adds a BC term to the actor loss (online actor), uses TD3 Q-learning.
    - BCQ:   trains an explicit behavior model, constrains the policy implicitly
             via BC-proximity at inference time.  No separate actor network.
    - BCQ:   also implements `get_advantages(s,a) = Q(s,a) − V(s)` for external
             policy extraction (like IQL).

Architecture:
    - state_encoder / action_encoder: TextEncoder
    - q1, q2, q1_target, q2_target: QNetwork  (twin-Q)
    - v: VNetwork                              (expectile value)
    - bc_net: BehaviorCloningNetwork           (s → predicted action embedding)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch
from .base import BaseOfflineAlgorithm, StateEncoder, ActionEncoder, TextEncoder, TrainMetrics
from .iql import QNetwork, VNetwork

logger = logging.getLogger(__name__)


class BehaviorCloningNetwork(nn.Module):
    """
    BehaviorCloningNetwork: s → predicted action embedding.

    A simple MLP that learns to reproduce the behavior policy's
    action distribution over the offline dataset.  Used in BCQ to
    constrain the implicit Q-policy to stay near dataset actions.
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
        """
        Args:
            state: (batch, state_dim)
        Returns:
            predicted action embedding (batch, action_dim)
        """
        return self.net(state)


class BCQ(BaseOfflineAlgorithm):
    """
    Batch-Constrained Q-learning for offline LLM agent training.

    Combines IQL-style twin-Q + V critic learning with an explicit
    Behavior Cloning network to prevent Q-overestimation on
    out-of-distribution actions.

    Parameters:
        replay_buffer: Pre-collected transition dataset.
        state_dim: State embedding dimension (default 256).
        action_dim: Action embedding dimension (default 256).
        hidden_dim: Hidden MLP width (default 256).
        lr: Learning rate for encoder + critic + BC optimizers (default 3e-4).
        gamma: Discount factor (default 0.99).
        tau: IQL expectile for V update (default 0.7).
        target_update_rate: Polyak coefficient for Q target updates (default 0.005).
        bc_weight: Weight of BC loss relative to Q loss (default 1.0).
        device: Torch device string (default "cpu").
        **kwargs: Forwarded to BaseOfflineAlgorithm.
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
        target_update_rate: float = 0.005,
        bc_weight: float = 1.0,
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
        self.target_update_rate = target_update_rate
        self.bc_weight = bc_weight

        # ------------------------------------------------------------------ #
        # Encoders                                                             #
        # ------------------------------------------------------------------ #
        self.state_encoder = StateEncoder(
            vocab_size=self._vocab_size, hidden_dim=state_dim
        ).to(self.device)
        self.action_encoder = ActionEncoder(
            vocab_size=self._vocab_size, hidden_dim=action_dim
        ).to(self.device)

        # ------------------------------------------------------------------ #
        # Critic: twin-Q + V with target networks                             #
        # ------------------------------------------------------------------ #
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.v = VNetwork(state_dim, hidden_dim).to(self.device)

        # ------------------------------------------------------------------ #
        # Behavior cloning network                                             #
        # ------------------------------------------------------------------ #
        self.bc_net = BehaviorCloningNetwork(state_dim, action_dim, hidden_dim).to(
            self.device
        )

        # ------------------------------------------------------------------ #
        # Optimizers                                                           #
        # ------------------------------------------------------------------ #
        encoder_params = (
            list(self.state_encoder.parameters())
            + list(self.action_encoder.parameters())
        )
        self.encoder_optimizer = torch.optim.Adam(encoder_params, lr=lr)

        critic_params = (
            list(self.q1.parameters())
            + list(self.q2.parameters())
            + list(self.v.parameters())
        )
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=lr)

        # BC net does NOT touch encoders — it operates on detached state embeddings
        self.bc_optimizer = torch.optim.Adam(self.bc_net.parameters(), lr=lr)

    # ---------------------------------------------------------------------- #
    # Private helpers                                                          #
    # ---------------------------------------------------------------------- #

    def _expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        """Asymmetric L2 expectile loss matching IQL pattern."""
        weight = torch.where(diff > 0, self.tau, 1.0 - self.tau)
        return (weight * diff.pow(2)).mean()

    def _update_v(self, states: torch.Tensor, actions: torch.Tensor) -> float:
        """Update V via expectile regression on min(Q1_tgt, Q2_tgt)."""
        with torch.no_grad():
            q_min = torch.min(
                self.q1_target(states, actions),
                self.q2_target(states, actions),
            )
        v_pred = self.v(states)
        v_loss = self._expectile_loss(q_min - v_pred)

        self.critic_optimizer.zero_grad()
        v_loss.backward()
        self.critic_optimizer.step()
        return v_loss.item()

    def _update_q(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        """Update Q via Bellman backup using V(s') as bootstrap target."""
        with torch.no_grad():
            v_next = self.v(next_states)
            q_target = (
                rewards.unsqueeze(-1)
                + self.gamma * (1.0 - dones.unsqueeze(-1)) * v_next
            )

        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        q_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

        self.critic_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()
        self.encoder_optimizer.step()
        return q_loss.item()

    def _update_bc(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> float:
        """
        Update behavior-cloning network: MSE(BC(s), a_data).

        BC operates on detached encoder outputs to prevent BC gradients
        from interfering with the critic's representation learning.
        """
        bc_pred = self.bc_net(states.detach())  # (B, action_dim)
        bc_loss = F.mse_loss(bc_pred, actions.detach()) * self.bc_weight

        self.bc_optimizer.zero_grad()
        bc_loss.backward()
        self.bc_optimizer.step()
        return bc_loss.item()

    # ---------------------------------------------------------------------- #
    # Public interface                                                         #
    # ---------------------------------------------------------------------- #

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """
        Single BCQ training step.

        Updates (in order):
          1. V via expectile regression (detached states).
          2. Q via Bellman backup (with encoder grad).
          3. BC network via MSE on dataset actions (detached).
          4. Polyak-average Q targets.

        Returns:
            TrainMetrics with loss = v_loss + q_loss + bc_loss and per-component extras.
        """
        states, actions, next_states, rewards, dones = self._encode_batch(batch)

        v_loss = self._update_v(states.detach(), actions.detach())
        q_loss = self._update_q(states, actions, rewards, next_states, dones)
        bc_loss = self._update_bc(states, actions)

        self._soft_update_target_pair(self.q1, self.q1_target, self.target_update_rate)
        self._soft_update_target_pair(self.q2, self.q2_target, self.target_update_rate)

        return TrainMetrics(
            loss=v_loss + q_loss + bc_loss,
            extra={
                "v_loss": v_loss,
                "q_loss": q_loss,
                "bc_loss": bc_loss,
            },
        )

    def get_action_values(
        self, states: list[str], actions: list[str]
    ) -> torch.Tensor:
        """
        Compute batch-constrained Q-values: min(Q1, Q2)(s, a).

        Note: these Q-values are most meaningful for actions near the
        behavior policy.  For OOD actions the estimates may be unreliable.

        Returns:
            (N,) tensor of Q-values.
        """
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        with torch.no_grad():
            s_enc = self.state_encoder(s_tokens)
            a_enc = self.action_encoder(a_tokens)
            q1 = self.q1(s_enc, a_enc)
            q2 = self.q2(s_enc, a_enc)
            return torch.min(q1, q2).squeeze(-1)

    def get_advantages(
        self, states: list[str], actions: list[str]
    ) -> torch.Tensor:
        """
        Compute A(s, a) = min(Q1, Q2)(s, a) − V(s).

        Registered in compute_weights.py _HAS_ADVANTAGES for direct
        advantage-weighted policy extraction without a separate actor.

        Returns:
            (N,) tensor of advantages.
        """
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        with torch.no_grad():
            s_enc = self.state_encoder(s_tokens)
            a_enc = self.action_encoder(a_tokens)
            q_min = torch.min(self.q1(s_enc, a_enc), self.q2(s_enc, a_enc))
            v = self.v(s_enc)
            return (q_min - v).squeeze(-1)

    def get_constrained_action(self, states: list[str]) -> torch.Tensor:
        """
        Return the BC-predicted action embedding for each state.

        This is the BCQ "safe" action: the most data-consistent action
        embedding the behavior network has learned.  Useful for deployment
        when a constrained action embedding is needed.

        Returns:
            (N, action_dim) tensor.
        """
        s_tokens = self._tokenize(states)
        with torch.no_grad():
            s_enc = self.state_encoder(s_tokens)
            return self.bc_net(s_enc)

    # ---------------------------------------------------------------------- #
    # Persistence                                                              #
    # ---------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Serialize all network state_dicts."""
        torch.save(
            {
                "state_encoder": self.state_encoder.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
                "v": self.v.state_dict(),
                "bc_net": self.bc_net.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Restore from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        self.v.load_state_dict(ckpt["v"])
        self.bc_net.load_state_dict(ckpt["bc_net"])
