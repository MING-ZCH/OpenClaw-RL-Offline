"""
Digi-Q: Offline TD Q-Learning with Best-of-N Policy Extraction.

Reference:
    Bai et al., "Digi-Q: Learning Q-Value Functions for Training
    Device-Control Agents", ICLR 2025, arXiv:2502.15760

Key ideas (adapted to lightweight text-encoder setting):
    1. **Representation fine-tuning** (Stage I): Train the state-action
       encoder via binary cross-entropy to predict whether a (s, a)
       transition is *actionable* (leads to meaningful state change).
       In device control, this is measured by pixel change; here we
       proxy it with reward > 0.  The encoder is then frozen for
       subsequent TD-learning stages.
    2. **Offline TD Q-function** (Stage II): Train Q_θ(s, a) and V_ψ(s)
       on top of the frozen encoder using standard TD(0) objectives
       with target networks and soft updates:
         J_Q = E[(Q(s,a) - r - γ V_target(s'))²]     (Eq. 4)
         J_V = E[(V(s) - Q_target(s,a))²]             (Eq. 5)
    3. **Best-of-N policy extraction** (Stage III): For each state,
       sample N candidate actions from the replay buffer, rank them
       by Q(s, a_i), and train the policy to imitate the top-ranked
       action that also has positive advantage Q(s,a*) - V(s) > 0:
         J_π = E[δ(a*) · log π(a*|s)]                 (Eq. 6)
       This is more aggressive than AWR (used by DigiRL) and avoids
       the instability of REINFORCE's negative gradient.

Compared to DigiRL (Bai et al. 2024):
    - DigiRL uses BCE value functions + doubly-robust advantage + hard-filter AWR.
    - Digi-Q uses explicit state-action Q-function via TD-learning + Best-of-N
      policy extraction.  The Q-function enables scoring *multiple* candidate
      actions without environment interaction, yielding better sample efficiency.
    - Digi-Q achieves 21.2% improvement over DigiRL on Android-in-the-Wild.

Implementation notes:
    - Inherits BaseOfflineAlgorithm; builds own encoders, Q/V/policy networks.
    - Q-function: MLP on concatenated (state_enc, action_enc) features.
    - V-function: MLP on state_enc features only.
    - Best-of-N: at each train step, for every state in the mini-batch we
      sample N actions from the buffer and score them with Q(s, a_i).
    - PolicyNetwork reused from off_policy_grpo.
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


# ---------------------------------------------------------------------------
# Network components
# ---------------------------------------------------------------------------


class QFunctionNet(nn.Module):
    """
    Q(s, a) -> scalar.  Takes concatenated (state, action) embeddings.

    Trained via MSE TD-learning against r + γ V_target(s').
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Returns Q-value scalar (B, 1)."""
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class VFunctionNet(nn.Module):
    """
    V(s) -> scalar.  State-only value function for training stability.

    Trained via MSE against Q_target(s, a) from the dataset action.
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
        """Returns V-value scalar (B, 1)."""
        return self.net(state)


class RepresentationHead(nn.Module):
    """
    Binary classifier for representation fine-tuning (Stage I).

    Predicts whether a (state, action) pair is *actionable* —
    i.e., whether executing this action leads to meaningful change.
    Trained via BCE; in device control this corresponds to detecting
    significant pixel change between consecutive screenshots.

    In our text-based setting we proxy "actionable" with reward > 0.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Returns logit for actionability (B, 1)."""
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Digi-Q algorithm
# ---------------------------------------------------------------------------


class DigiQ(BaseOfflineAlgorithm):
    """
    Digi-Q: offline TD Q-learning with Best-of-N policy extraction.

    Parameters:
        best_of_n: Number of candidate actions sampled per state for
            Best-of-N policy extraction (paper default: 16).
        tau_target: Soft update rate for target Q/V networks.
        actor_lr: Learning rate for the policy network.
        max_grad_norm: Gradient clipping threshold.
        n_critic_updates: Q/V gradient steps per train_step call.
        n_actor_updates: Actor gradient steps per train_step call.
        n_repr_updates: Representation fine-tuning steps per train_step.
        repr_frozen: If True, skip Stage I (encoder already frozen).
            Automatically set to True after the first call to train_step
            that completes representation fine-tuning.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        state_dim: int = 256,
        action_dim: int = 256,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        best_of_n: int = 16,
        tau_target: float = 0.005,
        actor_lr: Optional[float] = None,
        max_grad_norm: float = 1.0,
        n_critic_updates: int = 1,
        n_actor_updates: int = 1,
        n_repr_updates: int = 1,
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
        self.best_of_n = best_of_n
        self.tau_target = tau_target
        self.n_critic_updates = n_critic_updates
        self.n_actor_updates = n_actor_updates
        self.n_repr_updates = n_repr_updates
        actor_lr = actor_lr or lr
        self._repr_frozen = False

        # ---- Encoders ----
        self.state_encoder = StateEncoder(
            vocab_size=self._vocab_size, hidden_dim=state_dim
        ).to(self.device)
        self.action_encoder = ActionEncoder(
            vocab_size=self._vocab_size, hidden_dim=action_dim
        ).to(self.device)

        # ---- Representation fine-tuning head (Stage I) ----
        self.repr_head = RepresentationHead(
            state_dim, action_dim, hidden_dim
        ).to(self.device)

        # ---- Q-function and V-function (Stage II) ----
        self.q_net = QFunctionNet(state_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.v_net = VFunctionNet(state_dim, hidden_dim).to(self.device)

        # Target networks
        self.q_target = QFunctionNet(state_dim, action_dim, hidden_dim).to(
            self.device
        )
        self.v_target = VFunctionNet(state_dim, hidden_dim).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.v_target.load_state_dict(self.v_net.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad = False
        for p in self.v_target.parameters():
            p.requires_grad = False

        # ---- Policy (Stage III) ----
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(
            self.device
        )

        # ---- Optimizers ----
        encoder_params = list(self.state_encoder.parameters()) + list(
            self.action_encoder.parameters()
        )
        self.encoder_optimizer = torch.optim.Adam(encoder_params, lr=lr)
        self.repr_optimizer = torch.optim.Adam(
            self.repr_head.parameters(), lr=lr
        )
        self.critic_optimizer = torch.optim.Adam(
            list(self.q_net.parameters()) + list(self.v_net.parameters()),
            lr=lr,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=actor_lr
        )

    # ------------------------------------------------------------------
    # Soft target updates
    # ------------------------------------------------------------------

    def _soft_update_targets(self) -> None:
        """Polyak-average update of target Q and V networks."""
        tau = self.tau_target
        for p, p_target in zip(
            self.q_net.parameters(), self.q_target.parameters()
        ):
            p_target.data.mul_(1 - tau).add_(p.data, alpha=tau)
        for p, p_target in zip(
            self.v_net.parameters(), self.v_target.parameters()
        ):
            p_target.data.mul_(1 - tau).add_(p.data, alpha=tau)

    # ------------------------------------------------------------------
    # Stage I: Representation fine-tuning (BCE on actionability)
    # ------------------------------------------------------------------

    def _update_representation(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
    ) -> dict:
        """
        Train encoders + repr_head via BCE on actionability label.

        Label: 1 if reward > 0 (actionable transition), 0 otherwise.
        In the paper this is y_t = 1 if d(s_t, s_{t+1}) > epsilon (pixel change);
        we proxy this with reward > 0 since positive reward indicates meaningful
        state progress.
        """
        targets = (rewards > 0).float().unsqueeze(-1)  # (B, 1)
        total_loss = 0.0

        for _ in range(self.n_repr_updates):
            logits = self.repr_head(states, actions)  # (B, 1)
            loss = F.binary_cross_entropy_with_logits(logits, targets)

            # Backprop through encoder + repr_head
            self.encoder_optimizer.zero_grad()
            self.repr_optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    list(self.state_encoder.parameters())
                    + list(self.action_encoder.parameters())
                    + list(self.repr_head.parameters()),
                    self.max_grad_norm,
                )
            self.encoder_optimizer.step()
            self.repr_optimizer.step()
            total_loss += loss.item()

        return {"repr_bce_loss": total_loss / max(self.n_repr_updates, 1)}

    # ------------------------------------------------------------------
    # Stage II: TD-learning for Q and V
    # ------------------------------------------------------------------

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        TD(0) update for Q(s,a) and V(s).

        J_Q = E[(Q(s,a) - (r + γ V_target(s')))²]     (Eq. 4)
        J_V = E[(V(s) - Q_target(s,a))²]               (Eq. 5)
        """
        total_q_loss = 0.0
        total_v_loss = 0.0

        for _ in range(self.n_critic_updates):
            # Detach encoder outputs (frozen after Stage I in the paper;
            # here we call .detach() to prevent critic gradients flowing
            # back into the encoder when repr is already frozen).
            s_det = states.detach()
            a_det = actions.detach()
            ns_det = next_states.detach()

            # Q-function TD target: r + γ V_target(s')
            with torch.no_grad():
                v_next = self.v_target(ns_det).squeeze(-1)  # (B,)
                q_td_target = rewards + self.gamma * (1.0 - dones) * v_next

            q_pred = self.q_net(s_det, a_det).squeeze(-1)  # (B,)
            q_loss = F.mse_loss(q_pred, q_td_target)

            # V-function target: Q_target(s, a)  (using dataset action)
            with torch.no_grad():
                q_target_val = self.q_target(s_det, a_det).squeeze(-1)  # (B,)

            v_pred = self.v_net(s_det).squeeze(-1)  # (B,)
            v_loss = F.mse_loss(v_pred, q_target_val)

            critic_loss = q_loss + v_loss

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    list(self.q_net.parameters())
                    + list(self.v_net.parameters()),
                    self.max_grad_norm,
                )
            self.critic_optimizer.step()

            # Soft update targets
            self._soft_update_targets()

            total_q_loss += q_loss.item()
            total_v_loss += v_loss.item()

        n = max(self.n_critic_updates, 1)
        return {
            "q_td_loss": total_q_loss / n,
            "v_td_loss": total_v_loss / n,
        }

    # ------------------------------------------------------------------
    # Stage III: Best-of-N policy extraction
    # ------------------------------------------------------------------

    def _update_actor_best_of_n(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        batch: TransitionBatch,
    ) -> dict:
        """
        Best-of-N policy extraction (Eq. 6).

        For each state s in the mini-batch, we gather N candidate actions:
            - The original action from the buffer (always included)
            - (N-1) actions sampled uniformly from the buffer

        Rank by Q(s, a_i), pick the best, and only train if advantage > 0.
        """
        B = states.size(0)
        N = self.best_of_n
        total_actor_loss = 0.0
        n_kept = 0

        # Gather N candidate actions for each state
        # Strategy: sample (N-1) random actions from the batch + original action
        all_action_texts = list(batch.actions)

        for _ in range(self.n_actor_updates):
            with torch.no_grad():
                # V(s) for advantage computation
                v_s = self.v_net(states.detach()).squeeze(-1)  # (B,)

                # Build candidate action embeddings: (B, N, action_dim)
                # First candidate = the dataset action
                candidate_q = torch.zeros(B, N, device=self.device)
                candidate_actions = torch.zeros(
                    B, N, self.action_dim, device=self.device
                )

                # Encode original actions
                candidate_actions[:, 0, :] = actions.detach()
                candidate_q[:, 0] = self.q_net(
                    states.detach(), actions.detach()
                ).squeeze(-1)

                # Sample (N-1) additional actions from the buffer
                for j in range(1, N):
                    # Random indices into the batch
                    rand_idx = torch.randint(0, B, (B,))
                    sampled_a = actions[rand_idx].detach()
                    candidate_actions[:, j, :] = sampled_a
                    candidate_q[:, j] = self.q_net(
                        states.detach(), sampled_a
                    ).squeeze(-1)

                # Best-of-N: select argmax Q for each state
                best_idx = candidate_q.argmax(dim=1)  # (B,)
                best_q = candidate_q.gather(
                    1, best_idx.unsqueeze(1)
                ).squeeze(1)  # (B,)

                # Advantage filter: only train if Q(s, a*) - V(s) > 0
                advantage = best_q - v_s  # (B,)
                mask = (advantage > 0).float()  # (B,)

                # Get the best action embeddings
                best_actions = candidate_actions.gather(
                    1,
                    best_idx.unsqueeze(1)
                    .unsqueeze(2)
                    .expand(-1, -1, self.action_dim),
                ).squeeze(1)  # (B, action_dim)

            if mask.sum() < 1.0:
                continue

            # Train policy to imitate the best action
            log_pi = self.policy(
                states.detach(), best_actions.detach()
            )  # (B,)
            actor_loss = -(mask * log_pi).sum() / mask.sum()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
            self.actor_optimizer.step()

            total_actor_loss += actor_loss.item()
            n_kept += int(mask.sum().item())

        avg_loss = total_actor_loss / max(self.n_actor_updates, 1)
        return {
            "actor_loss": avg_loss,
            "actor_n_kept": float(n_kept) / max(self.n_actor_updates, 1),
        }

    # ------------------------------------------------------------------
    # train_step
    # ------------------------------------------------------------------

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """
        Digi-Q three-stage train step.

        1. Encode text batch
        2. (If not frozen) Stage I: Representation fine-tuning
        3. Stage II: TD-learning for Q and V critics
        4. Stage III: Best-of-N policy extraction
        """
        states, actions, next_states, rewards, dones = self._encode_batch(
            batch
        )

        # Stage I: Representation fine-tuning
        repr_metrics: dict = {}
        if not self._repr_frozen:
            repr_metrics = self._update_representation(
                states, actions, rewards
            )
            # After first pass, freeze encoder for subsequent TD-learning
            # (in practice the paper runs Stage I for several epochs first;
            # here we interleave but always detach encoder gradients in critic)
            self._repr_frozen = True

        # Stage II: Critic TD-learning
        critic_metrics = self._update_critic(
            states, actions, next_states, rewards, dones
        )

        # Stage III: Best-of-N policy extraction
        actor_metrics = self._update_actor_best_of_n(
            states, actions, batch
        )

        total_loss = (
            repr_metrics.get("repr_bce_loss", 0.0)
            + critic_metrics.get("q_td_loss", 0.0)
            + critic_metrics.get("v_td_loss", 0.0)
            + actor_metrics.get("actor_loss", 0.0)
        )

        extra = {}
        extra.update(repr_metrics)
        extra.update(critic_metrics)
        extra.update(actor_metrics)

        return TrainMetrics(loss=total_loss, extra=extra)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_action_values(
        self, states: list, actions: list
    ) -> torch.Tensor:
        """
        Return Q(s, a) for the given (state, action) pairs.

        Unlike DigiRL which returns V_step(s), Digi-Q returns the full
        state-action Q value, which can be used to rank among multiple
        candidate actions.
        """
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        with torch.no_grad():
            s_enc = self.state_encoder(s_tokens)
            a_enc = self.action_encoder(a_tokens)
            return self.q_net(s_enc, a_enc).squeeze(-1)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        torch.save(
            {
                "state_encoder": self.state_encoder.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "repr_head": self.repr_head.state_dict(),
                "q_net": self.q_net.state_dict(),
                "v_net": self.v_net.state_dict(),
                "q_target": self.q_target.state_dict(),
                "v_target": self.v_target.state_dict(),
                "policy": self.policy.state_dict(),
                "repr_frozen": self._repr_frozen,
            },
            path,
        )

    def load(self, path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.repr_head.load_state_dict(ckpt["repr_head"])
        self.q_net.load_state_dict(ckpt["q_net"])
        self.v_net.load_state_dict(ckpt["v_net"])
        self.q_target.load_state_dict(ckpt["q_target"])
        self.v_target.load_state_dict(ckpt["v_target"])
        self.policy.load_state_dict(ckpt["policy"])
        self._repr_frozen = ckpt.get("repr_frozen", True)
