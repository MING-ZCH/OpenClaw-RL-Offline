"""
Implicit Language Q-Learning (ILQL) for LLM-Based Agents.

Reference:
    Snell, Kostrikov, Su, Yang, Levine, "Offline RL for Natural Language
    Generation with Implicit Language Q Learning", ICLR 2023
    (arXiv:2206.11871)

Key idea:
    ILQL extends standard IQL to the language model setting by adding:

    1. *Conservative Q penalty (CQL-style)*: Regularizes Q-values to stay
       calibrated for in-distribution actions, preventing overestimation
       on out-of-distribution state-action pairs.

           L_CQL = E_data[ Q(s,a) ] - E_data[ log sum_a' exp(Q(s,a')/τ) ]

       In practice, this is implemented as a cross-entropy loss between the
       Q-head output (across all candidate actions per state) and the actual
       chosen action, scaled by temperature τ.

    2. *Advantage-weighted behavioral cloning (AWAC-style)*: Extracts the
       optimal policy by re-weighting the behavioral policy with advantages:

           L_policy = -E_data[ w(s,a) · log π(a|s) ]

       where  w(s,a) = exp( β · (Q(s,a) - V(s)) )    (if exp_weights=True)
         or   w(s,a) = β · 1[A > 0] + (1-β) · 1[A <= 0] (sign weighting)

    3. *Value function with expectile regression*: Same as IQL —

           L_V = E_τ[ (Q_target(s,a) - V(s))^2 ]

       where τ-expectile biases V toward upper quantiles of Q.

    4. *Twin Q Bellman backup*:

           L_Q = E[ (r + γ(1-d)V(s') - Q(s,a))^2 ]    (for both Q1, Q2)

    The total loss is:

        L = λ_awac · L_policy + λ_v · L_V + λ_q · L_Q + λ_cql · L_CQL

Offline adaptation:
    In our lightweight text-encoder framework without a real LM backbone,
    the "policy" network does advantage-weighted BC on (state, action) pairs.
    CQL regularization acts on the learned Q-function at the action level
    (not per-token), matching the offline agent RL setting.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch
from .base import BaseOfflineAlgorithm, StateEncoder, ActionEncoder, TrainMetrics
from .iql import QNetwork, VNetwork
from .off_policy_grpo import PolicyNetwork

logger = logging.getLogger(__name__)


class ILQL(BaseOfflineAlgorithm):
    """
    Implicit Language Q-Learning (ILQL).

    Extends IQL with:
        - CQL-style conservative Q penalty to keep Q calibrated.
        - Advantage-weighted behavioral cloning for policy extraction.
        - Optional exponential or sign-based advantage weighting.

    Parameters:
        replay_buffer: Pre-collected transition dataset.
        tau: Expectile parameter for V update (>0.5 biases toward upper Q).
        beta: Temperature for advantage-weighted policy extraction.
        cql_alpha: Weight of the CQL conservative penalty.
        cql_temp: Temperature for CQL log-sum-exp softmax regularizer.
        awac_weight: Weight of the advantage-weighted BC policy loss.
        v_loss_weight: Weight of the V expectile loss.
        q_loss_weight: Weight of the Q Bellman loss.
        exp_weights: If True, use exp(β·A) weighting; else sign-based.
        target_update_rate: Polyak averaging rate for target Q networks.
        n_cql_actions: Number of random actions to sample for CQL penalty
                       estimation per state (default 10).
        device: Torch device string (default "cpu").
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
        beta: float = 3.0,
        cql_alpha: float = 1.0,
        cql_temp: float = 1.0,
        awac_weight: float = 1.0,
        v_loss_weight: float = 1.0,
        q_loss_weight: float = 1.0,
        exp_weights: bool = True,
        target_update_rate: float = 0.005,
        n_cql_actions: int = 10,
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
        self.beta = beta
        self.cql_alpha = cql_alpha
        self.cql_temp = cql_temp
        self.awac_weight = awac_weight
        self.v_loss_weight = v_loss_weight
        self.q_loss_weight = q_loss_weight
        self.exp_weights = exp_weights
        self.target_update_rate = target_update_rate
        self.n_cql_actions = n_cql_actions

        # ----- Encoders -----
        self.state_encoder = StateEncoder(
            vocab_size=self._vocab_size, hidden_dim=state_dim
        ).to(self.device)
        self.action_encoder = ActionEncoder(
            vocab_size=self._vocab_size, hidden_dim=action_dim
        ).to(self.device)

        # ----- Twin Q + targets -----
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # ----- V function -----
        self.v = VNetwork(state_dim, hidden_dim).to(self.device)

        # ----- Policy for AWAC-style extraction -----
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # ----- Optimizers -----
        encoder_params = (
            list(self.state_encoder.parameters())
            + list(self.action_encoder.parameters())
        )
        self.encoder_optimizer = torch.optim.Adam(encoder_params, lr=lr)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.v_optimizer = torch.optim.Adam(self.v.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr
        )

    # ------------------------------------------------------------------ #
    # Loss components                                                      #
    # ------------------------------------------------------------------ #

    def _expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        """Asymmetric squared loss for expectile regression (Eq. in IQL)."""
        weight = torch.where(diff > 0, self.tau, 1.0 - self.tau)
        return (weight * diff.pow(2)).mean()

    def _update_v(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> float:
        """Update V via expectile regression on min(Q1_target, Q2_target)."""
        with torch.no_grad():
            q1_val = self.q1_target(states, actions)
            q2_val = self.q2_target(states, actions)
            q_val = torch.min(q1_val, q2_val)

        v_val = self.v(states)
        v_loss = self._expectile_loss(q_val - v_val)

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return v_loss.item()

    def _update_q(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        """Bellman backup: Q(s,a) → r + γ(1-d)V(s')."""
        with torch.no_grad():
            target = (
                rewards.unsqueeze(-1)
                + self.gamma * (1 - dones.unsqueeze(-1)) * self.v(next_states)
            )

        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)
        q_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)
        return q_loss

    def _cql_penalty(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Conservative Q-Learning regularizer (CQL).

        Penalizes Q-values by encouraging the Q-function on in-distribution
        (data) actions to be lower than on randomly sampled actions.

        L_CQL ≈ E_s[ log Σ_a' exp(Q(s,a')/τ) ] - E_{s,a~data}[ Q(s,a)/τ ]

        We approximate the log-sum-exp over actions by sampling n_cql_actions
        random action vectors per state.
        """
        B, D_a = actions.shape
        D_s = states.shape[-1]

        # Sample random actions (uniform over the action embedding space)
        rand_actions = torch.randn(
            B, self.n_cql_actions, D_a, device=self.device
        )

        # Expand states for broadcasting  →  (B, n_cql, D_s)
        states_exp = states.unsqueeze(1).expand(-1, self.n_cql_actions, -1)
        states_flat = states_exp.reshape(B * self.n_cql_actions, D_s)
        actions_flat = rand_actions.reshape(B * self.n_cql_actions, D_a)

        # Q-values for random actions
        q1_rand = self.q1(states_flat, actions_flat).view(B, self.n_cql_actions)
        q2_rand = self.q2(states_flat, actions_flat).view(B, self.n_cql_actions)

        # log-sum-exp over random actions
        logsumexp_q1 = torch.logsumexp(q1_rand / self.cql_temp, dim=1)
        logsumexp_q2 = torch.logsumexp(q2_rand / self.cql_temp, dim=1)

        # Q-values for data actions
        q1_data = self.q1(states, actions).squeeze(-1)
        q2_data = self.q2(states, actions).squeeze(-1)

        cql_loss = (
            (logsumexp_q1 - q1_data / self.cql_temp).mean()
            + (logsumexp_q2 - q2_data / self.cql_temp).mean()
        )
        return cql_loss

    def _awac_policy_loss(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Advantage-weighted behavioral cloning for policy extraction.

        w(s,a) = exp(β * A(s,a))   if exp_weights
                 β * sign(A>0) + (1-β) * sign(A<=0)  otherwise

        L_policy = -E[ w(s,a) * log_π(a|s) ]
        """
        with torch.no_grad():
            q1_val = self.q1_target(states, actions)
            q2_val = self.q2_target(states, actions)
            q_val = torch.min(q1_val, q2_val).squeeze(-1)
            v_val = self.v(states).squeeze(-1)
            advantage = q_val - v_val

            if self.exp_weights:
                weights = torch.exp(self.beta * advantage)
                # Clip to prevent overflow
                weights = weights.clamp(max=100.0)
            else:
                adv_sign = (advantage > 0.0).float()
                weights = self.beta * adv_sign + (1.0 - self.beta) * (1.0 - adv_sign)

        log_pi = self.policy(states, actions)  # (B,)
        loss = -(weights * log_pi).mean()
        return loss

    def _soft_update_targets(self) -> None:
        """Polyak averaging for target Q networks."""
        self._soft_update_target_pair(self.q1, self.q1_target, self.target_update_rate)
        self._soft_update_target_pair(self.q2, self.q2_target, self.target_update_rate)

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """
        Single ILQL training step.

        Total loss =  awac_weight * L_policy
                    + v_loss_weight * L_V
                    + q_loss_weight * L_Q
                    + cql_alpha * L_CQL
        """
        states, actions, next_states, rewards, dones = self._encode_batch(batch)

        # 1. V update (expectile regression) — detached from Q/encoder
        v_loss = self._update_v(states.detach(), actions.detach())

        # 2. Q Bellman + CQL — joint update
        q_bellman = self._update_q(states, actions, rewards, next_states, dones)
        cql_loss = self._cql_penalty(states.detach(), actions.detach())

        combined_q_loss = self.q_loss_weight * q_bellman + self.cql_alpha * cql_loss

        self.q_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        combined_q_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                list(self.q1.parameters())
                + list(self.q2.parameters())
                + list(self.state_encoder.parameters())
                + list(self.action_encoder.parameters()),
                self.max_grad_norm,
            )
        self.q_optimizer.step()
        self.encoder_optimizer.step()

        # 3. AWAC policy update
        awac_loss = self._awac_policy_loss(states.detach(), actions.detach())
        policy_loss = self.awac_weight * awac_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm
            )
        self.policy_optimizer.step()

        # 4. Soft update target networks
        self._soft_update_targets()

        total_loss = v_loss + combined_q_loss.item() + policy_loss.item()
        return TrainMetrics(
            loss=total_loss,
            extra={
                "v_loss": v_loss,
                "q_bellman_loss": q_bellman.item(),
                "cql_loss": cql_loss.item(),
                "awac_loss": awac_loss.item(),
            },
        )

    def get_action_values(
        self, states: list[str], actions: list[str]
    ) -> torch.Tensor:
        """Compute advantage-reweighted Q-values for state-action pairs."""
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
        """Compute advantages A(s,a) = Q(s,a) - V(s) for decoding guidance."""
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        with torch.no_grad():
            s_enc = self.state_encoder(s_tokens)
            a_enc = self.action_encoder(a_tokens)
            q1 = self.q1(s_enc, a_enc)
            q2 = self.q2(s_enc, a_enc)
            q_min = torch.min(q1, q2).squeeze(-1)
            v = self.v(s_enc).squeeze(-1)
            return q_min - v

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_encoder": self.state_encoder.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
                "v": self.v.state_dict(),
                "policy": self.policy.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        self.v.load_state_dict(ckpt["v"])
        self.policy.load_state_dict(ckpt["policy"])
