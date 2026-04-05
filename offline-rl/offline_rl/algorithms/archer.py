"""
ArCHer: Actor-Critic with Hierarchical Reinforcement Learning for language agents.

Reference:
    Zhou et al., "ArCHer: Training Language Model Agents via Hierarchical
    Multi-Turn RL", ICML 2024 (arXiv:2402.19446)

Key idea:
    ArCHer treats multi-turn dialogue as a two-level MDP:
    - High level (utterance/turn): IQL twin-Q + V critic with expectile regression.
      The critic operates on full conversation context (state) and the generated
      utterance (action), learning long-horizon values over multi-turn episodes.
    - Low level (token): AWR (Advantage-Weighted Regression) policy extraction.
      The policy is trained by re-weighting log-probabilities with exp(beta * A),
      where A = min(Q1,Q2)(s,a) - V(s) is the advantage estimate from the critic.

Compared to pure IQL in this library:
    - ArCHer adds an INLINE AWR actor update inside train_step().
    - Default tau=0.9 (higher pessimism) vs IQL's tau=0.7.
    - Three separate optimizer groups: encoder, critic (Q+V), actor.
    - Actor gradients are isolated: encoders are detached before actor forward
      pass so critic BP does not contaminate actor parameter updates.

Architecture in this discrete-embedding codebase:
    - state_encoder / action_encoder: TextEncoder(vocab_size, hidden_dim=state_dim)
    - q1, q2, q1_target, q2_target: QNetwork(state_dim, action_dim, hidden_dim)
    - v, v_target: VNetwork(state_dim, hidden_dim)
    - policy: PolicyNetwork(state_dim, action_dim, hidden_dim)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch
from .base import BaseOfflineAlgorithm, StateEncoder, ActionEncoder, TextEncoder, TrainMetrics
from .iql import QNetwork, VNetwork
from .off_policy_grpo import PolicyNetwork

logger = logging.getLogger(__name__)


class ArCHer(BaseOfflineAlgorithm):
    """
    ArCHer: Actor-Critic with Hierarchical RL for LLM-based agents.

    The algorithm combines IQL-style critic learning with AWR-style policy
    extraction in a single training loop, making it suitable for fine-tuning
    language model agents on offline multi-turn interaction datasets.

    Training loop per step:
        1. Encode batch into (s, a, s', r, d) tensors.
        2. V update via expectile regression:
               L_V = E_tau[|tau - 1(Q_tgt - V < 0)| * (Q_tgt - V)^2]
           where Q_tgt = min(Q1_tgt, Q2_tgt).
        3. Q update via Bellman backup:
               L_Q = MSE(Q1(s,a), r + gamma*(1-d)*V(s'))
                   + MSE(Q2(s,a), r + gamma*(1-d)*V(s'))
        4. Actor AWR update:
               A = min(Q1,Q2)(s,a) - V(s)        [detached]
               w = clamp(exp(beta * A), max=exp_clamp)
               L_actor = -mean(w * log_prob(a|s))
        5. Polyak-average Q targets.

    Parameters:
        replay_buffer: Pre-collected transition dataset.
        state_dim: Dimensionality of state embeddings (default 256).
        action_dim: Dimensionality of action embeddings (default 256).
        hidden_dim: Width of all MLP hidden layers (default 256).
        lr: Learning rate for encoder and critic optimizers (default 3e-4).
        gamma: Discount factor (default 0.99).
        tau: Expectile parameter for V update. tau=0.9 biases V toward
             the upper 90% quantile of Q, increasing pessimism (default 0.9).
        beta: AWR temperature. Higher beta makes policy more greedy w.r.t.
              advantage estimates (default 3.0).
        target_update_rate: Polyak coefficient for target network updates
                            (default 0.005).
        actor_lr: Learning rate for policy optimizer (default 3e-4).
        device: Torch device string (default "cpu").
        exp_clamp: Upper bound on exp(beta * A) to prevent gradient explosion
                   (default 10.0).
        **kwargs: Forwarded to BaseOfflineAlgorithm (e.g. max_token_len).
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        state_dim: int = 256,
        action_dim: int = 256,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.9,
        beta: float = 3.0,
        target_update_rate: float = 0.005,
        actor_lr: Optional[float] = 3e-4,
        device: str = "cpu",
        exp_clamp: float = 10.0,
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
        self.target_update_rate = target_update_rate
        self.actor_lr = actor_lr if actor_lr is not None else lr
        self.exp_clamp = exp_clamp

        # ------------------------------------------------------------------ #
        # Encoders (shared between critic and actor forward passes)            #
        # ------------------------------------------------------------------ #
        self.state_encoder = StateEncoder(
            vocab_size=self._vocab_size, hidden_dim=state_dim
        ).to(self.device)
        self.action_encoder = ActionEncoder(
            vocab_size=self._vocab_size, hidden_dim=action_dim
        ).to(self.device)

        # ------------------------------------------------------------------ #
        # Critic networks — twin-Q + V (IQL-style)                            #
        # ------------------------------------------------------------------ #
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        # Initialise targets identically to the online networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.v = VNetwork(state_dim, hidden_dim).to(self.device)

        # ------------------------------------------------------------------ #
        # Actor network — policy π(a|s) → log-prob scalar                    #
        # ------------------------------------------------------------------ #
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # ------------------------------------------------------------------ #
        # Optimizers                                                           #
        # Note: Use torch.optim.Adam (NOT AdamW) for PyTorch 1.13.1 compat.  #
        # Three separate groups to allow fine-grained LR control and to       #
        # keep actor updates isolated from critic/encoder gradients.           #
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

        # Actor optimizer only covers the policy network — NOT the encoders.
        # This enforces the hierarchy: the actor receives encoder outputs as
        # fixed features during its own backward pass.
        self.actor_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.actor_lr
        )

    # ---------------------------------------------------------------------- #
    # Private helpers                                                          #
    # ---------------------------------------------------------------------- #

    def _expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        """
        Asymmetric L2 loss for expectile regression.

        L_tau(u) = |tau - 1(u < 0)| * u^2

        When tau > 0.5, the loss penalises *under-estimates* more heavily,
        pushing V(s) toward the upper quantile of Q(s,a) samples.  For
        ArCHer we use tau=0.9 (vs IQL's default 0.7) to be more pessimistic
        about out-of-support value overestimation.
        """
        weight = torch.where(diff > 0, self.tau, 1.0 - self.tau)
        return (weight * diff.pow(2)).mean()

    def _update_v(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> float:
        """
        Update V(s) via expectile regression on min(Q1_tgt, Q2_tgt)(s,a).

        States and actions must be detached from the encoder graph so that
        this V update does not propagate gradients through the encoders
        (encoder gradients are managed exclusively by the Q update).
        """
        with torch.no_grad():
            q1_val = self.q1_target(states, actions)
            q2_val = self.q2_target(states, actions)
            q_min = torch.min(q1_val, q2_val)  # (batch, 1)

        v_pred = self.v(states)  # (batch, 1)
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
        """
        Update Q1 and Q2 via Bellman backup using V(s') as the bootstrap target.

        Q_target = r + gamma * (1 - d) * V(s')
        L_Q = MSE(Q1(s,a), Q_target) + MSE(Q2(s,a), Q_target)

        The encoder_optimizer is stepped here so that the shared encoders
        receive gradients via the Bellman loss path (the primary signal for
        learning useful state/action representations).
        """
        with torch.no_grad():
            v_next = self.v(next_states)  # (batch, 1)
            q_target = (
                rewards.unsqueeze(-1)
                + self.gamma * (1.0 - dones.unsqueeze(-1)) * v_next
            )  # (batch, 1)

        q1_pred = self.q1(states, actions)  # (batch, 1)
        q2_pred = self.q2(states, actions)  # (batch, 1)
        q_loss = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)

        self.critic_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()
        self.encoder_optimizer.step()

        return q_loss.item()

    def _update_actor(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> float:
        """
        Update policy via Advantage-Weighted Regression (AWR).

        The actor receives *detached* encoder outputs so that its backward
        pass does not flow gradients through the shared encoders.  This
        implements the hierarchical separation described in the ArCHer paper:
        the high-level critic and low-level actor are updated independently.

        w = clamp(exp(beta * A), 0, exp_clamp)   where A = Q_min(s,a) - V(s)
        L_actor = -mean(w * log_prob_policy(a|s))
        """
        # Detach encoder outputs: actor update must NOT touch encoder params
        s_det = states.detach()
        a_det = actions.detach()

        with torch.no_grad():
            q1_val = self.q1(s_det, a_det)   # (batch, 1)
            q2_val = self.q2(s_det, a_det)   # (batch, 1)
            q_min = torch.min(q1_val, q2_val)
            v_val = self.v(s_det)             # (batch, 1)
            advantage = (q_min - v_val).squeeze(-1)  # (batch,)

        # Clamp exponential weight to prevent log-scale gradient explosions
        w = torch.exp(self.beta * advantage).clamp(max=self.exp_clamp)  # (batch,)

        # log-prob of the policy for (s, a) pairs
        log_prob = self.policy(s_det, a_det)  # (batch,) — see PolicyNetwork.forward

        actor_loss = -(w * log_prob).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def _soft_update_targets(self) -> None:
        """Polyak-average Q-network targets after each training step."""
        self._soft_update_target_pair(self.q1, self.q1_target, self.target_update_rate)
        self._soft_update_target_pair(self.q2, self.q2_target, self.target_update_rate)

    # ---------------------------------------------------------------------- #
    # Public interface (BaseOfflineAlgorithm contract)                        #
    # ---------------------------------------------------------------------- #

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """
        Single ArCHer training step.

        Executes critic (V + Q) and actor (AWR) updates in sequence on the
        same encoded batch.  Returns a TrainMetrics object containing the
        aggregate loss and per-component breakdowns.

        Args:
            batch: A TransitionBatch sampled from the replay buffer.

        Returns:
            TrainMetrics with loss = v_loss + q_loss + actor_loss and
            extra = {v_loss, q_loss, actor_loss, mean_advantage}.
        """
        # 1. Encode raw text to dense vectors
        states, actions, next_states, rewards, dones = self._encode_batch(batch)

        # 2. V update: expectile regression on detached (s, a)
        v_loss = self._update_v(states.detach(), actions.detach())

        # 3. Q update: Bellman backup (also updates encoders)
        q_loss = self._update_q(states, actions, rewards, next_states, dones)

        # 4. Actor update: AWR with detached encoder outputs
        actor_loss = self._update_actor(states, actions)

        # 5. Polyak update Q targets
        self._soft_update_targets()

        # Compute mean advantage for monitoring (no grad)
        with torch.no_grad():
            s_det = states.detach()
            a_det = actions.detach()
            q1_val = self.q1(s_det, a_det)
            q2_val = self.q2(s_det, a_det)
            v_val = self.v(s_det)
            mean_adv = (torch.min(q1_val, q2_val) - v_val).mean().item()

        total_loss = v_loss + q_loss + actor_loss
        return TrainMetrics(
            loss=total_loss,
            extra={
                "v_loss": v_loss,
                "q_loss": q_loss,
                "actor_loss": actor_loss,
                "mean_advantage": mean_adv,
            },
        )

    def get_action_values(
        self, states: list[str], actions: list[str]
    ) -> torch.Tensor:
        """
        Compute min(Q1, Q2)(s, a) for a list of (state, action) text pairs.

        Returns:
            (N,) tensor of Q-values on CPU.
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
        Compute A(s, a) = min(Q1, Q2)(s, a) - V(s).

        Used by compute_weights.py (_HAS_ADVANTAGES registry) to derive
        per-sample AWR weights for LLM fine-tuning without a separate
        policy network (pure IQL-style advantage extraction).

        Returns:
            (N,) tensor of advantages on self.device.
        """
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        with torch.no_grad():
            s_enc = self.state_encoder(s_tokens)
            a_enc = self.action_encoder(a_tokens)
            q1 = self.q1(s_enc, a_enc)
            q2 = self.q2(s_enc, a_enc)
            q_min = torch.min(q1, q2)
            v = self.v(s_enc)
            return (q_min - v).squeeze(-1)

    def get_policy_log_probs(
        self, states: list[str], actions: list[str]
    ) -> torch.Tensor:
        """
        Query the trained policy network for log-probabilities.

        Convenience method for evaluating the actor independently from the
        critic.  Not required by the base interface but useful for logging.

        Returns:
            (N,) tensor of log-probabilities.
        """
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        with torch.no_grad():
            s_enc = self.state_encoder(s_tokens)
            a_enc = self.action_encoder(a_tokens)
            return self.policy(s_enc, a_enc)

    # ---------------------------------------------------------------------- #
    # Persistence                                                              #
    # ---------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Serialize all network state_dicts to a single checkpoint file."""
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
        """Restore all network parameters from a checkpoint file."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        self.v.load_state_dict(ckpt["v"])
        self.policy.load_state_dict(ckpt["policy"])
