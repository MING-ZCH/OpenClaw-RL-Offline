"""
Retrospex: Language Agent Meets Offline Reinforcement Learning Critic.

Trains a lightweight IQL-style Q-critic on offline agent trajectories.
At inference time, candidate actions are rescored by combining a frozen
LLM's log-probability with the critic's Q-value estimate:

    score(a | s) = λ_LM · log π_LLM(a | s) + λ_Q · Q_critic(s, a)

This avoids any policy gradient update: the LLM weights remain frozen,
and only the small Q/V networks are trained.  The critic acts as a
learned "advantage advisor" for beam/candidate ranking at decode time.

Design notes:
  - Follows an IQL-style offline critic: Q via Bellman, V via expectile.
  - `rescore_actions()` is the primary inference API.
  - `train_step()` runs the standard IQL critic update (no policy update).
  - `lambda_scale` can be overridden per-call in `rescore_actions()`.
  - Compatible as a drop-in augmentation for any frozen LLM agent.

References:
    Xiang et al., "Retrospex: Language Agent Meets Offline Reinforcement
    Learning Critic", EMNLP 2024 (arXiv 2505.11807).
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


class Retrospex(BaseOfflineAlgorithm):
    """
    Offline critic rescoring for frozen LLM agents (Retrospex).

    Training:
        - Q update (Bellman):   Q(s,a) ← r + γ(1-d)·V(s')
        - V update (expectile): V(s) ← expectile_τ(Q(s,a) - V(s))
        - No policy gradient; LLM weights are never touched.

    Inference:
        - Given a list of candidate action texts and their LLM log-probs,
          ``rescore_actions()`` returns a combined score for action ranking.

    Args:
        replay_buffer: Offline dataset of stored trajectories.
        tau: IQL expectile parameter for V-training (default 0.7).
        lambda_scale: Weight of Q-critic signal in combined rescoring score.
            Final score = lm_log_prob + lambda_scale * Q(s, a).
        gamma: Discount factor for Bellman targets.
        **kwargs: Forwarded to BaseOfflineAlgorithm.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        tau: float = 0.7,
        lambda_scale: float = 1.0,
        state_dim: int = 256,
        action_dim: int = 256,
        hidden_dim: int = 256,
        lr: float = 1e-4,
        gamma: float = 0.99,
        device: str = "cpu",
        max_token_len: int = 128,
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
        self.tau = tau
        self.lambda_scale = lambda_scale

        # State / action encoders
        self.state_encoder = StateEncoder(self._vocab_size, 256, state_dim).to(self.device)
        self.action_encoder = ActionEncoder(self._vocab_size, 256, action_dim).to(self.device)

        # Critic networks (IQL-style)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.v_net = VNetwork(state_dim, hidden_dim).to(self.device)

        # Target Q-networks (Polyak averages)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in list(self.q1_target.parameters()) + list(self.q2_target.parameters()):
            p.requires_grad = False

        # Single optimizer for all critic parameters
        self.optimizer = torch.optim.Adam(
            list(self.state_encoder.parameters())
            + list(self.action_encoder.parameters())
            + list(self.q1.parameters())
            + list(self.q2.parameters())
            + list(self.v_net.parameters()),
            lr=lr,
        )

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode(self, observations: List[str], actions: List[str]):
        """Encode a paired list of (obs, action) strings."""
        s = self.state_encoder(self._tokenize(observations))
        a = self.action_encoder(self._tokenize(actions))
        return s, a

    # ------------------------------------------------------------------
    # IQL critic update
    # ------------------------------------------------------------------

    def _expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        """Asymmetric L2 (expectile regression)."""
        weight = torch.where(diff > 0, self.tau, 1.0 - self.tau)
        return (weight * diff.pow(2)).mean()

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """One IQL critic update step (no policy update)."""
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)

        s, a = self._encode(batch.observation_contexts, batch.actions)
        s_next, a_dummy = self._encode(
            batch.next_observation_contexts,
            batch.actions,  # action not used for V(s')
        )

        # --- V target using frozen Q-targets ----------------------------
        with torch.no_grad():
            q1_t = self.q1_target(s, a).squeeze(-1)
            q2_t = self.q2_target(s, a).squeeze(-1)
            q_min_target = torch.min(q1_t, q2_t)

        # V loss: expectile regression
        v = self.v_net(s).squeeze(-1)
        v_loss = self._expectile_loss(q_min_target - v)

        # V(s') for Bellman target
        with torch.no_grad():
            v_next = self.v_net(s_next).squeeze(-1)
            bellman_target = rewards + self.gamma * (1.0 - dones) * v_next

        # Q loss: MSE Bellman
        q1 = self.q1(s, a).squeeze(-1)
        q2 = self.q2(s, a).squeeze(-1)
        q_loss = F.mse_loss(q1, bellman_target) + F.mse_loss(q2, bellman_target)

        total_loss = v_loss + q_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(
            list(self.state_encoder.parameters())
            + list(self.action_encoder.parameters())
            + list(self.q1.parameters())
            + list(self.q2.parameters())
            + list(self.v_net.parameters()),
            self.max_grad_norm or 1.0,
        )
        self.optimizer.step()

        # Polyak target update
        tau_polyak = 0.005
        for tgt, src in [(self.q1_target, self.q1), (self.q2_target, self.q2)]:
            for tp, sp in zip(tgt.parameters(), src.parameters()):
                tp.data.mul_(1.0 - tau_polyak).add_(tau_polyak * sp.data)

        return TrainMetrics(
            loss=total_loss.item(),
            extra={
                "q_loss": q_loss.item(),
                "v_loss": v_loss.item(),
                "v_mean": v.detach().mean().item(),
            },
        )

    # ------------------------------------------------------------------
    # Inference-time action rescoring (primary API of Retrospex)
    # ------------------------------------------------------------------

    def rescore_actions(
        self,
        observation: str,
        candidate_actions: List[str],
        lm_log_probs: Optional[List[float]] = None,
        lambda_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Rescore candidate actions at inference time.

        Combines the LLM's log-probability with the offline Q-critic:
            score(a | s) = lm_log_prob(a) + λ · Q_critic(s, a)

        If ``lm_log_probs`` is None, rescoring is purely Q-based.

        Args:
            observation: Current observation text.
            candidate_actions: List of candidate action texts.
            lm_log_probs: Log-probs from a frozen LLM (same length as actions).
            lambda_scale: Override for lambda (uses instance default if None).

        Returns:
            scores: (len(candidate_actions),) combined scores tensor.
        """
        lam = lambda_scale if lambda_scale is not None else self.lambda_scale
        obs_list = [observation] * len(candidate_actions)
        with torch.no_grad():
            s, a = self._encode(obs_list, candidate_actions)
            q1 = self.q1(s, a).squeeze(-1)
            q2 = self.q2(s, a).squeeze(-1)
            q_vals = torch.min(q1, q2)  # conservative Q (min of twin)

        if lm_log_probs is not None:
            lm_lp = torch.as_tensor(lm_log_probs, dtype=torch.float32, device=self.device)
        else:
            lm_lp = torch.zeros_like(q_vals)

        return lm_lp + lam * q_vals

    def get_action_values(self, states: List[str], actions: List[str]) -> torch.Tensor:
        """Return min Q-critic values for (state, action) pairs."""
        with torch.no_grad():
            s, a = self._encode(states, actions)
            return torch.min(self.q1(s, a), self.q2(s, a)).squeeze(-1)

    def save(self, path: str) -> None:
        """Save critic parameters."""
        torch.save(
            {
                "state_encoder": self.state_encoder.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "v_net": self.v_net.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
            },
            path,
        )
        logger.info("Saved Retrospex to %s", path)

    def load(self, path: str) -> None:
        """Load critic parameters."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.v_net.load_state_dict(ckpt["v_net"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])
        logger.info("Loaded Retrospex from %s", path)
