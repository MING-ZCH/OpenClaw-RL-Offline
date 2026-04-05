"""
KTO: Kahneman-Tversky Optimization for LLM alignment.

Reference:
    Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization",
    ICML 2024 (arXiv:2402.01306)

Key idea:
    KTO is a RLHF/alignment technique that optimises directly from *scalar*
    binary human-feedback labels (desirable / undesirable) without requiring
    paired preference data.  It is based on Kahneman-Tversky prospect theory:
    humans are loss-averse, so the disutility of an undesirable outcome is
    weighted more heavily than the utility of a desirable one.

    For agent offline RL, the binary label maps naturally to ``outcome_reward``:
        y = 1  iff  outcome_reward > success_threshold   (desirable = task success)
        y = 0  otherwise                                  (undesirable = task failure)

Loss (simplified, without separate z_ref estimation):
    r(s,a) = β · (log π(a|s) − log π_ref(a|s))      # log-ratio (implicit reward)

    L_KTO = −λ_D · E_y=1 [ σ( r(s,a) ) ]            # desirable: push ratio ↑
            −λ_U · E_y=0 [ σ(−r(s,a) ) ]            # undesirable: push ratio ↓

The reference policy π_ref is a frozen copy of the initial policy, as in standard
RLHF / DPO setups.  In the lightweight latent-embedding setup used here,
π_ref is initialized identically to π and then frozen.

Compared to DPO in this library:
    - KTO: single transitions with scalar labels — no preference pairs required.
    - DPO: requires paired (chosen, rejected) transitions.
    KTO is therefore the preferred choice for agent offline RL where individual
    trajectories have binary success/failure labels.

Architecture:
    - state_encoder / action_encoder: TextEncoder(vocab_size, hidden_dim)
    - policy: PolicyNetwork(state_dim, action_dim, hidden_dim) → log-prob
    - ref_policy: frozen clone of policy at init time
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch
from .base import BaseOfflineAlgorithm, StateEncoder, ActionEncoder, TextEncoder, TrainMetrics
from .off_policy_grpo import PolicyNetwork

logger = logging.getLogger(__name__)

_LOG_EPS = 1e-8  # Numerical stability floor for log


class KTO(BaseOfflineAlgorithm):
    """
    KTO: Kahneman-Tversky Optimization for offline LLM agent alignment.

    Trains a policy network using binary outcome labels, making it a natural
    fit for agent trajectory datasets where each episode is labelled
    success (1) or failure (0).  No paired preference data is required.

    Parameters:
        replay_buffer: Pre-collected transition dataset.
        state_dim: State/action embedding dimension (default 256).
        action_dim: Action embedding dimension (default 256).
        hidden_dim: Hidden layer width (default 256).
        lr: Encoder optimizer learning rate (default 3e-4).
        gamma: Discount factor (unused in KTO loss; kept for API consistency).
        beta: KTO temperature.  Larger → sharper distinction between
              desirable and undesirable outcomes (default 0.1).
        lambda_desirable: Weight on desirable-sample loss (default 1.0).
        lambda_undesirable: Weight on undesirable-sample loss.  Set higher
              than lambda_desirable to reflect loss-aversion (default 1.0).
        success_threshold: outcome_reward threshold for binary label (default 0.5).
        policy_lr: Learning rate for policy optimizer (default 3e-4).
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
        beta: float = 0.1,
        lambda_desirable: float = 1.0,
        lambda_undesirable: float = 1.0,
        success_threshold: float = 0.5,
        policy_lr: float = 3e-4,
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
        self.lambda_desirable = lambda_desirable
        self.lambda_undesirable = lambda_undesirable
        self.success_threshold = success_threshold
        self.policy_lr = policy_lr

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
        # Current policy and frozen reference policy                          #
        # ------------------------------------------------------------------ #
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # π_ref: frozen copy of the initial policy
        self.ref_policy = copy.deepcopy(self.policy)
        for param in self.ref_policy.parameters():
            param.requires_grad_(False)

        # ------------------------------------------------------------------ #
        # Optimizers — encoder and policy are updated; ref_policy is frozen  #
        # ------------------------------------------------------------------ #
        encoder_params = (
            list(self.state_encoder.parameters())
            + list(self.action_encoder.parameters())
        )
        self.encoder_optimizer = torch.optim.Adam(encoder_params, lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)

    # ---------------------------------------------------------------------- #
    # Training                                                                 #
    # ---------------------------------------------------------------------- #

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """
        Single KTO training step.

        Computes the KTO loss on the batch, applying desirable/undesirable
        split based on ``outcome_rewards``.

        Steps:
            1. Encode states/actions.
            2. Compute log-ratio = β · (log π(a|s) − log π_ref(a|s)).
            3. Split by binary label y (outcome_reward > threshold).
            4. Compute KTO loss and backprop.

        Returns:
            TrainMetrics with loss = total KTO loss and extras
            {loss_desirable, loss_undesirable, n_desirable, n_undesirable}.
        """
        # 1. Encode
        s_tokens = self._tokenize(batch.observation_contexts)
        a_tokens = self._tokenize(batch.actions)
        states = self.state_encoder(s_tokens)    # (B, state_dim)
        actions = self.action_encoder(a_tokens)  # (B, action_dim)

        # 2. Binary labels from outcome_rewards
        outcome_rewards = torch.as_tensor(
            batch.outcome_rewards, dtype=torch.float32
        ).to(self.device)
        y = (outcome_rewards > self.success_threshold).float()  # (B,)

        # Detach encoder outputs for policy update (encoders updated separately)
        s_det = states.detach()
        a_det = actions.detach()

        # 3. Log-probs from current and reference policy (no grad on ref)
        log_prob = self.policy(s_det, a_det)  # (B,)
        with torch.no_grad():
            log_ref = self.ref_policy(s_det, a_det)  # (B,)

        # 4. Implicit reward (log-ratio)
        log_ratio = self.beta * (log_prob - log_ref)  # (B,)

        # 5. KTO loss per sample
        #    Desirable (y=1): encourage σ(r) → 1, so loss = −log σ(r)
        #    Undesirable (y=0): encourage σ(−r) → 1, so loss = −log σ(−r)
        loss_per_sample = -(
            y * torch.log(torch.sigmoid(log_ratio).clamp(min=_LOG_EPS))
            + (1.0 - y) * torch.log(torch.sigmoid(-log_ratio).clamp(min=_LOG_EPS))
        )  # (B,)

        # Separate tracking for desirable / undesirable counts
        n_desirable = y.sum().item()
        n_undesirable = (1.0 - y).sum().item()

        # Weighted sum by lambda coefficients
        weighted = (
            self.lambda_desirable * y * loss_per_sample
            + self.lambda_undesirable * (1.0 - y) * loss_per_sample
        )
        kto_loss = weighted.mean()

        # 6. Backprop: update encoder + policy
        self.encoder_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        # Need encoder grad: compute loss again with encoder attached
        log_prob_full = self.policy(states, actions)
        with torch.no_grad():
            log_ref_full = self.ref_policy(states, actions)
        log_ratio_full = self.beta * (log_prob_full - log_ref_full)
        loss_full = -(
            y * torch.log(torch.sigmoid(log_ratio_full).clamp(min=_LOG_EPS))
            + (1.0 - y) * torch.log(torch.sigmoid(-log_ratio_full).clamp(min=_LOG_EPS))
        )
        weighted_full = (
            self.lambda_desirable * y * loss_full
            + self.lambda_undesirable * (1.0 - y) * loss_full
        )
        total_loss = weighted_full.mean()

        total_loss.backward()
        self.encoder_optimizer.step()
        self.policy_optimizer.step()

        # Compute per-group losses for monitoring
        with torch.no_grad():
            desirable_loss = (
                loss_per_sample * y
            ).sum().item() / max(n_desirable, 1.0)
            undesirable_loss = (
                loss_per_sample * (1.0 - y)
            ).sum().item() / max(n_undesirable, 1.0)

        return TrainMetrics(
            loss=total_loss.item(),
            extra={
                "loss_desirable": desirable_loss,
                "loss_undesirable": undesirable_loss,
                "n_desirable": n_desirable,
                "n_undesirable": n_undesirable,
            },
        )

    def get_action_values(
        self, states: list[str], actions: list[str]
    ) -> torch.Tensor:
        """
        Proxy Q-values: implicit KTO reward = β · (log π - log π_ref).

        Returns:
            (N,) tensor of implicit reward estimates.
        """
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        with torch.no_grad():
            s_enc = self.state_encoder(s_tokens)
            a_enc = self.action_encoder(a_tokens)
            log_prob = self.policy(s_enc, a_enc)
            log_ref = self.ref_policy(s_enc, a_enc)
            return self.beta * (log_prob - log_ref)

    def get_policy_log_probs(
        self, states: list[str], actions: list[str]
    ) -> torch.Tensor:
        """Return current policy log-probabilities for (state, action) pairs."""
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
        """Save policy, encoders, and frozen reference policy."""
        torch.save(
            {
                "state_encoder": self.state_encoder.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "policy": self.policy.state_dict(),
                "ref_policy": self.ref_policy.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Restore from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.policy.load_state_dict(ckpt["policy"])
        self.ref_policy.load_state_dict(ckpt["ref_policy"])
