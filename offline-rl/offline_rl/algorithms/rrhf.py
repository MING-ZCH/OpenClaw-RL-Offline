"""
RRHF: Rank Responses to align Human Feedback — ranking-based alignment
without PPO.

Reference:
    Yuan et al., "RRHF: Rank Responses to Align Language Models with
    Human Feedback without tears", NeurIPS 2023 (arXiv:2304.05302)

Key equations:
    Score(y|x) = log P_θ(y|x) = (1/|y|) Σ_t log P_θ(y_t | x, y_{<t})

    L_rank = Σ_{r_i > r_j} max(0, log P_θ(y_j|x) − log P_θ(y_i|x))    (ranking)

    Where r_i is the external reward/score for response y_i.
    Responses with higher reward r should have higher log P_θ(y|x).

    Total loss = L_rank + α_sft · L_SFT

    L_SFT = −log P_θ(y_best|x) is NLL on the highest-reward response,
    keeping the model well-calibrated on high-quality generations.

Design choices:
    - **Ranking loss**: Hinge-style margin that pushes the model's
      log-probability ranking in line with external reward ordering.
    - **SFT anchoring**: The NLL term on the best response prevents
      the model from collapsing while optimizing the ranking.
    - Simpler than PPO (1-2 models instead of 4), no reward model needed
      at training time (rewards come from the dataset).

Offline agent adaptation:
    Each batch item has an outcome_reward. RRHF constructs all unique pairs
    (i, j) where r_i > r_j in the batch and applies the ranking loss.
    L_SFT is computed on the transition with the highest reward.

Architecture:
    - state_encoder / action_encoder: TextEncoder(vocab_size, hidden_dim)
    - policy: PolicyNetwork(state_dim, action_dim, hidden_dim) → log-prob
    - No reference model needed.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch
from .base import (
    BaseOfflineAlgorithm,
    StateEncoder,
    ActionEncoder,
    TextEncoder,
    TrainMetrics,
)
from .off_policy_grpo import PolicyNetwork

logger = logging.getLogger(__name__)

_LOG_EPS = 1e-8


class RRHF(BaseOfflineAlgorithm):
    """
    Rank Responses to align Human Feedback (RRHF).

    Parameters:
        replay_buffer: Pre-collected transition dataset.
        state_dim: State embedding dimension (default 256).
        action_dim: Action embedding dimension (default 256).
        hidden_dim: Hidden MLP width (default 256).
        lr: Encoder optimizer learning rate (default 3e-4).
        gamma: Discount factor (kept for API; unused in RRHF loss).
        alpha_sft: Weight for the SFT anchor loss on the best response
                   (default 1.0).
        margin: Optional fixed margin in the hinge ranking loss.  If 0.0,
                use a pure hinge: max(0, log_p_j − log_p_i).  If > 0,
                use max(0, log_p_j − log_p_i + margin) (default 0.0).
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
        alpha_sft: float = 1.0,
        margin: float = 0.0,
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
        self.alpha_sft = alpha_sft
        self.margin = margin
        self.policy_lr = policy_lr

        # ---- Encoders ----
        self.state_encoder = StateEncoder(
            vocab_size=self._vocab_size, hidden_dim=state_dim
        ).to(self.device)
        self.action_encoder = ActionEncoder(
            vocab_size=self._vocab_size, hidden_dim=action_dim
        ).to(self.device)

        # ---- Policy (no reference model) ----
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(
            self.device
        )

        # ---- Optimizers ----
        encoder_params = list(self.state_encoder.parameters()) + list(
            self.action_encoder.parameters()
        )
        self.encoder_optimizer = torch.optim.Adam(encoder_params, lr=lr)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=policy_lr
        )

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _ranking_loss(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pairwise hinge ranking loss.

        For all pairs (i, j) with r_i > r_j:
            loss += max(0, log_p_j − log_p_i + margin)

        This pushes the model to assign higher log-probability to
        transitions with higher reward.

        Args:
            log_probs: (B,) model log-probabilities for each transition.
            rewards: (B,) external reward scores.

        Returns:
            Scalar ranking loss averaged over all valid pairs.
        """
        # Build all-pairs difference matrices
        # diff_r[i, j] = r_i − r_j
        diff_r = rewards.unsqueeze(1) - rewards.unsqueeze(0)  # (B, B)
        # diff_lp[i, j] = log_p_i − log_p_j
        diff_lp = log_probs.unsqueeze(1) - log_probs.unsqueeze(0)  # (B, B)

        # Mask: only pairs where r_i > r_j (upper triangle of sorted order)
        valid = diff_r > 0  # (B, B) bool

        n_pairs = valid.sum().item()
        if n_pairs == 0:
            return log_probs.new_tensor(0.0), 0

        # Hinge loss: max(0, log_p_j − log_p_i + margin)
        #           = max(0, −diff_lp[i,j] + margin)
        hinge = torch.clamp(-diff_lp + self.margin, min=0.0)  # (B, B)
        loss = (hinge * valid.float()).sum() / max(n_pairs, 1)
        return loss, n_pairs

    def _sft_loss(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """NLL loss on the highest-reward transition in the batch."""
        best_idx = rewards.argmax()
        return -log_probs[best_idx]

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """Single RRHF training step: ranking loss + SFT anchor."""
        s_tokens = self._tokenize(batch.observation_contexts)
        a_tokens = self._tokenize(batch.actions)

        s_enc = self.state_encoder(s_tokens)
        a_enc = self.action_encoder(a_tokens)

        log_probs = self.policy(s_enc, a_enc)  # (B,)

        rewards = torch.tensor(
            batch.outcome_rewards, dtype=torch.float32, device=self.device
        )

        loss_rank, n_pairs = self._ranking_loss(log_probs, rewards)
        loss_sft = self._sft_loss(log_probs, rewards)

        loss = loss_rank + self.alpha_sft * loss_sft

        self.encoder_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.policy_optimizer.step()

        return TrainMetrics(
            loss=loss.item(),
            extra={
                "n_pairs": n_pairs,
                "loss_rank": loss_rank.item() if isinstance(loss_rank, torch.Tensor) else loss_rank,
                "loss_sft": loss_sft.item(),
            },
        )

    def get_action_values(
        self, states: list, actions: list
    ) -> torch.Tensor:
        """Return log-prob policy scores as action values."""
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        with torch.no_grad():
            s_enc = self.state_encoder(s_tokens)
            a_enc = self.action_encoder(a_tokens)
            return self.policy(s_enc, a_enc)

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path) -> None:
        torch.save(
            {
                "state_encoder": self.state_encoder.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "policy": self.policy.state_dict(),
            },
            path,
        )

    def load(self, path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.policy.load_state_dict(ckpt["policy"])
