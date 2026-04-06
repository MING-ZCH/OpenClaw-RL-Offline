"""
ORPO: Odds Ratio Preference Optimization — monolithic alignment without
a reference model.

Reference:
    Hong et al., "ORPO: Monolithic Preference Optimization without
    Reference Model", arXiv 2403.07691, 2024.

Key equations:
    log P_θ(y|x) = (1/m) Σ_t log P_θ(y_t | x, y_{<t})           (Eq. 3)
    odds_θ(y|x)  = P_θ(y|x) / (1 − P_θ(y|x))                    (Eq. 4)
    OR_θ(y_w, y_l) = odds_θ(y_w|x) / odds_θ(y_l|x)              (Eq. 5)
    L_ORPO = L_SFT + λ · L_OR                                     (Eq. 6)
    L_OR   = −log σ( log( odds_θ(y_w|x) / odds_θ(y_l|x) ) )     (Eq. 7)

Design choices:
    - **No reference model**: Unlike DPO/IPO/CPO, ORPO is reference-free.
      Only a single forward pass per chosen/rejected, halving memory.
    - **Monolithic**: SFT and alignment in a single loss. λ controls the
      relative strength of the odds-ratio penalty.
    - **Odds ratio vs probability ratio**: The odds ratio gives milder
      discrimination than the probability ratio, preventing over-suppression
      of rejected tokens during domain adaptation.

Offline agent adaptation:
    Intra-batch pairing identical to DPO — transitions above an outcome-reward
    threshold are "winners", the rest are "losers". L_SFT is NLL on the
    winner actions; L_OR is the log-sigmoid odds-ratio contrast.

Architecture:
    - state_encoder / action_encoder: TextEncoder(vocab_size, hidden_dim)
    - policy: PolicyNetwork(state_dim, action_dim, hidden_dim) → log-prob
    - No reference model at all.
"""

from __future__ import annotations

import logging
from typing import Tuple

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


class ORPO(BaseOfflineAlgorithm):
    """
    Odds Ratio Preference Optimization (ORPO).

    Parameters:
        replay_buffer: Pre-collected transition dataset.
        state_dim: State embedding dimension (default 256).
        action_dim: Action embedding dimension (default 256).
        hidden_dim: Hidden MLP width (default 256).
        lr: Encoder optimizer learning rate (default 3e-4).
        gamma: Discount factor (kept for API; unused in ORPO loss).
        lam: Weighting coefficient λ for L_OR (default 0.5, paper uses
             0.1-1.0 depending on dataset).
        pairing_threshold: outcome_reward threshold separating winners from
              losers for intra-batch preference pairs (default 0.5).
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
        lam: float = 0.5,
        pairing_threshold: float = 0.5,
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
        self.lam = lam
        self.pairing_threshold = pairing_threshold
        self.policy_lr = policy_lr

        # ---- Encoders ----
        self.state_encoder = StateEncoder(
            vocab_size=self._vocab_size, hidden_dim=state_dim
        ).to(self.device)
        self.action_encoder = ActionEncoder(
            vocab_size=self._vocab_size, hidden_dim=action_dim
        ).to(self.device)

        # ---- Policy (no reference model!) ----
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

    @staticmethod
    def _log_odds(log_p: torch.Tensor) -> torch.Tensor:
        """Convert log-probability to log-odds: log(p / (1-p)).

        Numerically stable: log_odds = log_p − log(1 − exp(log_p))
                                      = log_p − log1p(−exp(log_p))
        Clamped to avoid ±inf propagation.
        """
        # Clamp log_p to a safe range to prevent exp overflow/underflow
        log_p_clamped = log_p.clamp(min=-20.0, max=-_LOG_EPS)
        # log(1 - p) = log1p(-exp(log_p))
        log_one_minus_p = torch.log1p(-log_p_clamped.exp())
        return (log_p_clamped - log_one_minus_p).clamp(min=-20.0, max=20.0)

    def _make_pairs(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        outcome_rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct (winner, loser) preference pairs from a mixed batch."""
        win_mask = outcome_rewards > self.pairing_threshold
        lose_mask = ~win_mask

        idx_w = win_mask.nonzero(as_tuple=True)[0]
        idx_l = lose_mask.nonzero(as_tuple=True)[0]

        if len(idx_w) == 0 or len(idx_l) == 0:
            empty = states.new_empty(0, states.shape[-1])
            return empty, actions.new_empty(0, actions.shape[-1]), empty.clone(), actions.new_empty(0, actions.shape[-1])

        n = min(len(idx_w), len(idx_l))
        perm_w = torch.randperm(len(idx_w), device=self.device)[:n]
        perm_l = torch.randperm(len(idx_l), device=self.device)[:n]
        idx_w, idx_l = idx_w[perm_w], idx_l[perm_l]

        return states[idx_w], actions[idx_w], states[idx_l], actions[idx_l]

    def _orpo_loss(
        self,
        s_w: torch.Tensor,
        a_w: torch.Tensor,
        s_l: torch.Tensor,
        a_l: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute ORPO loss.

        Returns (L_ORPO, L_SFT, L_OR) tuple.

        L_SFT = −mean(log π(a_w | s_w))                          (Eq. 1/2)
        L_OR  = −log σ( log_odds(π(a_w|s_w)) − log_odds(π(a_l|s_l)) )  (Eq. 7)
        L_ORPO = L_SFT + λ · L_OR                                 (Eq. 6)
        """
        log_pi_w = self.policy(s_w, a_w)  # (n,)
        log_pi_l = self.policy(s_l, a_l)  # (n,)

        # L_SFT: NLL on chosen actions
        loss_sft = -log_pi_w.mean()

        # L_OR: log-sigmoid of log odds ratio  (Eq. 7)
        log_odds_w = self._log_odds(log_pi_w)  # (n,)
        log_odds_l = self._log_odds(log_pi_l)  # (n,)
        log_or = log_odds_w - log_odds_l        # log(OR)
        loss_or = -F.logsigmoid(log_or).mean()

        loss_orpo = loss_sft + self.lam * loss_or
        return loss_orpo, loss_sft, loss_or

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """Single ORPO training step with intra-batch preference pairing."""
        s_tokens = self._tokenize(batch.observation_contexts)
        a_tokens = self._tokenize(batch.actions)

        s_enc = self.state_encoder(s_tokens)
        a_enc = self.action_encoder(a_tokens)

        outcome_rewards = torch.tensor(
            batch.outcome_rewards, dtype=torch.float32, device=self.device
        )

        s_w, a_w, s_l, a_l = self._make_pairs(s_enc, a_enc, outcome_rewards)
        n_pairs = s_w.shape[0]

        if n_pairs == 0:
            return TrainMetrics(
                loss=0.0,
                extra={"n_pairs": 0, "loss_sft": 0.0, "loss_or": 0.0},
            )

        loss_orpo, loss_sft, loss_or = self._orpo_loss(s_w, a_w, s_l, a_l)

        self.encoder_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        loss_orpo.backward()
        self.encoder_optimizer.step()
        self.policy_optimizer.step()

        return TrainMetrics(
            loss=loss_orpo.item(),
            extra={
                "n_pairs": n_pairs,
                "loss_sft": loss_sft.item(),
                "loss_or": loss_or.item(),
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
