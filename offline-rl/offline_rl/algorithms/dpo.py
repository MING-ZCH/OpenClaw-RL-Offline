"""
DPO: Direct Preference Optimization for LLM alignment.

Reference:
    Rafailov et al., "Direct Preference Optimization: Your Language Model is
    Secretly a Reward Model", NeurIPS 2023 (arXiv:2305.18290)

Key idea:
    DPO shows that the KL-constrained RLHF objective has a closed-form
    optimal policy, allowing direct optimisation from preference data
    without training a separate reward model.  Given a preference pair
    (a_w ≻ a_l | s), the loss is:

        L_DPO = −E[ log σ( β · (r_w − r_l) ) ]

    where r = log π(a|s) − log π_ref(a|s) is the implicit reward.

Pairing strategy for agent trajectory data:
    Standard DPO requires paired (chosen, rejected) samples.  Since the
    TransitionBatch contains single transitions annotated with ``outcome_rewards``,
    we construct soft preference pairs *within each batch*:

        "winners" (w): transitions with outcome_reward > pairing_threshold
        "losers"  (l): transitions with outcome_reward ≤ pairing_threshold

    The two sets are paired by *symmetric sampling with replacement* so sizes
    always match.  This intra-batch pairing is common in offline DPO variants
    (e.g. RAFT-style retrieval pairing).  For larger datasets, users can also
    supply explicit preference pairs via ``train_step_from_pairs()``.

Architecture:
    - state_encoder / action_encoder: TextEncoder(vocab_size, hidden_dim)
    - policy: PolicyNetwork(state_dim, action_dim, hidden_dim) → log-prob scalar
    - ref_policy: frozen clone of policy at initialisation

Compared to KTO in this library:
    - DPO: requires preference pairs — stronger signal per sample.
    - KTO: works on single transitions with binary labels — more data-efficient.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch
from .base import BaseOfflineAlgorithm, StateEncoder, ActionEncoder, TextEncoder, TrainMetrics
from .off_policy_grpo import PolicyNetwork

logger = logging.getLogger(__name__)

_LOG_EPS = 1e-8


class DPO(BaseOfflineAlgorithm):
    """
    Direct Preference Optimization for offline LLM agent alignment.

    Constructs implicit preference pairs from batches of trajectory
    transitions (using outcome_reward as a supervision signal) and
    trains a policy via the DPO contrastive objective.

    Parameters:
        replay_buffer: Pre-collected transition dataset.
        state_dim: State embedding dimension (default 256).
        action_dim: Action embedding dimension (default 256).
        hidden_dim: Hidden MLP width (default 256).
        lr: Encoder optimizer learning rate (default 3e-4).
        gamma: Discount factor (kept for API; unused in DPO loss).
        beta: DPO temperature / KL coefficient.  Larger → more conservative
              update relative to reference policy (default 0.1).
        pairing_threshold: outcome_reward threshold separating winners from
              losers when constructing intra-batch preference pairs (default 0.5).
        policy_lr: Learning rate for policy optimizer (default 3e-4).
        label_smoothing: Smoothing on the DPO margin target (σ(β·h)→σ between
              label_smoothing and 1−label_smoothing).  Set 0.0 for vanilla DPO,
              0.1 for robust DPO variant (default 0.0).
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
        pairing_threshold: float = 0.5,
        policy_lr: float = 3e-4,
        label_smoothing: float = 0.0,
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
        self.pairing_threshold = pairing_threshold
        self.policy_lr = policy_lr
        self.label_smoothing = label_smoothing

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
        # Current and reference policy                                        #
        # ------------------------------------------------------------------ #
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        self.ref_policy = copy.deepcopy(self.policy)
        for param in self.ref_policy.parameters():
            param.requires_grad_(False)

        # ------------------------------------------------------------------ #
        # Optimizers                                                           #
        # ------------------------------------------------------------------ #
        encoder_params = (
            list(self.state_encoder.parameters())
            + list(self.action_encoder.parameters())
        )
        self.encoder_optimizer = torch.optim.Adam(encoder_params, lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)

    # ---------------------------------------------------------------------- #
    # Internals                                                                #
    # ---------------------------------------------------------------------- #

    def _make_pairs(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        outcome_rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct (winner, loser) preference pairs from a mixed batch.

        Splits transitions by outcome_reward threshold, then pairs them.
        If one group is larger, the smaller group is sampled with replacement
        to match sizes.  Returns empty tensors if either group is absent.

        Returns:
            (states_w, actions_w, states_l, actions_l)
            each of shape (n_pairs, dim).
        """
        win_mask = outcome_rewards > self.pairing_threshold
        lose_mask = ~win_mask

        idx_w = win_mask.nonzero(as_tuple=True)[0]
        idx_l = lose_mask.nonzero(as_tuple=True)[0]

        if len(idx_w) == 0 or len(idx_l) == 0:
            # Cannot form pairs — return None-sentinel
            return (
                states.new_empty(0, states.shape[-1]),
                actions.new_empty(0, actions.shape[-1]),
                states.new_empty(0, states.shape[-1]),
                actions.new_empty(0, actions.shape[-1]),
            )

        n = min(len(idx_w), len(idx_l))

        # Subsample both groups to size n (random permutation)
        perm_w = torch.randperm(len(idx_w), device=self.device)[:n]
        perm_l = torch.randperm(len(idx_l), device=self.device)[:n]
        idx_w = idx_w[perm_w]
        idx_l = idx_l[perm_l]

        return (
            states[idx_w],
            actions[idx_w],
            states[idx_l],
            actions[idx_l],
        )

    def _dpo_loss(
        self,
        states_w: torch.Tensor,
        actions_w: torch.Tensor,
        states_l: torch.Tensor,
        actions_l: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DPO loss given paired (winner, loser) encoded tensors.

        L_DPO = −E[ log σ( β · (log_π_w − log_ref_w − log_π_l + log_ref_l) ) ]

        With label smoothing (IPO / robust DPO variant):
            L = −E[ (1−ε) · log σ(h) + ε · log σ(−h) ]
        """
        log_pi_w = self.policy(states_w, actions_w)    # (n,)
        log_pi_l = self.policy(states_l, actions_l)    # (n,)

        with torch.no_grad():
            log_ref_w = self.ref_policy(states_w, actions_w)  # (n,)
            log_ref_l = self.ref_policy(states_l, actions_l)  # (n,)

        # DPO margin
        h = self.beta * (log_pi_w - log_ref_w - log_pi_l + log_ref_l)  # (n,)

        if self.label_smoothing > 0.0:
            # Robust DPO / IPO-style smooth objective
            eps = self.label_smoothing
            loss = -(
                (1.0 - eps) * torch.log(torch.sigmoid(h).clamp(min=_LOG_EPS))
                + eps * torch.log(torch.sigmoid(-h).clamp(min=_LOG_EPS))
            ).mean()
        else:
            loss = -torch.log(torch.sigmoid(h).clamp(min=_LOG_EPS)).mean()

        return loss

    # ---------------------------------------------------------------------- #
    # Public interface                                                         #
    # ---------------------------------------------------------------------- #

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """
        Single DPO training step with intra-batch preference pair construction.

        If the batch lacks sufficient diversity (all successes or all failures),
        the step is skipped and zero loss is returned.

        Returns:
            TrainMetrics with loss = DPO loss and extra = {n_pairs, mean_margin}.
        """
        # 1. Encode
        s_tokens = self._tokenize(batch.observation_contexts)
        a_tokens = self._tokenize(batch.actions)
        states = self.state_encoder(s_tokens)
        actions = self.action_encoder(a_tokens)

        outcome_rewards = torch.as_tensor(
            batch.outcome_rewards, dtype=torch.float32
        ).to(self.device)

        # 2. Construct preference pairs
        states_w, actions_w, states_l, actions_l = self._make_pairs(
            states, actions, outcome_rewards
        )

        if states_w.shape[0] == 0:
            # No pairs available in this batch — return zero loss
            logger.debug("DPO: batch has no preference pairs; skipping update.")
            return TrainMetrics(loss=0.0, extra={"n_pairs": 0, "mean_margin": 0.0})

        # 3. Compute DPO loss and update
        self.encoder_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        loss = self._dpo_loss(states_w, actions_w, states_l, actions_l)
        loss.backward()

        self.encoder_optimizer.step()
        self.policy_optimizer.step()

        # Monitor margin (no grad needed — loss already computed)
        with torch.no_grad():
            log_pi_w = self.policy(states_w.detach(), actions_w.detach())
            log_pi_l = self.policy(states_l.detach(), actions_l.detach())
            log_ref_w = self.ref_policy(states_w.detach(), actions_w.detach())
            log_ref_l = self.ref_policy(states_l.detach(), actions_l.detach())
            h = self.beta * (log_pi_w - log_ref_w - log_pi_l + log_ref_l)
            mean_margin = h.mean().item()

        return TrainMetrics(
            loss=loss.item(),
            extra={
                "n_pairs": states_w.shape[0],
                "mean_margin": mean_margin,
            },
        )

    def train_step_from_pairs(
        self,
        states_chosen: list[str],
        actions_chosen: list[str],
        states_rejected: list[str],
        actions_rejected: list[str],
    ) -> TrainMetrics:
        """
        Train directly from explicit preference pairs (bypasses batch pairing).

        Use this when the dataset already provides (chosen, rejected) pairs —
        e.g., human preference annotations stored outside the transition buffer.

        Args:
            states_chosen:   List of N state strings for preferred actions.
            actions_chosen:  List of N preferred action strings.
            states_rejected: List of N state strings for rejected actions.
            actions_rejected:List of N rejected action strings.

        Returns:
            TrainMetrics with DPO loss and extras.
        """
        sc = self._tokenize(states_chosen)
        ac = self._tokenize(actions_chosen)
        sr = self._tokenize(states_rejected)
        ar = self._tokenize(actions_rejected)

        s_w = self.state_encoder(sc)
        a_w = self.action_encoder(ac)
        s_l = self.state_encoder(sr)
        a_l = self.action_encoder(ar)

        self.encoder_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        loss = self._dpo_loss(s_w, a_w, s_l, a_l)
        loss.backward()

        self.encoder_optimizer.step()
        self.policy_optimizer.step()

        return TrainMetrics(
            loss=loss.item(),
            extra={"n_pairs": len(states_chosen), "mean_margin": 0.0},
        )

    def get_action_values(
        self, states: list[str], actions: list[str]
    ) -> torch.Tensor:
        """
        Proxy Q-values: implicit DPO reward = β · (log π(a|s) − log π_ref(a|s)).

        Higher implicit reward → policy more likely to choose this action
        relative to the reference.

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
        """Return current policy log-probabilities."""
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
