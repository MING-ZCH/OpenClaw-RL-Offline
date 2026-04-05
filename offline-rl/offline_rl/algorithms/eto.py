"""
ETO: Exploration-based Trajectory Optimization for LLM agent alignment.

Reference:
    Song et al., "Trial and Error: Exploration-Based Trajectory Optimization
    for LLM Agents", ACL 2024 (arXiv:2403.02502)

Key idea:
    ETO combines exploration replay with contrastive DPO-style training.
    It maintains a contrastive buffer of (successful, failed) trajectory
    pairs and applies a preference optimisation loss similar to DPO, with
    an additional *exploration advantage bonus* that upweights "near-miss"
    failures — i.e. failed trajectories that achieved high intermediate
    reward.  This encourages the policy to keep exploring regions near the
    decision boundary between success and failure.

Offline adaptation:
    In our offline RL setup (no live environment interaction), we adapt
    ETO by:
      1. Using the offline replay buffer to construct contrastive pairs
         (winners vs. losers split by outcome reward).
      2. Adding an exploration bonus that re-weights pairs based on how
         close the failing trajectory was to success.

ETO loss:
    L_ETO = −E[ w(r_l) · log σ( β · (log_ratio_w − log_ratio_l) ) ]

    where
        log_ratio = log π(a|s) − log π_ref(a|s)
        w(r_l)    = exp(explore_alpha · r_l) / mean(exp(explore_alpha · r_l))

    The weighting w(r_l) assigns higher importance to near-miss losers
    (higher reward failures) so the policy is steered toward exploring
    those promising regions.

Architecture:
    Identical to DPO in this library:
      - state_encoder / action_encoder: TextEncoder(vocab_size, hidden_dim)
      - policy: PolicyNetwork(state_dim, action_dim, hidden_dim) → log-prob
      - ref_policy: frozen clone of policy at initialisation

Compared to DPO:
    - DPO: uniform weighting across all preference pairs.
    - ETO: exploration-bonus weighting that upweights near-miss failures,
      encouraging finer-grained exploration near the success boundary.
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


class ETO(BaseOfflineAlgorithm):
    """
    Exploration-based Trajectory Optimization for offline LLM agent alignment.

    Constructs contrastive preference pairs from batches of trajectory
    transitions and trains a policy via the ETO objective — a DPO-style
    contrastive loss augmented with an exploration bonus that upweights
    near-miss failures.

    Parameters:
        replay_buffer: Pre-collected transition dataset.
        state_dim: State embedding dimension (default 256).
        action_dim: Action embedding dimension (default 256).
        hidden_dim: Hidden MLP width (default 256).
        lr: Encoder optimizer learning rate (default 3e-4).
        gamma: Discount factor (kept for API; unused in ETO loss).
        beta: DPO temperature / KL coefficient.  Larger → more conservative
              update relative to reference policy (default 0.1).
        pairing_threshold: outcome_reward threshold separating winners from
              losers when constructing intra-batch preference pairs (default 0.5).
        policy_lr: Learning rate for policy optimizer (default 3e-4).
        explore_alpha: Exploration bonus scale for near-miss weighting.
              Higher values assign more weight to high-reward failures
              (default 1.0).
        pair_margin: Minimum reward difference threshold for pair formation.
              Pairs whose reward gap (winner − loser) is below this margin
              are discarded (default 0.0, meaning no filtering).
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
        explore_alpha: float = 1.0,
        pair_margin: float = 0.0,
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
        self.explore_alpha = explore_alpha
        self.pair_margin = pair_margin

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

    def _make_pairs_with_rewards(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        outcome_rewards: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct (winner, loser) preference pairs and return loser rewards.

        Like DPO's ``_make_pairs`` but also returns the outcome rewards of
        the losing trajectories so that the ETO exploration bonus can
        upweight near-miss failures.

        Optionally filters out pairs whose reward gap is below
        ``self.pair_margin``.

        Returns:
            (states_w, actions_w, states_l, actions_l, rewards_l)
            each of shape (n_pairs, dim), with rewards_l being (n_pairs,).
        """
        win_mask = outcome_rewards > self.pairing_threshold
        lose_mask = ~win_mask

        idx_w = win_mask.nonzero(as_tuple=True)[0]
        idx_l = lose_mask.nonzero(as_tuple=True)[0]

        if len(idx_w) == 0 or len(idx_l) == 0:
            empty_s = states.new_empty(0, states.shape[-1])
            empty_a = actions.new_empty(0, actions.shape[-1])
            return empty_s, empty_a, empty_s, empty_a, outcome_rewards.new_empty(0)

        n = min(len(idx_w), len(idx_l))

        # Subsample both groups to size n (random permutation)
        perm_w = torch.randperm(len(idx_w), device=self.device)[:n]
        perm_l = torch.randperm(len(idx_l), device=self.device)[:n]
        idx_w = idx_w[perm_w]
        idx_l = idx_l[perm_l]

        # Optional margin filtering
        if self.pair_margin > 0.0:
            gap = outcome_rewards[idx_w] - outcome_rewards[idx_l]
            valid = gap > self.pair_margin
            if valid.sum() == 0:
                empty_s = states.new_empty(0, states.shape[-1])
                empty_a = actions.new_empty(0, actions.shape[-1])
                return empty_s, empty_a, empty_s, empty_a, outcome_rewards.new_empty(0)
            idx_w = idx_w[valid]
            idx_l = idx_l[valid]

        return (
            states[idx_w],
            actions[idx_w],
            states[idx_l],
            actions[idx_l],
            outcome_rewards[idx_l],  # loser rewards for exploration weighting
        )

    def _eto_loss(
        self,
        states_w: torch.Tensor,
        actions_w: torch.Tensor,
        states_l: torch.Tensor,
        actions_l: torch.Tensor,
        rewards_l: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the ETO contrastive loss with exploration bonus.

        L_ETO = −E[ w(r_l) · log σ( β · (log_ratio_w − log_ratio_l) ) ]

        where w(r_l) = exp(explore_alpha · r_l), normalised by its mean
        to prevent gradient scale explosion.
        """
        log_pi_w = self.policy(states_w, actions_w)    # (n,)
        log_pi_l = self.policy(states_l, actions_l)    # (n,)

        with torch.no_grad():
            log_ref_w = self.ref_policy(states_w, actions_w)  # (n,)
            log_ref_l = self.ref_policy(states_l, actions_l)  # (n,)

        # DPO-style contrastive margin
        h = self.beta * (log_pi_w - log_ref_w - log_pi_l + log_ref_l)  # (n,)

        # Exploration bonus: upweight near-miss failures
        explore_weight = torch.exp(self.explore_alpha * rewards_l).detach()
        # Normalize weights to avoid gradient scale explosion
        explore_weight = explore_weight / explore_weight.mean().clamp(min=_LOG_EPS)

        loss = -(explore_weight * torch.log(torch.sigmoid(h).clamp(min=_LOG_EPS))).mean()
        return loss

    # ---------------------------------------------------------------------- #
    # Public interface                                                         #
    # ---------------------------------------------------------------------- #

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """
        Single ETO training step with intra-batch contrastive pair construction.

        If the batch lacks sufficient diversity (all successes or all failures),
        the step is skipped and zero loss is returned.

        Returns:
            TrainMetrics with loss = ETO loss and extra =
            {n_pairs, mean_margin, mean_explore_weight}.
        """
        # 1. Encode
        s_tokens = self._tokenize(batch.observation_contexts)
        a_tokens = self._tokenize(batch.actions)
        states = self.state_encoder(s_tokens)
        actions = self.action_encoder(a_tokens)

        outcome_rewards = torch.as_tensor(
            batch.outcome_rewards, dtype=torch.float32
        ).to(self.device)

        # 2. Construct preference pairs with loser rewards
        states_w, actions_w, states_l, actions_l, rewards_l = (
            self._make_pairs_with_rewards(states, actions, outcome_rewards)
        )

        if states_w.shape[0] == 0:
            logger.debug("ETO: batch has no preference pairs; skipping update.")
            return TrainMetrics(
                loss=0.0,
                extra={"n_pairs": 0, "mean_margin": 0.0, "mean_explore_weight": 0.0},
            )

        # 3. Compute ETO loss and update
        self.encoder_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        loss = self._eto_loss(states_w, actions_w, states_l, actions_l, rewards_l)
        loss.backward()

        self.encoder_optimizer.step()
        self.policy_optimizer.step()

        # 4. Monitor diagnostics (no grad needed)
        with torch.no_grad():
            log_pi_w = self.policy(states_w.detach(), actions_w.detach())
            log_pi_l = self.policy(states_l.detach(), actions_l.detach())
            log_ref_w = self.ref_policy(states_w.detach(), actions_w.detach())
            log_ref_l = self.ref_policy(states_l.detach(), actions_l.detach())
            h = self.beta * (log_pi_w - log_ref_w - log_pi_l + log_ref_l)
            mean_margin = h.mean().item()

            explore_weight = torch.exp(self.explore_alpha * rewards_l)
            explore_weight = explore_weight / explore_weight.mean().clamp(min=_LOG_EPS)
            mean_explore_weight = explore_weight.mean().item()

        return TrainMetrics(
            loss=loss.item(),
            extra={
                "n_pairs": states_w.shape[0],
                "mean_margin": mean_margin,
                "mean_explore_weight": mean_explore_weight,
            },
        )

    def get_action_values(
        self, states: list[str], actions: list[str]
    ) -> torch.Tensor:
        """
        Proxy Q-values: implicit reward = β · (log π(a|s) − log π_ref(a|s)).

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
