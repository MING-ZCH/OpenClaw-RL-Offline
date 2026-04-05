"""
SimPO: Simple Preference Optimization with a Reference-Free Reward.

Reference:
    Meng et al., "SimPO: Simple Preference Optimization with a Reference-Free
    Reward", NeurIPS 2024 (arXiv:2405.14734)

Key idea:
    SimPO is a **reference-free** preference optimization method.  Instead of
    computing log-ratio differences against a frozen reference policy (as DPO
    does), it uses the **average log-probability** of the action sequence as an
    implicit reward:

        r(s, a) = (1/|a|) · log π(a|s)       # length-normalized log-prob

    The loss adds a target reward margin γ:

        L_SimPO = −E[ log σ( β/|a_w| · log π(a_w|s) − β/|a_l| · log π(a_l|s) − γ ) ]

    In our latent-embedding setup, PolicyNetwork returns a single scalar
    log-prob per (s, a) rather than per-token log-probs.  We therefore treat
    the single log-prob as already "length-normalized" (equivalent to |a|=1).
    The gamma margin still provides the contrastive offset.

Key advantage over DPO:
    No reference model needed → ~50 % less GPU memory and simpler code.

Pairing strategy:
    Same intra-batch pairing as DPO — split transitions by outcome_reward
    threshold into winners / losers and pair them symmetrically.

Architecture (strictly simpler than DPO):
    - state_encoder / action_encoder: TextEncoder
    - policy: PolicyNetwork  ← NO ref_policy
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


class SimPO(BaseOfflineAlgorithm):
    """
    Simple Preference Optimization (reference-free).

    Trains a policy via the SimPO contrastive objective using intra-batch
    preference pairs constructed from outcome_reward annotations.

    Parameters:
        replay_buffer: Pre-collected transition dataset.
        state_dim: State embedding dimension (default 256).
        action_dim: Action embedding dimension (default 256).
        hidden_dim: Hidden MLP width (default 256).
        lr: Encoder optimizer learning rate (default 3e-4).
        beta: SimPO temperature.  Larger β → more conservative updates.
              Default is 2.0 (higher than DPO's 0.1 because there is no
              log-ratio against a reference policy).
        gamma_margin: Target reward margin (γ in the paper).  The loss
              penalises pairs whose implicit-reward difference is smaller
              than γ (default 0.5).  Named ``gamma_margin`` to avoid
              collision with the base-class ``gamma`` (discount factor).
        pairing_threshold: outcome_reward threshold that separates winners
              from losers when constructing intra-batch pairs (default 0.5).
        policy_lr: Learning rate for the policy optimizer (default 3e-4).
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
        beta: float = 2.0,
        gamma_margin: float = 0.5,
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
            gamma=0.99,  # base-class discount factor (unused by SimPO)
            device=device,
            **kwargs,
        )
        self.beta = beta
        self.gamma_margin = gamma_margin
        self.pairing_threshold = pairing_threshold
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
        # Policy — NO reference model (key SimPO simplification)              #
        # ------------------------------------------------------------------ #
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)

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
            return (
                states.new_empty(0, states.shape[-1]),
                actions.new_empty(0, actions.shape[-1]),
                states.new_empty(0, states.shape[-1]),
                actions.new_empty(0, actions.shape[-1]),
            )

        n = min(len(idx_w), len(idx_l))

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

    def _simpo_loss(
        self,
        states_w: torch.Tensor,
        actions_w: torch.Tensor,
        states_l: torch.Tensor,
        actions_l: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SimPO loss given paired (winner, loser) encoded tensors.

        L_SimPO = −E[ log σ( β · (log π_w − log π_l) − γ ) ]

        No reference model is involved — the margin γ provides the
        contrastive offset that replaces the reference log-ratio in DPO.
        """
        log_pi_w = self.policy(states_w, actions_w)  # (n,)
        log_pi_l = self.policy(states_l, actions_l)  # (n,)

        h = self.beta * (log_pi_w - log_pi_l) - self.gamma_margin
        loss = -torch.log(torch.sigmoid(h).clamp(min=_LOG_EPS)).mean()
        return loss

    # ---------------------------------------------------------------------- #
    # Public interface                                                         #
    # ---------------------------------------------------------------------- #

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """
        Single SimPO training step with intra-batch preference pair construction.

        If the batch lacks sufficient diversity (all successes or all failures),
        the step is skipped and zero loss is returned.

        Returns:
            TrainMetrics with loss and extra = {n_pairs, mean_margin}.
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
            logger.debug("SimPO: batch has no preference pairs; skipping update.")
            return TrainMetrics(loss=0.0, extra={"n_pairs": 0, "mean_margin": 0.0})

        # 3. Compute SimPO loss and update
        self.encoder_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        loss = self._simpo_loss(states_w, actions_w, states_l, actions_l)
        loss.backward()

        self.encoder_optimizer.step()
        self.policy_optimizer.step()

        # 4. Monitor margin (detached — no gradient needed)
        with torch.no_grad():
            log_pi_w = self.policy(states_w.detach(), actions_w.detach())
            log_pi_l = self.policy(states_l.detach(), actions_l.detach())
            h = self.beta * (log_pi_w - log_pi_l) - self.gamma_margin
            mean_margin = h.mean().item()

        return TrainMetrics(
            loss=loss.item(),
            extra={
                "n_pairs": states_w.shape[0],
                "mean_margin": mean_margin,
            },
        )

    def get_action_values(
        self, states: list[str], actions: list[str]
    ) -> torch.Tensor:
        """
        Implicit SimPO reward: β · log π(a|s).

        Since there is no reference policy, the scaled log-probability under
        the current policy serves directly as the action-value proxy.

        Returns:
            (N,) tensor of implicit reward estimates.
        """
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        with torch.no_grad():
            s_enc = self.state_encoder(s_tokens)
            a_enc = self.action_encoder(a_tokens)
            log_prob = self.policy(s_enc, a_enc)
            return self.beta * log_prob

    # ---------------------------------------------------------------------- #
    # Persistence                                                              #
    # ---------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Save policy and encoders (no reference model to persist)."""
        torch.save(
            {
                "state_encoder": self.state_encoder.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "policy": self.policy.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Restore from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.policy.load_state_dict(ckpt["policy"])
