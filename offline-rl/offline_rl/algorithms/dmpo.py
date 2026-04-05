"""
DMPO: Direct Multi-Turn Preference Optimization for language agents.

Reference:
    Shi et al., "Direct Multi-Turn Preference Optimization for Language
    Agents", EMNLP 2024 (arXiv:2406.14868)

Key idea:
    Standard DPO applies a single-turn policy constraint (KL to reference)
    which suffers from distribution shift in multi-turn agent settings.
    DMPO fixes this by:

    1. Using *state-action occupancy measure* constraints instead of
       per-transition policy constraints.
    2. Applying *length normalisation* to the Bradley–Terry model so that
       the implicit reward is comparable across trajectories of different
       lengths.

    The DMPO loss (per trajectory pair):

        h_w = Σ_t [log π(a_t|s_t) − log π_ref(a_t|s_t)] / T_w^p
        h_l = Σ_t [log π(a_t|s_t) − log π_ref(a_t|s_t)] / T_l^p

        L_DMPO = −E[ log σ( β · (h_w − h_l) ) ]

    where T_w, T_l are trajectory lengths and p is a tuneable exponent
    (default 0.5, i.e. sqrt-normalisation).

Per-transition approximation:
    Our offline dataset stores individual (s, a) transitions, not complete
    trajectories.  We approximate the trajectory-length signal with the
    *observation-context string length* — longer context strings correspond
    to later steps in longer trajectories.  Each transition's log-ratio is
    down-weighted by  1 / len(context)^p  before applying the DPO margin.

Architecture:
    Identical to DPO (state_encoder, action_encoder, policy, ref_policy,
    two optimisers) plus a ``length_power`` hyper-parameter.
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


class DMPO(BaseOfflineAlgorithm):
    """
    Direct Multi-Turn Preference Optimization for offline LLM agent alignment.

    Extends DPO with length-normalised implicit rewards so that the
    Bradley–Terry preference model is not biased by trajectory length
    — a critical fix for multi-turn agent tasks where chosen and rejected
    rollouts may differ substantially in step count.

    Parameters:
        replay_buffer: Pre-collected transition dataset.
        state_dim: State embedding dimension (default 256).
        action_dim: Action embedding dimension (default 256).
        hidden_dim: Hidden MLP width (default 256).
        lr: Encoder optimiser learning rate (default 3e-4).
        gamma: Discount factor (kept for API; unused in DMPO loss).
        beta: DPO temperature / KL coefficient (default 0.1).
        pairing_threshold: outcome_reward threshold separating winners from
              losers when constructing intra-batch preference pairs (default 0.5).
        policy_lr: Learning rate for policy optimiser (default 3e-4).
        length_power: Exponent for length normalisation.  0.5 → sqrt,
              1.0 → linear, 0.0 → no normalisation (default 0.5).
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
        length_power: float = 0.5,
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
        self.length_power = length_power

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
        # Optimisers                                                           #
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

    def _make_pair_indices(
        self,
        outcome_rewards: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Return paired (winner, loser) *indices* from outcome_rewards.

        Splits transitions by ``pairing_threshold``, then pairs by random
        permutation so both groups have equal size.

        Returns:
            (idx_w, idx_l) each of shape (n_pairs,), or (None, None) if
            either group is empty.
        """
        win_mask = outcome_rewards > self.pairing_threshold
        lose_mask = ~win_mask

        idx_w = win_mask.nonzero(as_tuple=True)[0]
        idx_l = lose_mask.nonzero(as_tuple=True)[0]

        if len(idx_w) == 0 or len(idx_l) == 0:
            return None, None

        n = min(len(idx_w), len(idx_l))

        perm_w = torch.randperm(len(idx_w), device=self.device)[:n]
        perm_l = torch.randperm(len(idx_l), device=self.device)[:n]
        return idx_w[perm_w], idx_l[perm_l]

    def _dmpo_loss(
        self,
        states_w: torch.Tensor,
        actions_w: torch.Tensor,
        states_l: torch.Tensor,
        actions_l: torch.Tensor,
        len_w: Optional[torch.Tensor] = None,
        len_l: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the DMPO loss given paired (winner, loser) encoded tensors.

        If trajectory-length proxies ``len_w`` / ``len_l`` are provided the
        log-ratios are down-weighted by  1 / len^length_power  before computing
        the DPO margin.  Otherwise falls back to vanilla DPO (uniform weights).

            weight_i = 1 / max(1, len_i)^p
            h = β · (weight_w · lr_w  −  weight_l · lr_l)
            L = −E[ log σ(h) ]
        """
        log_pi_w = self.policy(states_w, actions_w)          # (n,)
        log_pi_l = self.policy(states_l, actions_l)          # (n,)

        with torch.no_grad():
            log_ref_w = self.ref_policy(states_w, actions_w)  # (n,)
            log_ref_l = self.ref_policy(states_l, actions_l)  # (n,)

        log_ratio_w = log_pi_w - log_ref_w
        log_ratio_l = log_pi_l - log_ref_l

        # Length normalisation
        if len_w is not None and len_l is not None:
            weight_w = 1.0 / (len_w.float().clamp(min=1.0) ** self.length_power)
            weight_l = 1.0 / (len_l.float().clamp(min=1.0) ** self.length_power)
        else:
            weight_w = torch.ones_like(log_ratio_w)
            weight_l = torch.ones_like(log_ratio_l)

        h = self.beta * (weight_w * log_ratio_w - weight_l * log_ratio_l)
        loss = -torch.log(torch.sigmoid(h).clamp(min=_LOG_EPS)).mean()
        return loss

    # ---------------------------------------------------------------------- #
    # Public interface                                                         #
    # ---------------------------------------------------------------------- #

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """
        Single DMPO training step with length-normalised intra-batch pairing.

        Steps:
            1. Encode observation contexts and actions.
            2. Compute a trajectory-length proxy from observation-context
               string lengths (longer context ≈ later step in longer trajectory).
            3. Pair winners / losers by ``outcome_reward`` threshold.
            4. Compute the DMPO loss (length-normalised DPO margin).
            5. Back-propagate and update encoders + policy.

        Returns:
            TrainMetrics with loss and extra = {n_pairs, mean_margin,
            mean_len_w, mean_len_l}.
        """
        # 1. Encode
        s_tokens = self._tokenize(batch.observation_contexts)
        a_tokens = self._tokenize(batch.actions)
        states = self.state_encoder(s_tokens)
        actions = self.action_encoder(a_tokens)

        outcome_rewards = torch.as_tensor(
            batch.outcome_rewards, dtype=torch.float32
        ).to(self.device)

        # 2. Trajectory-length proxy from observation context string lengths
        obs_lengths = torch.tensor(
            [max(1, len(ctx)) for ctx in batch.observation_contexts],
            dtype=torch.float32,
            device=self.device,
        )

        # 3. Pair indices
        idx_w, idx_l = self._make_pair_indices(outcome_rewards)

        if idx_w is None:
            logger.debug("DMPO: batch has no preference pairs; skipping update.")
            return TrainMetrics(
                loss=0.0,
                extra={"n_pairs": 0, "mean_margin": 0.0,
                       "mean_len_w": 0.0, "mean_len_l": 0.0},
            )

        # 4. Compute loss and update
        self.encoder_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        loss = self._dmpo_loss(
            states[idx_w], actions[idx_w],
            states[idx_l], actions[idx_l],
            len_w=obs_lengths[idx_w],
            len_l=obs_lengths[idx_l],
        )
        loss.backward()

        self.encoder_optimizer.step()
        self.policy_optimizer.step()

        # 5. Monitoring metrics
        with torch.no_grad():
            log_pi_w = self.policy(states[idx_w].detach(), actions[idx_w].detach())
            log_pi_l = self.policy(states[idx_l].detach(), actions[idx_l].detach())
            log_ref_w = self.ref_policy(states[idx_w].detach(), actions[idx_w].detach())
            log_ref_l = self.ref_policy(states[idx_l].detach(), actions[idx_l].detach())

            lr_w = log_pi_w - log_ref_w
            lr_l = log_pi_l - log_ref_l
            w_w = 1.0 / (obs_lengths[idx_w].clamp(min=1.0) ** self.length_power)
            w_l = 1.0 / (obs_lengths[idx_l].clamp(min=1.0) ** self.length_power)
            h = self.beta * (w_w * lr_w - w_l * lr_l)
            mean_margin = h.mean().item()

        return TrainMetrics(
            loss=loss.item(),
            extra={
                "n_pairs": int(idx_w.shape[0]),
                "mean_margin": mean_margin,
                "mean_len_w": obs_lengths[idx_w].mean().item(),
                "mean_len_l": obs_lengths[idx_l].mean().item(),
            },
        )

    def train_step_from_pairs(
        self,
        states_chosen: list[str],
        actions_chosen: list[str],
        states_rejected: list[str],
        actions_rejected: list[str],
        lengths_chosen: Optional[list[int]] = None,
        lengths_rejected: Optional[list[int]] = None,
    ) -> TrainMetrics:
        """
        Train directly from explicit preference pairs (bypasses batch pairing).

        Use this when the dataset already provides (chosen, rejected) pairs —
        e.g., human preference annotations stored outside the transition buffer.

        Args:
            states_chosen:    List of N state strings for preferred actions.
            actions_chosen:   List of N preferred action strings.
            states_rejected:  List of N state strings for rejected actions.
            actions_rejected: List of N rejected action strings.
            lengths_chosen:   Optional trajectory lengths for chosen examples.
            lengths_rejected: Optional trajectory lengths for rejected examples.

        Returns:
            TrainMetrics with DMPO loss and extras.
        """
        sc = self._tokenize(states_chosen)
        ac = self._tokenize(actions_chosen)
        sr = self._tokenize(states_rejected)
        ar = self._tokenize(actions_rejected)

        s_w = self.state_encoder(sc)
        a_w = self.action_encoder(ac)
        s_l = self.state_encoder(sr)
        a_l = self.action_encoder(ar)

        len_w = None
        len_l = None
        if lengths_chosen is not None and lengths_rejected is not None:
            len_w = torch.tensor(lengths_chosen, dtype=torch.float32, device=self.device)
            len_l = torch.tensor(lengths_rejected, dtype=torch.float32, device=self.device)

        self.encoder_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        loss = self._dmpo_loss(s_w, a_w, s_l, a_l, len_w=len_w, len_l=len_l)
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
