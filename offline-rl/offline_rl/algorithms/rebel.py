"""
REBEL: Reinforcement Learning via Regressing Relative Rewards.

Reference:
    Gao et al., "REBEL: Reinforcement Learning via Regressing Relative Rewards",
    NeurIPS 2024 (arXiv:2404.16767)

Key idea:
    REBEL eliminates the need for a separate critic / value function by casting
    policy optimisation as pairwise squared-error regression on reward
    differences.  Given a pair (y, y') for prompt x:

        L = ( (1/eta) * (log pi(y|x)/pi_t(y|x) - log pi(y'|x)/pi_t(y'|x))
              - (r(x,y) - r(x,y')) )^2

    Here pi_t is the snapshot policy from the previous iteration (plays the
    role of a baseline), and eta is a learning rate / scale parameter.

    In offline mode the "reference" distribution mu is a static dataset, so
    y' is simply sampled from the replay buffer.  No value function, no
    clipping -- just squared loss on reward differences.

Compared to existing algorithms in this library:
    - GRPO: uses PPO-style clipping + group-relative normalisation.
    - RW-FT: reward-weighted supervised fine-tuning (one-shot, no pairwise).
    - AWAC: needs value function V(s) for weighting.
    - REBEL: lightest-weight RL; only policy + reward, squared regression.

Implementation in the discrete-embedding codebase:
    - Inherits OffPolicyGRPO for ready-made encoders + policy network.
    - Overrides train_step() entirely: no clipping, no KL penalty, no value fn.
    - Intra-batch pairing: transitions are shuffled into (y, y') pairs.
    - get_action_values() returns the policy log-prob π(a|s) as a value proxy
      (same semantics as parent GRPO).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch
from .off_policy_grpo import OffPolicyGRPO, PolicyNetwork
from .base import StateEncoder, ActionEncoder, TrainMetrics

logger = logging.getLogger(__name__)


class REBEL(OffPolicyGRPO):
    """
    REBEL: critic-free RL via pairwise reward regression.

    Parameters:
        eta: Scale parameter for the log-ratio term (higher = smaller updates).
            Corresponds to 1/eta in the paper's Eq. 1.
        ref_update_interval: How often (in train_step calls) to snapshot
            the reference policy pi_t.  Default 1 = every step.
        **kwargs: Forwarded to OffPolicyGRPO (which handles encoders / policy).
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        eta: float = 1.0,
        ref_update_interval: int = 1,
        state_dim: int = 256,
        action_dim: int = 256,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
        **kwargs,
    ):
        # Parent sets up state_encoder, action_encoder, policy, ref_policy, optimizer
        super().__init__(
            replay_buffer=replay_buffer,
            clip_ratio=0.2,          # unused; parent requires it
            kl_coeff=0.0,            # unused; no KL in REBEL
            n_policy_updates=1,      # we override train_step; 1 is fine
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma,
            device=device,
        )
        self.eta = eta
        self.ref_update_interval = ref_update_interval
        self._step_counter = 0

    # ------------------------------------------------------------------
    # Train step: pairwise squared regression on reward differences
    # ------------------------------------------------------------------

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """
        REBEL train step.

        1. Encode batch -> (s, a, s', r, d).
        2. Randomly pair transitions: (y_i, y'_j) where j is a random permutation.
        3. For each pair compute:
              delta_logr = log pi(y_i|s_i)/pi_t(y_i|s_i)
                         - log pi(y'_j|s_j)/pi_t(y'_j|s_j)
              delta_reward = r_i - r_j
              loss = (delta_logr / eta  -  delta_reward)^2
        4. Update policy (and encoders).
        5. Periodically snapshot pi -> pi_t.
        """
        states, actions = self._encode_batch(batch)
        rewards = torch.as_tensor(
            batch.rewards, dtype=torch.float32
        ).to(self.device)
        batch_size = states.size(0)

        # -- current policy log-probs --
        log_pi = self.policy(states, actions)              # (B,)
        # -- reference policy log-probs (frozen) --
        with torch.no_grad():
            log_pi_ref = self.ref_policy(states, actions)  # (B,)

        # log-ratio: log(pi / pi_t)
        log_ratio = log_pi - log_pi_ref                   # (B,)

        # -- random pairing via permutation --
        perm = torch.randperm(batch_size, device=self.device)
        log_ratio_prime = log_ratio[perm]
        rewards_prime = rewards[perm]

        # delta terms
        delta_log = (log_ratio - log_ratio_prime) / self.eta  # (B,)
        delta_r = rewards - rewards_prime                      # (B,)

        # squared regression loss
        rebel_loss = ((delta_log - delta_r) ** 2).mean()

        # backward
        self.optimizer.zero_grad()
        rebel_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                list(self.state_encoder.parameters())
                + list(self.action_encoder.parameters())
                + list(self.policy.parameters()),
                self.max_grad_norm,
            )
        self.optimizer.step()

        # periodically snapshot pi -> ref_policy
        self._step_counter += 1
        if self._step_counter % self.ref_update_interval == 0:
            self.ref_policy.load_state_dict(self.policy.state_dict())

        # metrics
        with torch.no_grad():
            mean_abs_delta_r = delta_r.abs().mean().item()
            mean_abs_delta_log = delta_log.abs().mean().item()

        return TrainMetrics(
            loss=rebel_loss.item(),
            extra={
                "rebel_loss": rebel_loss.item(),
                "mean_abs_delta_reward": mean_abs_delta_r,
                "mean_abs_delta_log_ratio": mean_abs_delta_log,
                "mean_reward": rewards.mean().item(),
            },
        )

    # ------------------------------------------------------------------
    # Inference: use parent's get_action_values() (returns policy log-prob)
    # ------------------------------------------------------------------
    # Inherited from OffPolicyGRPO:
    #   get_action_values(states, actions) -> policy log-prob (scalar per pair)

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save all parameters."""
        torch.save(
            {
                "state_encoder": self.state_encoder.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "policy": self.policy.state_dict(),
                "ref_policy": self.ref_policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step_counter": self._step_counter,
                "eta": self.eta,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load parameters."""
        ckpt = torch.load(path, map_location=self.device)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.policy.load_state_dict(ckpt["policy"])
        self.ref_policy.load_state_dict(ckpt["ref_policy"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if "step_counter" in ckpt:
            self._step_counter = ckpt["step_counter"]
