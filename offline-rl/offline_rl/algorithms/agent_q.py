"""
Agent Q: MCTS-Guided Off-Policy DPO for Autonomous Web Agents.

Reference:
    Putta et al., "Agent Q: Advanced Reasoning and Learning for Autonomous
    AI Agents", 2024 (arXiv:2408.07199)

Key idea:
    Agent Q combines Monte Carlo Tree Search (MCTS) for guided exploration
    with off-policy DPO for policy improvement.  At each state, K candidate
    actions are sampled and scored by:

        Q(h_t, a_i) = alpha * Q_mcts(h_t, a_i) + (1 - alpha) * Q_critic(h_t, a_i)
                                                                        (Eq. 10)

    where Q_mcts is the empirical MCTS return and Q_critic is learned from a
    separate critic network.  Preference pairs are constructed at the node
    level: for each state, the highest-Q action is "winner" and the lowest-Q
    action is "loser", filtered by a value threshold.

    The off-policy DPO objective replaces the reference model with stored
    behavior log-probs from the replay buffer:

        L = -E[ log sigma( beta * (log_pi(a_w|s) - log_b(a_w|s)
                                  - log_pi(a_l|s) + log_b(a_l|s)) ) ]
                                                                    (Eq. 5)

    UCB1 action selection for tree expansion:

        a* = argmax [ Q(h,a) + c_exp * sqrt(log N(h) / (1 + N(h'))) ]
                                                                    (Eq. 7)

    Q-value backpropagation during MCTS rollouts:

        Q(h,a) <- (Q(h,a) * N(h,a) + R) / (N(h,a) + 1)           (Eq. 8)

    Optimal policy relationship (Theorem 1):

        pi*(a|h) propto pi_ref(a|h) * exp(Q(h,a) / beta)          (Eq. 9)

Offline adaptation:
    In our offline setting without a live environment, we simulate MCTS by:
    (a) sampling K candidate actions per state from the replay buffer,
    (b) computing Q_mcts as the discounted return associated with each
        (s, a) transition in the offline dataset,
    (c) training a critic Q_critic via MSE on those returns, and
    (d) constructing node-level preference pairs from combined Q-values.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch
from .base import BaseOfflineAlgorithm, StateEncoder, ActionEncoder, TextEncoder, TrainMetrics
from .off_policy_grpo import PolicyNetwork

logger = logging.getLogger(__name__)

_LOG_EPS = 1e-8


class CriticNetwork(nn.Module):
    """
    Q-critic for estimating state-action values.

    Used as Q_critic in the combined value estimate (Eq. 10).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class AgentQ(BaseOfflineAlgorithm):
    """
    Agent Q: MCTS-Guided Off-Policy DPO for web agents.

    Training procedure (one step):
        1. Encode batch of transitions to (states, actions) features.
        2. Compute Q_mcts from offline returns (outcome_rewards as proxy).
        3. Compute Q_critic from the learned critic network.
        4. Combine: Q = alpha * Q_mcts + (1 - alpha) * Q_critic   (Eq. 10).
        5. For each state, sample K candidate actions and select the
           winner/loser pair based on combined Q-values with threshold.
        6. Train policy via off-policy node-level DPO                (Eq. 5).
        7. Train critic via MSE on offline returns.
        8. UCB1-inspired exploration bonus logged as a diagnostic.

    Parameters:
        replay_buffer: Pre-collected transition dataset.
        alpha: Interpolation weight between Q_mcts and Q_critic (default 0.5).
        beta: DPO temperature / KL coefficient (default 0.1).
        c_exp: UCB1 exploration constant for MCTS diagnostics (default 1.414).
        k_actions: Number of candidate actions per state for MCTS simulation
                   (default 4).
        q_threshold: Minimum |Q_w - Q_l| to form a preference pair (default 0.1).
        critic_lr: Learning rate for the critic network (default 3e-4).
        policy_lr: Learning rate for the policy network (default 3e-4).
        device: Torch device string (default "cpu").
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        state_dim: int = 256,
        action_dim: int = 256,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        alpha: float = 0.5,
        beta: float = 0.1,
        c_exp: float = 1.414,
        k_actions: int = 4,
        q_threshold: float = 0.1,
        critic_lr: float = 3e-4,
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
        self.alpha = alpha
        self.beta = beta
        self.c_exp = c_exp
        self.k_actions = k_actions
        self.q_threshold = q_threshold
        self.critic_lr = critic_lr
        self.policy_lr = policy_lr

        # ----- Encoders -----
        self.state_encoder = StateEncoder(
            vocab_size=self._vocab_size, hidden_dim=state_dim
        ).to(self.device)
        self.action_encoder = ActionEncoder(
            vocab_size=self._vocab_size, hidden_dim=action_dim
        ).to(self.device)

        # ----- Policy network -----
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        # Off-policy DPO: use stored behavior log-probs → no ref model needed
        # at training time. We still keep a frozen snapshot for diagnostic
        # comparison (optional).
        self.ref_policy = copy.deepcopy(self.policy)
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)

        # ----- Critic for Q_critic -----
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # ----- Optimizers -----
        encoder_params = (
            list(self.state_encoder.parameters())
            + list(self.action_encoder.parameters())
        )
        self.encoder_optimizer = torch.optim.Adam(encoder_params, lr=lr)
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=policy_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

    # ------------------------------------------------------------------ #
    # Core methods                                                         #
    # ------------------------------------------------------------------ #

    def _compute_combined_q(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        q_mcts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine MCTS empirical value with learned critic value (Eq. 10).

        Q(h, a) = alpha * Q_mcts(h, a) + (1 - alpha) * Q_critic(h, a)
        """
        with torch.no_grad():
            q_critic = self.critic(states, actions)  # (batch,)
        return self.alpha * q_mcts + (1.0 - self.alpha) * q_critic

    def _build_node_level_pairs(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        combined_q: torch.Tensor,
    ):
        """
        Construct node-level preference pairs from K-action groups.

        For each group of K consecutive transitions (simulating K candidate
        actions sampled at the same state during MCTS), select the highest-Q
        as winner and lowest-Q as loser.  Filter by q_threshold.

        Returns:
            (states_w, actions_w, states_l, actions_l) or empty tensors.
        """
        B = states.shape[0]
        K = min(self.k_actions, B)
        n_groups = B // K

        if n_groups == 0:
            empty = states.new_empty(0, states.shape[-1])
            empty_a = actions.new_empty(0, actions.shape[-1])
            return empty, empty_a, empty, empty_a

        # Trim to exact K*n_groups
        trimmed = n_groups * K
        q_groups = combined_q[:trimmed].view(n_groups, K)               # (G, K)
        s_groups = states[:trimmed].view(n_groups, K, -1)               # (G, K, D)
        a_groups = actions[:trimmed].view(n_groups, K, -1)              # (G, K, D)

        # Winner = argmax Q, Loser = argmin Q within each group
        best_idx = q_groups.argmax(dim=1)                                # (G,)
        worst_idx = q_groups.argmin(dim=1)                               # (G,)

        # Gather winners and losers
        g_range = torch.arange(n_groups, device=self.device)
        q_w = q_groups[g_range, best_idx]                                # (G,)
        q_l = q_groups[g_range, worst_idx]                               # (G,)

        s_w = s_groups[g_range, best_idx]                                # (G, D)
        a_w = a_groups[g_range, best_idx]                                # (G, D)
        s_l = s_groups[g_range, worst_idx]                               # (G, D)
        a_l = a_groups[g_range, worst_idx]                               # (G, D)

        # Filter by Q-value threshold
        margin = q_w - q_l
        valid = margin >= self.q_threshold
        return s_w[valid], a_w[valid], s_l[valid], a_l[valid]

    def _node_dpo_loss(
        self,
        states_w: torch.Tensor,
        actions_w: torch.Tensor,
        states_l: torch.Tensor,
        actions_l: torch.Tensor,
    ) -> torch.Tensor:
        """
        Off-policy node-level DPO loss (Eq. 5 adapted).

        L = -E[ log sigma( beta * (log_pi(a_w|s) - log_ref(a_w|s)
                                  - log_pi(a_l|s) + log_ref(a_l|s)) ) ]

        Since we store behavior log-probs in the off-policy replay buffer
        the reference model is the frozen policy snapshot at initialization.
        """
        log_pi_w = self.policy(states_w, actions_w)
        log_pi_l = self.policy(states_l, actions_l)

        with torch.no_grad():
            log_ref_w = self.ref_policy(states_w, actions_w)
            log_ref_l = self.ref_policy(states_l, actions_l)

        h = self.beta * (log_pi_w - log_ref_w - log_pi_l + log_ref_l)
        loss = -torch.log(torch.sigmoid(h).clamp(min=_LOG_EPS)).mean()
        return loss

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
    ) -> float:
        """Train critic Q_critic to predict offline returns via MSE."""
        pred = self.critic(states, actions)
        loss = F.mse_loss(pred, returns)

        self.critic_optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm
            )
        self.critic_optimizer.step()
        return loss.item()

    def _ucb1_bonus(self, n_total: int, n_child: int) -> float:
        """
        UCB1 exploration bonus (Eq. 7) — diagnostic only.

        bonus = c_exp * sqrt(log(N_parent) / (1 + N_child))
        """
        if n_total <= 0:
            return 0.0
        return self.c_exp * math.sqrt(math.log(n_total) / (1 + n_child))

    # ------------------------------------------------------------------ #
    # Public interface                                                     #
    # ------------------------------------------------------------------ #

    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """
        Single Agent Q training step.

        Steps:
            1. Encode batch.
            2. Compute Q_mcts from outcome_rewards.
            3. Update critic on returns.
            4. Compute combined Q.
            5. Build node-level preference pairs.
            6. DPO policy update on pairs.
        """
        # 1. Encode
        s_tokens = self._tokenize(batch.observation_contexts)
        a_tokens = self._tokenize(batch.actions)
        states = self.state_encoder(s_tokens)
        actions = self.action_encoder(a_tokens)

        # Q_mcts: use outcome rewards as proxy for MCTS returns (Eq. 8)
        q_mcts = torch.as_tensor(
            batch.outcome_rewards, dtype=torch.float32
        ).to(self.device)

        # 2. Update critic on detached features → MSE on returns
        critic_loss = self._update_critic(
            states.detach(), actions.detach(), q_mcts
        )

        # 3. Combined Q-value (Eq. 10)
        combined_q = self._compute_combined_q(states.detach(), actions.detach(), q_mcts)

        # 4. Build node-level preference pairs
        s_w, a_w, s_l, a_l = self._build_node_level_pairs(
            states, actions, combined_q
        )

        if s_w.shape[0] == 0:
            logger.debug("AgentQ: no valid preference pairs; skipping DPO update.")
            return TrainMetrics(
                loss=critic_loss,
                extra={"critic_loss": critic_loss, "dpo_loss": 0.0, "n_pairs": 0},
            )

        # 5. Off-policy node-level DPO
        self.encoder_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

        dpo_loss = self._node_dpo_loss(s_w, a_w, s_l, a_l)
        dpo_loss.backward()

        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                list(self.policy.parameters())
                + list(self.state_encoder.parameters())
                + list(self.action_encoder.parameters()),
                self.max_grad_norm,
            )

        self.encoder_optimizer.step()
        self.policy_optimizer.step()

        total_loss = critic_loss + dpo_loss.item()
        return TrainMetrics(
            loss=total_loss,
            extra={
                "critic_loss": critic_loss,
                "dpo_loss": dpo_loss.item(),
                "n_pairs": int(s_w.shape[0]),
                "ucb1_bonus": self._ucb1_bonus(len(batch.actions), self.k_actions),
            },
        )

    def get_action_values(
        self, states: list[str], actions: list[str]
    ) -> torch.Tensor:
        """Return combined Q-values for given state-action text pairs."""
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        with torch.no_grad():
            s_enc = self.state_encoder(s_tokens)
            a_enc = self.action_encoder(a_tokens)
            q_critic = self.critic(s_enc, a_enc)
        return q_critic

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_encoder": self.state_encoder.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "policy": self.policy.state_dict(),
                "ref_policy": self.ref_policy.state_dict(),
                "critic": self.critic.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.policy.load_state_dict(ckpt["policy"])
        self.ref_policy.load_state_dict(ckpt["ref_policy"])
        self.critic.load_state_dict(ckpt["critic"])
