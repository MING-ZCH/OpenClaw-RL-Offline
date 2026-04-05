"""
Decision Transformer (DT) for offline RL of LLM-based agents.

Reference: Chen et al., "Decision Transformer: Reinforcement Learning via
Sequence Modeling", NeurIPS 2021 (arXiv:2106.01345)

Key idea: Frame offline RL as a sequence-to-sequence problem.  Given a
context of (return-to-go, state, action) tuples and a target return,
autoregressively predict the next action.  Training is purely supervised
(behavioral cloning on return-conditioned sequences), avoiding Q-learning
instability.

Architecture:
    Each timestep t contributes 3 tokens: (R_t, s_t, a_t)
    R_t (scalar RTG) → linear projection to d_model
    s_t (text)       → TextEncoder → d_model
    a_t (text)       → ActionEncoder → d_model
    Timestep t       → positional embedding (nn.Embedding)
    Sequence of 3*K tokens fed through a causal GPT-style transformer.
    Action prediction head at every action position predicts action embedding.

Integration note:
    DT requires trajectory-level batches rather than individual transitions.
    The `_needs_trajectory_batch` flag signals the training loop to use
    `replay_buffer.sample_trajectories()` instead of `sample_transitions()`.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch
from offline_rl.data.trajectory_store import Trajectory
from .base import BaseOfflineAlgorithm, StateEncoder, ActionEncoder, TrainMetrics

logger = logging.getLogger(__name__)


class _CausalTransformer(nn.Module):
    """
    Small GPT-style causal transformer encoder.

    Applies self-attention with a causal (upper-triangular) mask so that
    each token can only attend to itself and earlier tokens.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) input sequence.
        Returns:
            (batch, seq_len, d_model) causally-attended output.
        """
        seq_len = x.size(1)
        # Causal mask: -inf above diagonal (additive attention mask)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device),
            diagonal=1,
        )
        return self.transformer(x, mask=causal_mask)


class DecisionTransformer(BaseOfflineAlgorithm):
    """
    Decision Transformer for LLM-based agent offline training.

    The model learns to predict actions conditional on:
    - A target return-to-go (how much reward to collect from here onward)
    - The observation history up to the current step
    - The action history up to the current step

    Training is supervised BC: MSE between predicted and actual action
    embeddings across all timesteps in the context window.

    This class overrides `_needs_trajectory_batch = True`, which causes
    the training loop to call `replay_buffer.sample_trajectories()` and
    pass the resulting `list[Trajectory]` to `train_step`.

    Inference:
        Pass a target return and a state; the model greedily produces action
        embeddings which are compared to known actions for evaluation.
    """

    _needs_trajectory_batch: bool = True  # training loop uses sample_trajectories()

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        state_dim: int = 256,
        action_dim: int = 256,
        hidden_dim: int = 256,
        lr: float = 1e-4,
        gamma: float = 0.99,
        context_len: int = 20,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        max_ep_len: int = 200,
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
        self.context_len = context_len
        self.gamma = gamma
        self.max_ep_len = max_ep_len
        d_model = hidden_dim

        # ── Text encoders (shared with evaluation path) ───────────────────
        self.state_encoder = StateEncoder(
            vocab_size=self._vocab_size, hidden_dim=state_dim
        ).to(self.device)
        self.action_encoder = ActionEncoder(
            vocab_size=self._vocab_size, hidden_dim=action_dim
        ).to(self.device)

        # ── Input projections ─────────────────────────────────────────────
        # Project state_dim → d_model (if they differ)
        self.state_proj = nn.Linear(state_dim, d_model).to(self.device)
        self.action_proj = nn.Linear(action_dim, d_model).to(self.device)
        # Return-to-go is a single scalar per timestep
        self.rtg_proj = nn.Linear(1, d_model).to(self.device)
        # Timestep positional embedding
        self.timestep_emb = nn.Embedding(max_ep_len, d_model).to(self.device)
        # Token-type embedding: 0=RTG, 1=state, 2=action
        self.token_type_emb = nn.Embedding(3, d_model).to(self.device)

        # ── Causal transformer backbone ───────────────────────────────────
        self.transformer = _CausalTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dim_feedforward=d_model * 4,
        ).to(self.device)

        # ── Action prediction head ────────────────────────────────────────
        self.action_head = nn.Linear(d_model, action_dim).to(self.device)

        # ── LayerNorm before transformer ──────────────────────────────────
        self.embed_ln = nn.LayerNorm(d_model).to(self.device)

        # ── Optimizer (all parameters under one optimizer) ────────────────
        self.optimizer = torch.optim.Adam(
            list(self.state_encoder.parameters())
            + list(self.action_encoder.parameters())
            + list(self.state_proj.parameters())
            + list(self.action_proj.parameters())
            + list(self.rtg_proj.parameters())
            + list(self.timestep_emb.parameters())
            + list(self.token_type_emb.parameters())
            + list(self.transformer.parameters())
            + list(self.action_head.parameters())
            + list(self.embed_ln.parameters()),
            lr=lr,
        )

        self._total_steps = 0

    # ──────────────────────────────────────────────────────────────────────
    # Trajectory preparation helpers
    # ──────────────────────────────────────────────────────────────────────

    def _compute_returns_to_go(self, rewards: List[float]) -> List[float]:
        """Compute discounted returns-to-go for a trajectory."""
        rtgs = []
        g = 0.0
        for r in reversed(rewards):
            g = r + self.gamma * g
            rtgs.insert(0, g)
        return rtgs

    def _traj_to_tensors(self, traj: Trajectory):
        """Convert a Trajectory to padded/truncated tensors.

        Returns dicts of string lists and float lists for a single trajectory,
        ready for tokenization.  The context window is the LAST `context_len`
        steps of the trajectory (most recent context).
        """
        steps = traj.steps
        n = len(steps)

        # Build per-step observation contexts and rewards
        obs_contexts = []
        action_texts = []
        rewards_list = []

        for i, step in enumerate(steps):
            ctx = "Task: %s\n" % traj.instruction
            for prev_step in steps[:i]:
                ctx += "Step %d: %s\n" % (prev_step.step_idx, prev_step.action)
            obs_contexts.append(ctx)
            action_texts.append(step.action)
            r = step.reward
            if r == 0.0 and i == n - 1:
                r = traj.outcome_reward
            rewards_list.append(float(r))

        rtgs = self._compute_returns_to_go(rewards_list)

        # Truncate/select context window (last context_len steps)
        start = max(0, n - self.context_len)
        obs_contexts = obs_contexts[start:]
        action_texts = action_texts[start:]
        rtgs = rtgs[start:]
        timesteps = list(range(start, start + len(obs_contexts)))

        return obs_contexts, action_texts, rtgs, timesteps

    def _encode_sequence_batch(
        self, trajectories: List[Trajectory]
    ):
        """Encode a batch of trajectories into padded sequence tensors.

        Returns:
            rtg_emb:    (B, K, d_model)
            state_emb:  (B, K, d_model)
            action_emb: (B, K, d_model)
            mask:       (B, K) bool — True = valid token, False = pad
        """
        d_model = self.transformer.transformer.layers[0].self_attn.embed_dim
        K = self.context_len
        B = len(trajectories)

        rtg_emb = torch.zeros(B, K, d_model, device=self.device)
        state_emb = torch.zeros(B, K, d_model, device=self.device)
        action_emb = torch.zeros(B, K, d_model, device=self.device)
        ts_idx = torch.zeros(B, K, dtype=torch.long, device=self.device)
        mask = torch.zeros(B, K, dtype=torch.bool, device=self.device)

        for b, traj in enumerate(trajectories):
            obs_ctxs, act_texts, rtgs, timesteps = self._traj_to_tensors(traj)
            T = len(obs_ctxs)

            # Encode states
            s_tok = self._tokenize(obs_ctxs)       # (T, max_token_len)
            a_tok = self._tokenize(act_texts)       # (T, max_token_len)
            s_emb = self.state_encoder(s_tok)       # (T, state_dim)
            a_emb = self.action_encoder(a_tok)      # (T, action_dim)

            s_proj = self.state_proj(s_emb)         # (T, d_model)
            a_proj = self.action_proj(a_emb)        # (T, d_model)

            rtg_t = torch.tensor(
                rtgs, dtype=torch.float32, device=self.device
            ).unsqueeze(-1)                         # (T, 1)
            r_proj = self.rtg_proj(rtg_t)           # (T, d_model)

            ts_t = torch.tensor(
                timesteps, dtype=torch.long, device=self.device
            ).clamp(max=self.max_ep_len - 1)        # (T,)

            # Fill into padded tensors (pack into the last T positions)
            rtg_emb[b, :T] = r_proj
            state_emb[b, :T] = s_proj
            action_emb[b, :T] = a_proj
            ts_idx[b, :T] = ts_t
            mask[b, :T] = True

        # Add timestep positional embeddings
        ts_emb = self.timestep_emb(ts_idx)          # (B, K, d_model)
        rtg_emb = rtg_emb + ts_emb
        state_emb = state_emb + ts_emb
        action_emb = action_emb + ts_emb

        return rtg_emb, state_emb, action_emb, mask

    def _build_interleaved_sequence(
        self,
        rtg_emb: torch.Tensor,    # (B, K, d)
        state_emb: torch.Tensor,  # (B, K, d)
        action_emb: torch.Tensor, # (B, K, d)
    ) -> torch.Tensor:
        """Build the interleaved token sequence: [R1, s1, a1, R2, s2, a2, ...].

        Returns:
            (B, 3*K, d_model) input tensor with token-type embeddings added.
        """
        B, K, d = rtg_emb.shape

        # Token type offsets: 0=RTG, 1=state, 2=action
        type_0 = self.token_type_emb(
            torch.zeros(B, K, dtype=torch.long, device=rtg_emb.device)
        )
        type_1 = self.token_type_emb(
            torch.ones(B, K, dtype=torch.long, device=rtg_emb.device)
        )
        type_2 = self.token_type_emb(
            torch.full((B, K), 2, dtype=torch.long, device=rtg_emb.device)
        )

        rtg_tok = rtg_emb + type_0         # (B, K, d)
        state_tok = state_emb + type_1     # (B, K, d)
        action_tok = action_emb + type_2   # (B, K, d)

        # Interleave: shape (B, 3*K, d)
        # Stack along new dim and reshape
        interleaved = torch.stack(
            [rtg_tok, state_tok, action_tok], dim=2
        ).view(B, 3 * K, d)

        return self.embed_ln(interleaved)

    # ──────────────────────────────────────────────────────────────────────
    # Override train() to use trajectory-level sampling
    # ──────────────────────────────────────────────────────────────────────

    def train(
        self,
        num_steps: int = 1000,
        batch_size: int = 16,
        log_interval: int = 100,
    ):
        """Trajectory-batch training loop for Decision Transformer."""
        all_metrics = []
        for step in range(num_steps):
            trajs = self.replay_buffer.sample_trajectories(batch_size)
            if not trajs:
                logger.warning("Replay buffer has no trajectories; skipping step.")
                continue
            metrics = self.train_step(trajs)
            all_metrics.append(metrics)

            if (step + 1) % log_interval == 0:
                recent = all_metrics[-log_interval:]
                avg_loss = sum(m.loss for m in recent) / len(recent)
                logger.info(
                    "Step %d/%d: avg_loss=%.4f", step + 1, num_steps, avg_loss
                )
        return all_metrics

    # ──────────────────────────────────────────────────────────────────────
    # Core train_step on a list of Trajectory objects
    # ──────────────────────────────────────────────────────────────────────

    def train_step(self, batch):  # type: ignore[override]
        """BC training step on a batch of trajectories.

        Args:
            batch: list[Trajectory] from replay_buffer.sample_trajectories().

        Returns:
            TrainMetrics with BC MSE loss.
        """
        if not batch:
            return TrainMetrics(loss=0.0, extra={"bc_loss": 0.0})

        self._total_steps += 1
        trajectories = batch

        # Encode sequences
        rtg_emb, state_emb, action_emb, mask = self._encode_sequence_batch(
            trajectories
        )

        B, K, _ = rtg_emb.shape

        # Build interleaved DT sequence: (B, 3*K, d_model)
        seq = self._build_interleaved_sequence(rtg_emb, state_emb, action_emb)

        # Causal transformer forward: (B, 3*K, d_model)
        out = self.transformer(seq)

        # Action tokens are at positions 2, 5, 8, ..., 3t+2 (0-indexed)
        # The model predicts action at position 3t+2 using output at 3t+1 (state pos)
        # Following the original DT paper: predict a_t from output at s_t position
        state_positions = torch.arange(1, 3 * K, 3, device=self.device)  # 1,4,7,...
        action_positions = torch.arange(2, 3 * K, 3, device=self.device)  # 2,5,8,...

        state_out = out[:, state_positions, :]      # (B, K, d_model) — state hidden states
        pred_actions = self.action_head(state_out)  # (B, K, action_dim)

        # Target: actual action embeddings
        # action_emb was projected to d_model; we need the original action_dim targets
        # Re-use the action_emb which is (B, K, d_model projected from action_dim)
        # For BC loss, compare predicted action to the action_proj output (d_model)
        # OR: compare to the encoded action before projection (preferred for BC fidelity)
        # Here we match dimensions: pred_actions is (B, K, action_dim)
        # We need targets in action_dim space → use action encoder output directly

        # We already have action_emb = action_proj(action_tokens), which is d_model.
        # Instead, retrieve the raw action encoder outputs again for BC target.
        # To avoid double computation, decode action_emb back via a pseudo inverse
        # is impractical. We'll compute targets inline (cheap since cached in encoder).
        # Note: target is action_encoder output (action_dim), not projected (d_model).
        # pred_actions: (B, K, action_dim) matches this.

        # Build target action embeddings (action_dim space)
        target_actions = torch.zeros(
            B, K, self.action_dim, device=self.device
        )
        for b, traj in enumerate(trajectories):
            obs_ctxs, act_texts, _, _ = self._traj_to_tensors(traj)
            T = len(act_texts)
            a_tok = self._tokenize(act_texts)
            with torch.no_grad():
                a_emb_raw = self.action_encoder(a_tok)  # (T, action_dim)
            target_actions[b, :T] = a_emb_raw

        # BC loss: MSE on valid (non-padded) tokens only
        # mask: (B, K) — True = valid
        valid_mask = mask.unsqueeze(-1).float()  # (B, K, 1)
        bc_loss_all = F.mse_loss(pred_actions, target_actions, reduction="none")
        bc_loss = (bc_loss_all * valid_mask).sum() / (valid_mask.sum() + 1e-8)

        self.optimizer.zero_grad()
        bc_loss.backward()
        if self.max_grad_norm is not None:
            clip_grad_norm_(
                [p for p in self.optimizer.param_groups[0]["params"]],
                self.max_grad_norm,
            )
        self.optimizer.step()

        return TrainMetrics(
            loss=bc_loss.item(),
            extra={"bc_loss": bc_loss.item()},
        )

    # ──────────────────────────────────────────────────────────────────────
    # Evaluation interface
    # ──────────────────────────────────────────────────────────────────────

    def get_action_values(
        self, states: list[str], actions: list[str]
    ) -> torch.Tensor:
        """
        Proxy Q-values: negative single-step BC reconstruction loss.

        DT has no Q-function.  We use -BC_loss(s, a) as a measure of how
        well the model reconstructs the action given state context alone.
        Higher value = action is more consistent with learned policy.

        Returns:
            (B,) tensor of -MSE reconstruction scores.
        """
        s_tokens = self._tokenize(states)
        a_tokens = self._tokenize(actions)
        B = len(states)

        with torch.no_grad():
            s_emb = self.state_encoder(s_tokens)     # (B, state_dim)
            a_emb = self.action_encoder(a_tokens)    # (B, action_dim)

            # Single-step sequence: just (R=1, s, a) per sample
            rtg_scalar = torch.ones(B, 1, 1, device=self.device)
            r_proj = self.rtg_proj(rtg_scalar)                      # (B, 1, d)
            s_proj = self.state_proj(s_emb).unsqueeze(1)            # (B, 1, d)
            a_proj = self.action_proj(a_emb).unsqueeze(1)           # (B, 1, d)

            ts_emb = self.timestep_emb(
                torch.zeros(B, 1, dtype=torch.long, device=self.device)
            )
            r_proj = r_proj + ts_emb
            s_proj = s_proj + ts_emb
            a_proj = a_proj + ts_emb

            seq = self._build_interleaved_sequence(r_proj, s_proj, a_proj)
            out = self.transformer(seq)  # (B, 3, d_model)

            # State position = index 1 in (R, s, a) sequence
            state_out = out[:, 1, :]                         # (B, d_model)
            pred_a = self.action_head(state_out)             # (B, action_dim)

            mse = F.mse_loss(pred_a, a_emb, reduction="none").mean(dim=-1)  # (B,)
            return -mse  # Higher = better reconstruction = more likely action

    def save(self, path) -> None:
        torch.save(
            {
                "state_encoder": self.state_encoder.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "state_proj": self.state_proj.state_dict(),
                "action_proj": self.action_proj.state_dict(),
                "rtg_proj": self.rtg_proj.state_dict(),
                "timestep_emb": self.timestep_emb.state_dict(),
                "token_type_emb": self.token_type_emb.state_dict(),
                "transformer": self.transformer.state_dict(),
                "action_head": self.action_head.state_dict(),
                "embed_ln": self.embed_ln.state_dict(),
                "total_steps": self._total_steps,
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info("Saved DecisionTransformer to %s", path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.state_proj.load_state_dict(ckpt["state_proj"])
        self.action_proj.load_state_dict(ckpt["action_proj"])
        self.rtg_proj.load_state_dict(ckpt["rtg_proj"])
        self.timestep_emb.load_state_dict(ckpt["timestep_emb"])
        self.token_type_emb.load_state_dict(ckpt["token_type_emb"])
        self.transformer.load_state_dict(ckpt["transformer"])
        self.action_head.load_state_dict(ckpt["action_head"])
        self.embed_ln.load_state_dict(ckpt["embed_ln"])
        self._total_steps = ckpt.get("total_steps", 0)
        logger.info("Loaded DecisionTransformer from %s", path)
