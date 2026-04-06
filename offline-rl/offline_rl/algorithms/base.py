"""
Base class for offline RL algorithms.

All offline RL algorithms operate on pre-collected trajectories stored in
a ReplayBuffer. They share a common interface for training and evaluation.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from offline_rl.data.replay_buffer import ReplayBuffer, TransitionBatch

logger = logging.getLogger(__name__)


@dataclass
class TrainMetrics:
    """Metrics from a single training step."""
    loss: float
    extra: dict[str, float]

    def log_str(self) -> str:
        parts = [f"loss={self.loss:.4f}"]
        for k, v in self.extra.items():
            parts.append(f"{k}={v:.4f}")
        return " ".join(parts)


class TextEncoder(nn.Module):
    """
    Encodes text token IDs into a fixed-dim vector via EmbeddingBag + FC.

    For LLM-based agents, this would typically use the LLM's hidden states.
    This lightweight version uses a learned embedding for CPU testing.
    Used for both state and action encoding (identical architecture).
    """

    def __init__(self, vocab_size: int = 32000, embed_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.embed = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.fc = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, seq_len) integer tensor
        Returns:
            (batch, hidden_dim) embedding
        """
        embedded = self.embed(token_ids)
        return self.activation(self.fc(embedded))


# Backward-compatible aliases
StateEncoder = TextEncoder
ActionEncoder = TextEncoder


class BaseOfflineAlgorithm(abc.ABC):
    """
    Abstract base class for offline RL algorithms.

    Subclasses implement the core training logic. All algorithms share:
    - A state encoder (text → vector)
    - An action encoder (text → vector)
    - A replay buffer for sampling
    - A simple tokenizer for text→token_ids conversion
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        state_dim: int = 256,
        action_dim: int = 256,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
        max_token_len: int = 128,
        max_grad_norm: Optional[float] = None,
    ):
        self.replay_buffer = replay_buffer
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.device = torch.device(device)
        self.max_token_len = max_token_len
        self.max_grad_norm = max_grad_norm

        # Lightweight tokenizer: simple character-level hash
        # In production, this would be replaced by the LLM's tokenizer
        self._vocab_size = 5000  # Small for CPU testing; production uses 32000+

    def _tokenize(self, texts: list[str]) -> torch.Tensor:
        """Simple hash-based tokenization for testing. Replace with real tokenizer."""
        batch = []
        for text in texts:
            ids = [hash(c) % self._vocab_size for c in text[:self.max_token_len]]
            if len(ids) < self.max_token_len:
                ids += [0] * (self.max_token_len - len(ids))
            batch.append(ids[:self.max_token_len])
        return torch.tensor(batch, dtype=torch.long, device=self.device)

    def _encode_batch(self, batch: TransitionBatch):
        """Encode text batch into state/action/next_state tensors + rewards/dones.

        Shared by IQL, CQL, AWAC. GRPO overrides with a lighter version.
        Assumes subclass has defined self.state_encoder and self.action_encoder.
        """
        s_tokens = self._tokenize(batch.observation_contexts)
        a_tokens = self._tokenize(batch.actions)
        ns_tokens = self._tokenize(batch.next_observation_contexts)

        states = self.state_encoder(s_tokens)
        actions = self.action_encoder(a_tokens)
        next_states = self.state_encoder(ns_tokens)

        rewards = torch.as_tensor(
            batch.rewards, dtype=torch.float32
        ).to(self.device)
        dones = torch.as_tensor(
            batch.dones, dtype=torch.float32
        ).to(self.device)

        return states, actions, next_states, rewards, dones

    @staticmethod
    def _soft_update_target_pair(
        source: nn.Module, target: nn.Module, tau: float
    ) -> None:
        """Polyak averaging: target = tau * source + (1 - tau) * target."""
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    @staticmethod
    def _compute_grad_norm(*modules: nn.Module) -> float:
        """Compute total gradient L2 norm across modules (for logging)."""
        total_norm_sq = 0.0
        for m in modules:
            for p in m.parameters():
                if p.grad is not None:
                    total_norm_sq += p.grad.data.norm(2).item() ** 2
        return total_norm_sq ** 0.5

    @abc.abstractmethod
    def train_step(self, batch: TransitionBatch) -> TrainMetrics:
        """Perform a single training step on a batch of transitions."""

    @abc.abstractmethod
    def get_action_values(
        self, states: list[str], actions: list[str]
    ) -> torch.Tensor:
        """Compute Q-values for state-action pairs."""

    def train(
        self,
        num_steps: int = 1000,
        batch_size: int = 64,
        log_interval: int = 100,
    ) -> list[TrainMetrics]:
        """Run offline training loop."""
        all_metrics = []
        for step in range(num_steps):
            batch = self.replay_buffer.sample_transitions(batch_size)
            metrics = self.train_step(batch)
            all_metrics.append(metrics)

            if (step + 1) % log_interval == 0:
                avg_loss = sum(m.loss for m in all_metrics[-log_interval:]) / log_interval
                logger.info("Step %d/%d: avg_loss=%.4f", step + 1, num_steps, avg_loss)

        return all_metrics

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Save model parameters."""

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """Load model parameters."""
