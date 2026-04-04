"""
Custom loss function for offline RL policy extraction.

Uses IQL/CQL-computed advantage weights to re-weight the LLM's log-probabilities
during fine-tuning, enabling proper offline policy extraction.

Usage in shell script:
    --loss-type custom_loss
    --custom-loss-function-path offline_loss.advantage_weighted_loss_function

Environment variables:
    OFFLINE_WEIGHT_PATH: Path to pre-computed advantage weights (JSON/numpy)
    OFFLINE_WEIGHT_TEMPERATURE: Temperature for advantage weighting (default: 3.0)

This loss implements advantage-weighted regression:
    L = -Σ w_i * log π(a_i | s_i)
where w_i = exp(β * A_i) / Σ exp(β * A_j)
and A_i are advantages from a pre-trained IQL/CQL critic.
"""

import json
import logging
import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_cached_weights = None


def _load_advantage_weights() -> dict:
    """Load pre-computed advantage weights from disk."""
    global _cached_weights

    if _cached_weights is not None:
        return _cached_weights

    weight_path = os.environ.get("OFFLINE_WEIGHT_PATH")
    if weight_path and os.path.exists(weight_path):
        with open(weight_path) as f:
            _cached_weights = json.load(f)
        logger.info("Loaded %d advantage weights from %s", len(_cached_weights), weight_path)
    else:
        _cached_weights = {}
        logger.warning("No advantage weights found at %s, using uniform weights", weight_path)

    return _cached_weights


def _get_weight_for_sample(sample, temperature: float = 3.0) -> float:
    """Get advantage weight for a sample."""
    weights = _load_advantage_weights()

    # Look up by trajectory_id + step_idx
    key = None
    if hasattr(sample, "metadata") and sample.metadata:
        traj_id = sample.metadata.get("trajectory_id", "")
        step_idx = sample.metadata.get("step_idx", 0)
        key = "%s:%s" % (traj_id, step_idx)

    if key and key in weights:
        advantage = weights[key]
        # Clamp the exponent for numerical stability (exp(10) ~ 22026)
        clamped = max(min(temperature * advantage, 10.0), -10.0)
        return float(torch.exp(torch.tensor(clamped)).item())

    # Default: use reward signal as advantage proxy
    reward = sample.reward
    if isinstance(reward, dict):
        reward = reward.get("score", reward.get("step_reward", 0.0))
    elif reward is None:
        reward = 0.0

    return max(float(reward) + 1.0, 0.01)  # Ensure positive weight


def reset_weight_cache():
    """Reset the cached weights. Useful for testing or reconfiguration."""
    global _cached_weights
    _cached_weights = None


def advantage_weighted_loss_function(
    args,          # type: Any
    log_probs,     # type: List[torch.Tensor]
    old_log_probs, # type: List[torch.Tensor]
    advantages,    # type: List[torch.Tensor]
    loss_masks,    # type: List[torch.Tensor]
    samples,       # type: List[Any]
):  # type: (...) -> Tuple[torch.Tensor, Dict[str, float]]
    """
    Advantage-weighted policy loss for offline RL.

    Compatible with slime's custom_loss interface:
        --loss-type custom_loss
        --custom-loss-function-path offline_loss.advantage_weighted_loss_function

    Combines standard PPO clipped surrogate with offline advantage weights.

    Args:
        args: Training arguments
        log_probs: Current policy log-probs per sample [list of (T,) tensors]
        old_log_probs: Old policy log-probs per sample [list of (T,) tensors]
        advantages: Per-token advantages [list of (T,) tensors]
        loss_masks: Per-token loss masks [list of (T,) tensors]
        samples: list of slime Sample objects

    Returns:
        (loss, metrics_dict)
    """
    temperature = float(os.environ.get("OFFLINE_WEIGHT_TEMPERATURE", "3.0"))
    kl_coeff = float(os.environ.get("OFFLINE_KL_COEFF", "0.0"))  # optional KL penalty
    eps_clip = getattr(args, "eps_clip", 0.2) if args else 0.2
    eps_clip_high = getattr(args, "eps_clip_high", 0.28) if args else 0.28

    # Handle empty input
    if isinstance(log_probs, (list, tuple)) and len(log_probs) == 0:
        dummy = torch.tensor(0.0, requires_grad=True)
        return dummy, {"offline_loss/pg_loss": 0.0, "offline_loss/clip_frac": 0.0,
                       "offline_loss/clipped_ratio": 0.0, "offline_loss/mean_weight": 0.0,
                       "offline_loss/n_tokens": 0.0}

    total_loss = torch.tensor(0.0, device=log_probs[0].device, requires_grad=True)
    total_tokens = 0
    clip_fracs = []
    offline_weights_used = []

    for i, (lp, olp, adv, mask) in enumerate(zip(log_probs, old_log_probs, advantages, loss_masks)):
        if mask.sum() == 0:
            continue

        # Offline advantage weight for this sample
        sample = samples[i] if i < len(samples) else None
        offline_weight = _get_weight_for_sample(sample, temperature) if sample else 1.0
        offline_weights_used.append(offline_weight)

        # PPO clipped surrogate
        ratio = torch.exp(lp - olp)
        surr1 = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip_high)
        surr2 = clipped * adv

        per_token_loss = -torch.min(surr1, surr2)

        # Apply loss mask and offline weight
        masked_loss = (per_token_loss * mask).sum()
        n_tokens = mask.sum().clamp(min=1)
        sample_loss = (masked_loss / n_tokens) * offline_weight

        total_loss = total_loss + sample_loss
        total_tokens += n_tokens.item()

        # Track clip fraction
        with torch.no_grad():
            clip_frac = ((ratio - 1.0).abs() > eps_clip).float()
            clip_fracs.append((clip_frac * mask).sum().item() / max(n_tokens.item(), 1))

    # Normalize by number of samples
    n_samples = max(len(log_probs), 1)
    total_loss = total_loss / n_samples

    # Also expose clipped_ratio for backward compat with tests
    clip_frac_val = sum(clip_fracs) / max(len(clip_fracs), 1)
    mean_weight = sum(offline_weights_used) / max(len(offline_weights_used), 1)
    metrics = {
        "offline_loss/pg_loss": total_loss.item(),
        "offline_loss/clip_frac": clip_frac_val,
        "offline_loss/clipped_ratio": clip_frac_val,
        "offline_loss/mean_weight": mean_weight,
        "offline_loss/n_tokens": float(total_tokens),
    }

    # Optional KL divergence penalty (matching upstream pattern)
    if kl_coeff > 0 and total_tokens > 0:
        kl_divs = []
        for lp, olp, mask in zip(log_probs, old_log_probs, loss_masks):
            if mask.sum() == 0:
                continue
            # Approximate KL: E[log(pi/pi_old)] = E[lp - olp]
            approx_kl = ((lp - olp) * mask).sum() / mask.sum().clamp(min=1)
            kl_divs.append(approx_kl)
        if kl_divs:
            mean_kl = torch.stack(kl_divs).mean()
            total_loss = total_loss + kl_coeff * mean_kl
            metrics["offline_loss/kl_penalty"] = mean_kl.item()
            metrics["offline_loss/kl_coeff"] = kl_coeff

    return total_loss, metrics
