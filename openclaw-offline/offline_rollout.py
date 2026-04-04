"""
Offline rollout function for slime training.

Replaces online environment interaction with pre-collected trajectory replay.
Compatible with slime's --rollout-function-path mechanism.

Usage in shell script:
    --rollout-function-path offline_rollout.generate_rollout_offline
    --disable-rollout-global-dataset

Environment variables:
    OFFLINE_TRAJECTORY_STORE: Path to trajectories.jsonl
    OFFLINE_MODE: "step" | "trajectory" | "dynamic_history" (default: "step")
    OFFLINE_N_SAMPLES_PER_PROMPT: Group size (default: 1)
"""

import logging
import os
import sys
import time

# Ensure the offline-rl package is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OFFLINE_RL_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "offline-rl")
if _OFFLINE_RL_DIR not in sys.path:
    sys.path.insert(0, _OFFLINE_RL_DIR)

from slime.rollout.base_types import RolloutFnTrainOutput, RolloutFnEvalOutput
from slime.utils.types import Sample

from offline_rl.data.trajectory_store import TrajectoryStore
from offline_rl.data.offline_data_source import OfflineDataSource, SampleLite

logger = logging.getLogger(__name__)

_global_data_source = None

_VALID_MODES = {"step", "trajectory", "dynamic_history"}


def reset_data_source():
    """Reset the cached data source. Useful for testing or reconfiguration."""
    global _global_data_source
    _global_data_source = None


def _get_data_source(args) -> OfflineDataSource:
    """Initialize or return cached OfflineDataSource."""
    global _global_data_source

    if _global_data_source is not None:
        return _global_data_source

    store_path = os.environ.get(
        "OFFLINE_TRAJECTORY_STORE",
        getattr(args, "offline_trajectory_store", None),
    )
    if not store_path:
        raise ValueError(
            "No trajectory store specified. Set OFFLINE_TRAJECTORY_STORE env var "
            "or pass --offline-trajectory-store <path>"
        )

    mode = os.environ.get("OFFLINE_MODE", "step")
    if mode not in _VALID_MODES:
        raise ValueError(
            "Invalid OFFLINE_MODE='%s'. Must be one of: %s" % (mode, ", ".join(sorted(_VALID_MODES)))
        )
    n_samples = int(os.environ.get("OFFLINE_N_SAMPLES_PER_PROMPT", "1"))

    store = TrajectoryStore(store_path)
    count = store.count()
    if count == 0:
        raise ValueError(f"Trajectory store is empty: {store_path}")

    _global_data_source = OfflineDataSource(
        store=store,
        mode=mode,
        n_samples_per_prompt=n_samples,
        max_token_length=getattr(args, "rollout_max_response_len", 8192),
        shuffle=True,
    )

    logger.info(
        "Initialized offline data source: %s (%d trajectories, %d samples)",
        store_path, count, len(_global_data_source),
    )
    return _global_data_source


def _sample_lite_to_sample(sl: SampleLite) -> Sample:
    """Convert SampleLite to slime Sample."""
    s = Sample()
    s.group_index = sl.group_index
    s.index = sl.index
    s.prompt = sl.prompt
    s.tokens = sl.tokens
    s.response = sl.response
    s.response_length = sl.response_length or len(sl.response)
    s.label = sl.label
    s.loss_mask = sl.loss_mask
    s.weight_versions = sl.weight_versions or []
    s.rollout_log_probs = sl.rollout_log_probs
    s.multimodal_inputs = sl.multimodal_inputs
    s.remove_sample = sl.remove_sample

    # Reward handling: slime expects float | dict | None
    s.reward = sl.reward

    # Metadata: merge offline-rl specific fields
    s.metadata = dict(sl.metadata) if sl.metadata else {}
    s.metadata["trajectory_id"] = sl.trajectory_id
    s.metadata["step_idx"] = sl.step_idx
    s.metadata["offline_replay"] = True

    # Status mapping
    status_map = {
        "pending": Sample.Status.PENDING,
        "completed": Sample.Status.COMPLETED,
        "truncated": Sample.Status.TRUNCATED,
        "aborted": Sample.Status.ABORTED,
        "failed": Sample.Status.FAILED,
    }
    s.status = status_map.get(sl.status, Sample.Status.COMPLETED)

    return s


def generate_rollout_offline(args, rollout_id, data_buffer, evaluation=False):
    """
    Offline rollout function compatible with slime's training loop.

    Replaces online environment interaction by replaying pre-collected
    trajectories from a TrajectoryStore.

    Signature matches:
        openclaw_rollout.generate_rollout_openclaw(args, rollout_id, data_buffer, evaluation)

    Returns:
        RolloutFnTrainOutput or RolloutFnEvalOutput
    """
    if evaluation:
        # For evaluation, return empty output (offline has no live eval)
        logger.info("[OfflineRollout] Evaluation requested — skipping (no live environment)")
        return RolloutFnEvalOutput(data={}, metrics={"offline/eval_skip": 1.0})

    start = time.time()
    ds = _get_data_source(args)

    # Get samples matching the rollout batch size
    batch_size = getattr(args, "rollout_batch_size", 16)
    groups_lite = ds.get_samples(batch_size)

    # Convert SampleLite → slime Sample
    samples = []  # type: list  # list[list[Sample]]
    total_reward = 0.0
    n_reward_samples = 0

    for group_idx, group in enumerate(groups_lite):
        if not group:
            continue  # skip empty groups
        converted_group = []
        for sample_idx, sl in enumerate(group):
            sl.group_index = group_idx
            sl.index = sample_idx
            s = _sample_lite_to_sample(sl)
            converted_group.append(s)

            # Track reward for metrics
            if isinstance(s.reward, dict):
                r = s.reward.get("score", s.reward.get("step_reward", 0.0))
            elif isinstance(s.reward, (int, float)):
                r = float(s.reward)
            else:
                r = 0.0
            total_reward += r
            n_reward_samples += 1

        samples.append(converted_group)

    elapsed = time.time() - start
    avg_reward = total_reward / max(n_reward_samples, 1)

    metrics = {
        "offline/num_groups": float(len(samples)),
        "offline/avg_reward": avg_reward,
        "offline/replay_time": elapsed,
        "offline/pool_size": float(len(ds)),
        "offline/rollout_id": float(rollout_id) if isinstance(rollout_id, (int, float)) else 0.0,
    }

    logger.info(
        "[OfflineRollout] rollout_id=%s, groups=%d, avg_reward=%.3f, time=%.2fs",
        rollout_id, len(samples), avg_reward, elapsed,
    )

    return RolloutFnTrainOutput(samples=samples, metrics=metrics)
