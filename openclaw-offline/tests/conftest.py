"""
Shared test fixtures for openclaw-offline integration tests.
"""

import json
import os
import sys

import pytest

# ---------- Mock slime types ----------

class _MockStatus:
    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"
    FAILED = "failed"


class MockSample:
    """Minimal mock of slime.utils.types.Sample."""
    Status = _MockStatus

    def __init__(self):
        self.group_index = 0
        self.index = 0
        self.prompt = ""
        self.tokens = []
        self.response = ""
        self.response_length = 0
        self.label = ""
        self.loss_mask = None
        self.weight_versions = []
        self.rollout_log_probs = None
        self.multimodal_inputs = None
        self.remove_sample = False
        self.reward = 0.0
        self.metadata = {}
        self.status = _MockStatus.COMPLETED


class MockRolloutFnTrainOutput:
    def __init__(self, samples, metrics):
        self.samples = samples
        self.metrics = metrics


class MockRolloutFnEvalOutput:
    def __init__(self, data, metrics):
        self.data = data
        self.metrics = metrics


# Install mocks before any test imports the integration module
def _install_slime_mocks():
    sys.modules.setdefault("slime", type(sys)("slime"))
    sys.modules.setdefault("slime.rollout", type(sys)("slime.rollout"))
    sys.modules.setdefault("slime.rollout.base_types", type(sys)("slime.rollout.base_types"))
    sys.modules.setdefault("slime.utils", type(sys)("slime.utils"))
    sys.modules.setdefault("slime.utils.types", type(sys)("slime.utils.types"))

    sys.modules["slime.rollout.base_types"].RolloutFnTrainOutput = MockRolloutFnTrainOutput
    sys.modules["slime.rollout.base_types"].RolloutFnEvalOutput = MockRolloutFnEvalOutput
    sys.modules["slime.utils.types"].Sample = MockSample

_install_slime_mocks()

# Ensure paths
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_OPENCLAW_OFFLINE_DIR = os.path.dirname(_TEST_DIR)
_OFFLINE_RL_DIR = os.path.join(os.path.dirname(_OPENCLAW_OFFLINE_DIR), "offline-rl")

for p in [_OPENCLAW_OFFLINE_DIR, _OFFLINE_RL_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------- Shared fixtures ----------

@pytest.fixture
def trajectory_jsonl(tmp_path):
    """Create a small trajectory JSONL file with 5 trajectories × 3 steps."""
    data = []
    for t in range(5):
        steps = [
            {
                "step_idx": s,
                "action": "click button_{}".format(s),
                "response": "screen after step {}".format(s),
                "reward": 1.0 if s == 2 else 0.0,
                "done": s == 2,
                "info": {},
            }
            for s in range(3)
        ]
        traj = {
            "trajectory_id": "traj_{}".format(t),
            "instruction": "Complete task {}".format(t),
            "domain": "test",
            "example_id": "ex_{}".format(t),
            "outcome_reward": 1.0,
            "eval_score": 1.0,
            "num_steps": 3,
            "status": "completed",
            "source": "test",
            "steps": steps,
            "metadata": {"env_type": "osworld"},
        }
        data.append(json.dumps(traj))

    path = str(tmp_path / "test_traj.jsonl")
    with open(path, "w") as f:
        f.write("\n".join(data))
    return path


@pytest.fixture
def mock_args(trajectory_jsonl):
    """Create a mock args namespace for rollout functions."""

    class Args:
        offline_trajectory_store = trajectory_jsonl
        rollout_max_response_len = 128
        rollout_batch_size = 4

    return Args()


@pytest.fixture
def weight_file(tmp_path):
    """Create a weights JSON file mapping traj_id:step_idx → advantage."""
    weights = {}
    for t in range(5):
        for s in range(3):
            weights["traj_{}:{}".format(t, s)] = float(t * 3 + s) * 0.1
    path = str(tmp_path / "weights.json")
    with open(path, "w") as f:
        json.dump(weights, f)
    return path


@pytest.fixture(autouse=True)
def cleanup_env_vars():
    """Clean up OFFLINE_* env vars after each test."""
    yield
    for key in list(os.environ.keys()):
        if key.startswith("OFFLINE_"):
            os.environ.pop(key, None)
