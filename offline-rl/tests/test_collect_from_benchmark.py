"""Tests for multi-benchmark offline trajectory collection."""

from __future__ import annotations

import os
import sys


SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import collect_from_benchmark as collect_script


def test_supported_benchmark_adapters_collect_episodes():
    for env_name in ["androidworld", "webarena", "osworld", "alfworld"]:
        adapter = collect_script._get_adapter(env_name)
        trajectory = collect_script._collect_episode_adapter(adapter, "%s-ep" % env_name, agent_success_rate=1.0)
        assert trajectory.source == env_name
        assert trajectory.num_steps >= 1
        assert trajectory.domain
        assert trajectory.instruction


def test_mock_server_collection_path():
    server = collect_script._get_adapter("mock")
    trajectory = collect_script._collect_episode_mock_server(server, "mock-ep", agent_success_rate=0.5)
    assert trajectory.source == "mock"
    assert trajectory.num_steps >= 1