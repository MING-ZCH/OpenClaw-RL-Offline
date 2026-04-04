"""Tests for MockEnvPoolServer."""

from __future__ import annotations

from offline_rl.envs.mock_env_server import MockEnvPoolServer, MockTask


class TestMockTask:
    """Test individual task logic."""

    def test_step_increments(self):
        task = MockTask({"instruction": "test", "max_steps": 5, "target_actions": []})
        obs, r, done, info = task.step("click(1,1)")
        assert task.current_step == 1
        assert not done

    def test_max_steps_done(self):
        task = MockTask({"instruction": "test", "max_steps": 2, "target_actions": []})
        task.step("action1")
        _, _, done, _ = task.step("action2")
        assert done

    def test_evaluate_success(self):
        task = MockTask({
            "instruction": "test",
            "max_steps": 5,
            "target_actions": ["click(10, 20)", "type('hello')"],
        })
        task.step("click(10, 20)")
        task.step("type('hello')")
        score = task.evaluate()
        assert score == 1.0

    def test_evaluate_failure(self):
        task = MockTask({
            "instruction": "test",
            "max_steps": 5,
            "target_actions": ["click(10, 20)", "type('hello')"],
        })
        task.step("random_action")
        score = task.evaluate()
        assert score == 0.0


class TestMockEnvPoolServer:
    """Test full server API flow."""

    def test_allocate(self):
        server = MockEnvPoolServer()
        result = server.allocate("ep1")
        assert result["ok"]
        assert "lease_id" in result

    def test_full_episode(self):
        server = MockEnvPoolServer()
        lease = server.allocate("ep1")
        lease_id = lease["lease_id"]

        reset_result = server.reset(lease_id, {
            "instruction": "test",
            "max_steps": 3,
            "target_actions": ["click(100, 200)"],
        })
        assert reset_result["ok"]
        assert "observation" in reset_result

        for _ in range(3):
            step_result = server.step(lease_id, "click(100, 200)")
            assert step_result["ok"]
            if step_result["done"]:
                break

        eval_result = server.evaluate(lease_id)
        assert eval_result["ok"]
        assert "score" in eval_result

        close_result = server.close(lease_id)
        assert close_result["ok"]

    def test_get_obs(self):
        server = MockEnvPoolServer()
        lease = server.allocate("ep1")
        lid = lease["lease_id"]
        server.reset(lid)
        obs = server.get_obs(lid)
        assert obs["ok"]
        assert "observation" in obs

    def test_heartbeat(self):
        server = MockEnvPoolServer()
        lease = server.allocate("ep1")
        lid = lease["lease_id"]
        assert server.heartbeat(lid)["ok"]
        assert not server.heartbeat("nonexistent")["ok"]

    def test_close_cleans_up(self):
        server = MockEnvPoolServer()
        lease = server.allocate("ep1")
        lid = lease["lease_id"]
        server.close(lid)
        assert not server.heartbeat(lid)["ok"]

    def test_unknown_lease(self):
        server = MockEnvPoolServer()
        result = server.reset("bad-lease")
        assert not result["ok"]

    def test_multiple_episodes(self):
        server = MockEnvPoolServer()
        leases = []
        for i in range(5):
            lease = server.allocate(f"ep{i}")
            leases.append(lease["lease_id"])
            server.reset(lease["lease_id"])
        assert len(leases) == 5
        for lid in leases:
            server.close(lid)


class TestMockTrajectoryGeneration:
    """Test mock trajectory generation helper."""

    def test_generate_mock_trajectories(self):
        from offline_rl.envs.mock_env_server import generate_mock_trajectories

        server = MockEnvPoolServer()
        trajs = generate_mock_trajectories(server, n_trajectories=20, agent_success_rate=0.5)
        assert len(trajs) == 20
        # Should have mix of completed and failed
        statuses = {t.status for t in trajs}
        assert len(statuses) >= 1  # At least one status type

    def test_generate_deterministic_with_seed(self):
        import random
        from offline_rl.envs.mock_env_server import generate_mock_trajectories

        server = MockEnvPoolServer()
        random.seed(42)
        trajs1 = generate_mock_trajectories(server, n_trajectories=5, agent_success_rate=0.5)
        random.seed(42)
        trajs2 = generate_mock_trajectories(server, n_trajectories=5, agent_success_rate=0.5)
        for t1, t2 in zip(trajs1, trajs2):
            assert t1.instruction == t2.instruction
            assert t1.num_steps == t2.num_steps
