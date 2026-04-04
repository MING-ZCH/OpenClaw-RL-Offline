"""
Quick Start Example: Offline RL Training Pipeline

This script demonstrates the complete pipeline:
1. Collect trajectories from mock environments
2. Store and inspect trajectory data
3. Train multiple offline RL algorithms
4. Compare algorithm performance

Runs entirely on CPU with no external dependencies.
Expected runtime: ~15 seconds on an 8G machine.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from offline_rl.data import Step, Trajectory, TrajectoryStore, ReplayBuffer
from offline_rl.envs import MockOSWorldAdapter, MockAndroidWorldAdapter, MockWebArenaAdapter


def collect_trajectories(adapter, store, num_episodes=10):
    """Collect trajectories from a mock environment adapter."""
    tasks = adapter.get_task_configs()
    collected = 0

    for i in range(num_episodes):
        task = tasks[i % len(tasks)]
        lease = adapter.allocate(f"ep_{i}")
        adapter.reset(lease["lease_id"])

        steps = []
        done = False
        step_idx = 0

        while not done and step_idx < task.max_steps:
            # Use target actions if available, otherwise generic action
            if step_idx < len(task.target_actions):
                action = task.target_actions[step_idx]
            else:
                action = "wait(1)"

            result = adapter.step(lease["lease_id"], action)
            done = result["done"]
            steps.append(Step(
                step_idx=step_idx,
                action=action,
                response=f"Observation at step {step_idx}",
                reward=0.0,
                done=done,
            ))
            step_idx += 1

        score = adapter.evaluate(lease["lease_id"])["score"]
        adapter.close(lease["lease_id"])

        traj = Trajectory(
            trajectory_id=f"{adapter.BENCHMARK_NAME}_ep{i}",
            domain=task.domain,
            example_id=task.task_id,
            instruction=task.instruction,
            steps=steps,
            outcome_reward=1.0 if score > 0.5 else -1.0,
            eval_score=score,
            num_steps=len(steps),
            status="completed" if score > 0.5 else "failed",
            source="mock",
        )
        store.append(traj)
        collected += 1

    return collected


def main():
    import tempfile
    tmpdir = tempfile.mkdtemp()
    store_path = os.path.join(tmpdir, "trajectories.jsonl")

    # ================================================================
    # Step 1: Collect trajectories from multiple environments
    # ================================================================
    print("=" * 60)
    print("Step 1: Collecting trajectories from mock environments")
    print("=" * 60)

    store = TrajectoryStore(store_path)
    total = 0

    for AdapterClass in [MockOSWorldAdapter, MockAndroidWorldAdapter, MockWebArenaAdapter]:
        adapter = AdapterClass()
        info = adapter.get_benchmark_info()
        n = collect_trajectories(adapter, store, num_episodes=10)
        total += n
        print(f"  {info['name']}: collected {n} episodes")

    print(f"\nTotal: {total} trajectories")

    # ================================================================
    # Step 2: Inspect trajectory data
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 2: Trajectory statistics")
    print("=" * 60)

    stats = store.stats()
    print(f"  Total trajectories: {stats['total_trajectories']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Average steps: {stats['avg_steps']:.1f}")
    print(f"  Domains: {list(stats['domains'].keys())}")

    # ================================================================
    # Step 3: Load into ReplayBuffer
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 3: Loading into ReplayBuffer")
    print("=" * 60)

    buffer = ReplayBuffer(store=store, max_trajectories=5000, seed=42)
    loaded = buffer.load_from_store()
    print(f"  Loaded {loaded} trajectories")
    print(f"  Total transitions: {len(buffer)}")
    print(f"  Buffer stats: {buffer.stats()}")

    # ================================================================
    # Step 4: Train offline RL algorithms
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 4: Training offline RL algorithms")
    print("=" * 60)

    from offline_rl.algorithms import IQL, CQL, AWAC, OffPolicyGRPO

    num_steps = 50
    batch_size = min(16, len(buffer))
    results = {}

    # --- IQL ---
    print("\n  [IQL] Training...")
    iql = IQL(replay_buffer=buffer, state_dim=64, action_dim=64, hidden_dim=64,
              lr=3e-4, device="cpu")
    iql_metrics = iql.train(num_steps=num_steps, batch_size=batch_size, log_interval=num_steps)
    final_loss = iql_metrics[-1].loss
    results["IQL"] = final_loss
    print(f"  [IQL] Final loss: {final_loss:.4f}")
    print(f"  [IQL] Extra: {iql_metrics[-1].extra}")

    # --- CQL ---
    print("\n  [CQL] Training...")
    cql = CQL(replay_buffer=buffer, state_dim=64, action_dim=64, hidden_dim=64,
              alpha=0.5, lr=3e-4, device="cpu")
    cql_metrics = cql.train(num_steps=num_steps, batch_size=batch_size, log_interval=num_steps)
    final_loss = cql_metrics[-1].loss
    results["CQL"] = final_loss
    print(f"  [CQL] Final loss: {final_loss:.4f}")
    print(f"  [CQL] Extra: {cql_metrics[-1].extra}")

    # --- AWAC ---
    print("\n  [AWAC] Training...")
    awac = AWAC(replay_buffer=buffer, state_dim=64, action_dim=64, hidden_dim=64,
                lam=1.0, lr=3e-4, device="cpu")
    awac_metrics = awac.train(num_steps=num_steps, batch_size=batch_size, log_interval=num_steps)
    final_loss = awac_metrics[-1].loss
    results["AWAC"] = final_loss
    print(f"  [AWAC] Final loss: {final_loss:.4f}")
    print(f"  [AWAC] Extra: {awac_metrics[-1].extra}")

    # --- Off-Policy GRPO ---
    print("\n  [GRPO] Training...")
    grpo = OffPolicyGRPO(replay_buffer=buffer, state_dim=64, action_dim=64, hidden_dim=64,
                         clip_ratio=0.2, kl_coeff=0.01, n_policy_updates=2,
                         lr=1e-4, device="cpu")
    grpo_metrics = grpo.train(num_steps=num_steps, batch_size=batch_size, log_interval=num_steps)
    final_loss = grpo_metrics[-1].loss
    results["GRPO"] = final_loss
    print(f"  [GRPO] Final loss: {final_loss:.4f}")
    print(f"  [GRPO] Extra: {grpo_metrics[-1].extra}")

    # ================================================================
    # Step 5: Compare results
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 5: Algorithm comparison")
    print("=" * 60)
    print(f"\n  {'Algorithm':<10} {'Final Loss':>12}")
    print(f"  {'-'*10} {'-'*12}")
    for algo_name, loss in results.items():
        print(f"  {algo_name:<10} {loss:>12.4f}")

    # ================================================================
    # Step 6: Demonstrate Q-value evaluation
    # ================================================================
    print("\n" + "=" * 60)
    print("Step 6: Q-value evaluation")
    print("=" * 60)

    test_states = ["Task: Open Chrome and search for something\nStep 0: click(540,50)"]
    test_actions = ["type('hello world')"]

    for name, algo in [("IQL", iql), ("CQL", cql), ("AWAC", awac), ("GRPO", grpo)]:
        values = algo.get_action_values(test_states, test_actions)
        print(f"  {name}: Q/logprob = {values.item():.4f}")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("Quick start complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
