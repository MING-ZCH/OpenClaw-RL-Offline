"""
TrajectoryStore: Persistent storage for agent trajectories.

Compatible with gui-rl's trajectory.json output format. Stores trajectories
as JSONL for memory-efficient streaming access on low-memory machines.

Format per line:
{
    "trajectory_id": str,
    "domain": str,
    "example_id": str,
    "instruction": str,
    "steps": [
        {
            "step_idx": int,
            "observation_file": str | None,
            "action": str,
            "response": str,
            "reward": float,
            "done": bool,
        }
    ],
    "outcome_reward": float,   # 1.0 or -1.0 (gui-rl binary)
    "eval_score": float,       # 0.0 or 1.0 (OSWorld evaluator)
    "num_steps": int,
    "status": str,             # "completed" | "failed" | "truncated" | "aborted"
    "source": str,             # "gui-rl" | "manual" | ...
}
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class Step:
    step_idx: int
    action: str
    response: str = ""
    reward: float = 0.0
    done: bool = False
    observation_file: Optional[str] = None
    info: dict = field(default_factory=dict)


@dataclass
class Trajectory:
    trajectory_id: str
    domain: str
    example_id: str
    instruction: str
    steps: list[Step]
    outcome_reward: float
    eval_score: float
    num_steps: int
    status: str = "completed"
    source: str = "manual"
    metadata: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.eval_score > 0.5

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        steps = [Step(**s) for s in d.pop("steps", [])]
        return cls(steps=steps, **d)


class TrajectoryStore:
    """
    Append-only JSONL trajectory store with streaming read support.

    Designed for 8G CPU machines: never loads full dataset into memory.
    Supports filtering by domain, success status, etc.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._count_cache: Optional[int] = None

    def append(self, trajectory: Trajectory) -> None:
        """Append a single trajectory to the store."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trajectory.to_dict(), ensure_ascii=False) + "\n")
        self._count_cache = None

    def append_batch(self, trajectories: list[Trajectory]) -> None:
        """Append multiple trajectories at once."""
        with open(self.path, "a", encoding="utf-8") as f:
            for traj in trajectories:
                f.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
        self._count_cache = None

    def iter_trajectories(
        self,
        domain: Optional[str] = None,
        success_only: bool = False,
        max_count: Optional[int] = None,
    ) -> Iterator[Trajectory]:
        """
        Stream trajectories from disk. Memory-efficient: yields one at a time.

        Args:
            domain: Filter by domain name (e.g. "os", "chrome", "vlc")
            success_only: Only yield successful trajectories
            max_count: Maximum number of trajectories to yield
        """
        if not self.path.exists():
            return

        count = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON line in %s", self.path)
                    continue

                traj = Trajectory.from_dict(d)

                if domain and traj.domain != domain:
                    continue
                if success_only and not traj.success:
                    continue

                yield traj
                count += 1
                if max_count and count >= max_count:
                    return

    def count(self) -> int:
        """Count total trajectories (cached after first call)."""
        if self._count_cache is not None:
            return self._count_cache
        if not self.path.exists():
            self._count_cache = 0
            return 0
        count = 0
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        self._count_cache = count
        return count

    def stats(self) -> dict[str, Any]:
        """Compute basic statistics over the store."""
        total = 0
        success = 0
        domains: dict[str, int] = {}
        total_steps = 0
        for traj in self.iter_trajectories():
            total += 1
            if traj.success:
                success += 1
            domains[traj.domain] = domains.get(traj.domain, 0) + 1
            total_steps += traj.num_steps
        return {
            "total_trajectories": total,
            "successful": success,
            "success_rate": success / total if total > 0 else 0.0,
            "domains": domains,
            "avg_steps": total_steps / total if total > 0 else 0.0,
        }

    @classmethod
    def import_from_gui_rl_results(
        cls,
        results_dir: str | Path,
        output_path: str | Path,
    ) -> "TrajectoryStore":
        """
        Import gui-rl result directories into a TrajectoryStore.

        Expected structure:
        results_dir/
          pyautogui/screenshot/<model>/<domain>/<example_id>/<run>/
            trajectory.json
            result.txt
            traj.jsonl
        """
        store = cls(output_path)
        results_dir = Path(results_dir)
        imported = 0

        # Walk the results directory tree
        for result_txt in results_dir.rglob("result.txt"):
            run_dir = result_txt.parent
            try:
                eval_score = float(result_txt.read_text().strip())
            except (ValueError, OSError):
                continue

            traj_json = run_dir / "trajectory.json"
            traj_jsonl = run_dir / "traj.jsonl"

            # Parse the directory path to extract domain/example_id
            # Pattern: .../pyautogui/screenshot/<model>/<domain>/<example_id>/<run>/
            parts = run_dir.parts
            domain = "unknown"
            example_id = "unknown"
            try:
                # Find "screenshot" in path, domain is 2 levels after model
                for i, p in enumerate(parts):
                    if p == "screenshot" and i + 3 < len(parts):
                        domain = parts[i + 2]
                        example_id = parts[i + 3]
                        break
            except (IndexError, ValueError):
                pass

            steps = []
            instruction = ""

            # Try trajectory.json first (richer format)
            if traj_json.exists():
                try:
                    with open(traj_json, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for i, trace in enumerate(data.get("trajectory", [])):
                        steps.append(Step(
                            step_idx=trace.get("step_idx", i),
                            action=str(trace.get("actions", [""])[0] if trace.get("actions") else ""),
                            response=str(trace.get("response", "")),
                            reward=0.0,
                            done=(i == len(data.get("trajectory", [])) - 1),
                        ))
                except (json.JSONDecodeError, OSError):
                    pass

            # Fallback to traj.jsonl
            if not steps and traj_jsonl.exists():
                try:
                    with open(traj_jsonl, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            entry = json.loads(line)
                            if "Error" in entry:
                                continue
                            steps.append(Step(
                                step_idx=entry.get("step_num", 0),
                                action=str(entry.get("action", "")),
                                response=str(entry.get("response", "")),
                                reward=float(entry.get("reward", 0.0)),
                                done=bool(entry.get("done", False)),
                            ))
                except (json.JSONDecodeError, OSError):
                    pass

            if not steps:
                continue

            outcome_reward = 1.0 if eval_score > 0.5 else -1.0
            status = "completed" if eval_score > 0.5 else "failed"

            traj = Trajectory(
                trajectory_id=str(uuid.uuid4()),
                domain=domain,
                example_id=example_id,
                instruction=instruction,
                steps=steps,
                outcome_reward=outcome_reward,
                eval_score=eval_score,
                num_steps=len(steps),
                status=status,
                source="gui-rl",
            )
            store.append(traj)
            imported += 1

        logger.info("Imported %d trajectories from %s", imported, results_dir)
        return store


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TrajectoryStore CLI")
    sub = parser.add_subparsers(dest="command")

    imp = sub.add_parser("import", help="Import gui-rl results")
    imp.add_argument("--import-dir", required=True)
    imp.add_argument("--output", required=True)

    stats_cmd = sub.add_parser("stats", help="Show store statistics")
    stats_cmd.add_argument("--path", required=True)

    args = parser.parse_args()
    if args.command == "import":
        store = TrajectoryStore.import_from_gui_rl_results(args.import_dir, args.output)
        print(f"Imported. Stats: {store.stats()}")
    elif args.command == "stats":
        store = TrajectoryStore(args.path)
        print(json.dumps(store.stats(), indent=2))
