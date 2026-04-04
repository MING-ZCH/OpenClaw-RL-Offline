# OpenClaw-RL-Plus

OpenClaw-RL-Plus is a curated fork of OpenClaw-RL focused on benchmark-aware offline and off-policy reinforcement learning.

It keeps the original online training method folders needed for reproducibility, and adds a cleaner offline stack for:

- benchmark trajectory collection;
- offline policy learning on replayed data;
- slime-compatible offline fine-tuning with pre-collected trajectories.

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Research%20Release-orange.svg)](#status)

## What This Fork Adds

- A standalone [offline-rl](./offline-rl) package with trajectory storage, replay sampling, offline data source adapters, and offline RL algorithms.
- Unified mock benchmark support for OSWorld, AndroidWorld, WebArena, and AlfWorld.
- A dedicated [openclaw-offline](./openclaw-offline) bridge that replays offline trajectories into the original slime training stack.
- Benchmark-specific offline launcher wrappers for OSWorld, AndroidWorld, WebArena, and AlfWorld.
- Windows-friendly PowerShell wrappers for benchmark collection, plus PowerShell forwarding entry points for offline training.
- Documentation rewritten around the components that are actually present in this repository snapshot.

## Repository Scope

This repository currently centers on three use cases:

1. online OpenClaw optimization with the original method folders;
2. GUI-oriented agent training via [gui-rl](./gui-rl);
3. offline benchmark replay and offline RL extension work via [offline-rl](./offline-rl) and [openclaw-offline](./openclaw-offline).

Unlike the full upstream project page, this README intentionally avoids documenting folders or demos that are not included in this fork.

## Repository Layout

| Folder | Purpose |
|---|---|
| [openclaw-rl](./openclaw-rl) | Binary-RL training path using next-state reward signals |
| [openclaw-opd](./openclaw-opd) | On-policy distillation training path |
| [openclaw-combine](./openclaw-combine) | Combined Binary RL + OPD training path |
| [gui-rl](./gui-rl) | GUI agent data generation and environment integration |
| [offline-rl](./offline-rl) | Offline RL package: data layer, benchmark adapters, and replay-based algorithms |
| [openclaw-offline](./openclaw-offline) | slime integration layer for full offline fine-tuning on replayed trajectories |
| [slime](./slime) | Underlying distributed training framework used by the OpenClaw stack |

## Quick Start

### Option A: Run The Original Online Methods

Use the upstream-compatible method folders when you want online OpenClaw optimization.

```bash
cd slime

# Binary RL
bash ../openclaw-rl/run_qwen35_4b_openclaw_rl.sh

# On-policy Distillation
bash ../openclaw-opd/run_qwen35_4b_openclaw_opd.sh

# Combined Binary RL + OPD
bash ../openclaw-combine/run_qwen35_4b_openclaw_combine.sh
```

For method details, see the per-folder READMEs in [openclaw-rl](./openclaw-rl), [openclaw-opd](./openclaw-opd), and [openclaw-combine](./openclaw-combine).

### Option B: Run The Offline Extension Workflow

Use the offline extension when you want benchmark collection, replay-based RL, or full offline fine-tuning.

#### 1. Collect trajectories

```bash
cd offline-rl

python scripts/collect_from_benchmark.py --env osworld --n 100 --output data/osworld_trajs.jsonl
python scripts/collect_from_benchmark.py --env androidworld --n 100 --output data/androidworld_trajs.jsonl
python scripts/collect_from_benchmark.py --env webarena --n 100 --output data/webarena_trajs.jsonl
python scripts/collect_from_benchmark.py --env alfworld --n 100 --output data/alfworld_trajs.jsonl
```

```powershell
cd offline-rl

.\scripts\run_collect_osworld.ps1
.\scripts\run_collect_androidworld.ps1
.\scripts\run_collect_webarena.ps1
.\scripts\run_collect_alfworld.ps1
```

#### 2. Train a lightweight offline baseline directly in offline-rl

```bash
python scripts/train_offline.py --algo iql --data data/osworld_trajs.jsonl --steps 500
python scripts/train_offline.py --algo cql --data data/webarena_trajs.jsonl --steps 500
python scripts/train_offline.py --algo awac --data data/alfworld_trajs.jsonl --steps 500
```

#### 3. Launch full slime-based offline fine-tuning

```bash
cd ../openclaw-offline

bash run_qwen35_4b_osworld_offline_rl.sh
bash run_qwen35_4b_androidworld_offline_rl.sh
bash run_qwen35_4b_webarena_offline_rl.sh
bash run_qwen35_4b_alfworld_offline_rl.sh
```

```powershell
cd ..\openclaw-offline

.\run_qwen35_4b_osworld_offline_rl.ps1
.\run_qwen35_4b_androidworld_offline_rl.ps1
.\run_qwen35_4b_webarena_offline_rl.ps1
.\run_qwen35_4b_alfworld_offline_rl.ps1
```

If you prefer the generic launcher, set `OFFLINE_TRAJECTORY_STORE` yourself and then run either [openclaw-offline/run_qwen35_4b_offline_rl.sh](./openclaw-offline/run_qwen35_4b_offline_rl.sh) or `run_qwen35_4b_offline_rl.ps1`.
The PowerShell launchers forward to WSL first and then fall back to Git Bash when available. Full offline training still requires the same Linux-like multi-GPU runtime expected by slime and upstream OpenClaw-RL.

## Supported Offline Benchmarks

| Benchmark | Mock collection | Task configs | Offline replay wrapper |
|---|---|---|---|
| OSWorld | Yes | Yes | Yes |
| AndroidWorld | Yes | Yes | Yes |
| WebArena | Yes | Yes | Yes |
| AlfWorld | Yes | Yes | Yes |

The mock adapters are designed for CPU validation and repo-level testing. Real benchmark execution still requires the corresponding external packages, simulators, or service environments.

## Recommended Reading Order

1. Start here for the repository overview.
2. Read [offline-rl/README.md](./offline-rl/README.md) for collection, replay, and offline algorithm usage.
3. Read [openclaw-offline/README.md](./openclaw-offline/README.md) for slime-based offline fine-tuning.
4. Use the method-specific READMEs under [openclaw-rl](./openclaw-rl), [openclaw-opd](./openclaw-opd), and [openclaw-combine](./openclaw-combine) only when you need the original online methods.

<a id="status"></a>
## Status

- `offline-rl` CPU test suite is passing in this fork.
- Multi-benchmark collection has been validated for OSWorld, AndroidWorld, WebArena, and AlfWorld.
- Short offline-training smoke runs have been validated on replayed benchmark trajectories.
- Full-scale LLM training still requires the original slime runtime stack, model checkpoints, and a Linux-like multi-GPU environment.

## Notes

- The offline collection entry points are available as direct Python commands, bash wrappers, and PowerShell wrappers.
- The PowerShell offline-training launchers are forwarding entry points; real training still runs through WSL or another Linux-like shell environment.
- This fork intentionally keeps the algorithm folders and file layout close to upstream so existing launch and integration patterns remain recognizable.

## Acknowledgement

OpenClaw-RL-Plus is built on top of the original OpenClaw-RL project from Gen-Verse. This fork focuses on making offline RL and benchmark replay a first-class, easier-to-publish part of the repository without rewriting the upstream method organization.