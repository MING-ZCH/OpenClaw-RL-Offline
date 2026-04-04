# Promotion / PR Draft

This file contains bilingual publicity copy for OpenClaw-RL-Offline. The messaging is intentionally explicit that the project is based on OpenClaw-RL rather than positioned as a disconnected new method.

## Chinese Short Version

我们开源了 OpenClaw-RL-Offline，一个基于 OpenClaw-RL 的 offline fork，重点把 benchmark 轨迹采集、offline replay、以及基于 slime 的离线微调链路系统化整理出来。

它不是简单改名或堆脚本，而是把几件过去不够清晰的事情做实了：

- 支持 OSWorld、AndroidWorld、WebArena、AlfWorld 的统一离线采集与回放流程；
- 提供 IQL、CQL、AWAC、Off-Policy GRPO 等轻量 offline RL baseline；
- 可以把预采集轨迹重新接回原始 OpenClaw / slime 训练路径；
- 对实现边界做了更明确说明，区分“真实已实现能力”和“依赖外部运行时的部分”。

如果你关注的是 agent RL 在 benchmark 数据上的 offline 训练与复现，这个仓库会比直接从原始目录结构里拼装 workflow 更容易上手。

同时需要明确说明：OpenClaw-RL-Offline 是建立在 OpenClaw-RL 之上的 offline 扩展分支，完整大规模训练能力依然依赖原始 slime runtime、模型权重以及 Linux 风格多 GPU 环境。

## Chinese Long Version

OpenClaw-RL-Offline is our offline-oriented fork of OpenClaw-RL.

The main goal is not to replace the original OpenClaw project, but to make the offline side of the workflow first-class and easier to validate:

- benchmark trajectory collection into replayable JSONL stores,
- lightweight offline RL baselines for policy/value-side validation,
- and replay-based fine-tuning through the original slime training interfaces.

Compared with a generic “plus” style fork, the new name reflects the real differentiator of the repository: offline data, offline replay, and offline policy optimization.

What is already included:

- a real offline data layer with `TrajectoryStore`, `ReplayBuffer`, and prioritized replay;
- lightweight but executable implementations of IQL, CQL, AWAC, and Off-Policy GRPO;
- support for behavior-policy log-probs in replayed GRPO datasets when they are available;
- CPU-friendly validation workflows for OSWorld, AndroidWorld, WebArena, and AlfWorld;
- a bridge back into the original OpenClaw / slime trainer for full offline fine-tuning.

What this repository does not overclaim:

- the lightweight offline baselines are not framed as a replacement for full Qwen3-VL training;
- mock adapters are not marketed as full benchmark execution;
- large-scale training still depends on the upstream runtime stack.

If your goal is to study or build offline agent RL workflows around the OpenClaw ecosystem, OpenClaw-RL-Offline provides a cleaner and more auditable starting point.

## English Short Version

We are releasing OpenClaw-RL-Offline, an offline-oriented fork of OpenClaw-RL focused on benchmark trajectory replay, lightweight offline RL baselines, and slime-compatible offline fine-tuning.

Instead of presenting offline support as scattered scripts, this repo makes the offline path explicit:

- benchmark trajectory collection for OSWorld, AndroidWorld, WebArena, and AlfWorld,
- executable IQL / CQL / AWAC / Off-Policy GRPO baselines,
- and a bridge that replays offline data back into the original OpenClaw training flow.

OpenClaw-RL-Offline is built on top of OpenClaw-RL. It is intended to make offline agent RL workflows easier to validate and publish, while still acknowledging that full large-scale training depends on the upstream slime runtime and model stack.

## English Long Version

OpenClaw-RL-Offline is a benchmark-aware offline fork of OpenClaw-RL.

The goal of this project is straightforward: make offline data collection, replay-based policy optimization, and offline fine-tuning first-class parts of the OpenClaw workflow.

This repository keeps the original OpenClaw method folders for reproducibility, but adds a clearer offline stack around them:

- trajectory collection into replayable JSONL stores,
- lightweight offline RL baselines for algorithmic validation,
- behavior-policy-aware Off-Policy GRPO when replay data includes log-probs,
- and an `openclaw-offline` bridge that plugs offline data back into the original slime-based trainer.

Just as importantly, the repo now documents its boundaries more explicitly. The offline algorithms are real and executable, but the lightweight encoders are meant for CPU-friendly validation rather than for claiming immediate full-scale multimodal training. Real benchmark fidelity and large-scale training still depend on the upstream runtime, external benchmark environments, and proper multi-GPU infrastructure.

If you are looking for a practical starting point for offline agent RL research built around the OpenClaw ecosystem, this is the layer we wanted to make easier to understand, reproduce, and extend.