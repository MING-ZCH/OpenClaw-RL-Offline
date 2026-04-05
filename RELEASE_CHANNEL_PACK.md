# Release Channel Pack

This document provides bilingual release copy for OpenClaw-RL-Offline across GitHub, X/Twitter, and Chinese-language community channels. The messaging is intentionally explicit that this project is built on top of OpenClaw-RL.

Repository URL: https://github.com/MING-ZCH/OpenClaw-RL-Offline

## Positioning Guardrails

- Present the repository as an offline extension of OpenClaw-RL, not as a disconnected new framework.
- Emphasize the real replay/data/bridge path and lightweight executable baselines.
- Avoid claiming that the lightweight baselines replace full Qwen3-VL large-scale multimodal training.
- Make it clear that large-scale training still depends on the upstream slime runtime and proper benchmark environments.

## One-Line Project Summary

### English

OpenClaw-RL-Offline is an offline-oriented fork of OpenClaw-RL that makes benchmark trajectory replay, lightweight offline RL baselines, and slime-compatible offline fine-tuning easier to validate and publish.

### 中文

OpenClaw-RL-Offline 是一个基于 OpenClaw-RL 的离线强化学习分支，重点把 benchmark 轨迹回放、轻量 offline RL baseline，以及兼容 slime 的离线微调链路整理成更容易验证和发布的工作流。

## GitHub Release Notes

### Suggested Release Title

- English: OpenClaw-RL-Offline: benchmark-aware offline replay and fine-tuning
- 中文: OpenClaw-RL-Offline：面向 benchmark 回放与离线微调的 OpenClaw 扩展

### English Version

OpenClaw-RL-Offline is now available as an offline-oriented fork of OpenClaw-RL.

This release turns the offline side of the OpenClaw workflow into a clearer and more auditable stack:

- benchmark trajectory collection into replayable JSONL stores;
- lightweight offline RL baselines including IQL, CQL, AWAC, and Off-Policy GRPO;
- replay-aware GRPO support that uses behavior-policy log-probs when datasets provide them;
- and an openclaw-offline bridge that feeds offline trajectories back into the original slime training flow.

This repository is built on top of OpenClaw-RL rather than replacing it. The goal is to make offline agent RL workflows easier to inspect, validate on CPU, and extend for publication-oriented research.

What is included in this release:

- bilingual root documentation;
- workflow diagrams and entry-point tables;
- an implementation-status document that separates real implemented features from intentional lightweight approximations;
- Windows-friendly collection and launch wrappers;
- and validated smoke-tested offline training paths.

Important scope note:

The lightweight offline baselines are intended for algorithm validation and replay pipeline testing. Full large-scale multimodal training still depends on the upstream slime runtime, model checkpoints, and proper benchmark environments.

Repository: https://github.com/MING-ZCH/OpenClaw-RL-Offline

### 中文版本

OpenClaw-RL-Offline 现已作为 OpenClaw-RL 的 offline 扩展分支发布。

这个版本把 OpenClaw 工作流中原本较分散的离线部分整理成了更清晰、更可审计的一套链路：

- 将 benchmark 轨迹采集为可回放的 JSONL 数据集；
- 提供 IQL、CQL、AWAC、Off-Policy GRPO 等轻量 offline RL baseline；
- 在数据提供行为策略 log-prob 时，让 GRPO 优先使用真实离线策略概率；
- 通过 openclaw-offline 将离线轨迹重新接回原始 slime 训练流程。

这个仓库建立在 OpenClaw-RL 之上，并不是要替代 upstream。它的目标是让 offline agent RL 的数据、回放、算法验证与发布过程更容易理解、复现和扩展。

本次发布包含：

- 中英文仓库总览文档；
- 工作流图和入口选择表；
- 明确区分“真实已实现能力”与“有意保持轻量近似”的实现状态文档；
- 更友好的 Windows 采集与启动脚本入口；
- 以及经过 smoke test 验证的离线训练路径。

范围说明：

这里的轻量 offline baseline 主要用于算法验证和回放链路测试，并不等价于完整的大规模多模态训练。真正的大规模训练仍然依赖 upstream slime runtime、模型权重以及相应 benchmark 环境。

仓库地址：https://github.com/MING-ZCH/OpenClaw-RL-Offline

## GitHub Short Announcement

### English

Released OpenClaw-RL-Offline, an offline fork built on top of OpenClaw-RL.

It adds a clearer replay stack for benchmark trajectory collection, lightweight offline RL baselines, behavior-policy-aware Off-Policy GRPO, and a slime-compatible offline fine-tuning bridge.

Repo: https://github.com/MING-ZCH/OpenClaw-RL-Offline

### 中文

已发布 OpenClaw-RL-Offline。这是一个基于 OpenClaw-RL 的 offline 扩展分支，重点补全了 benchmark 轨迹采集、离线回放、轻量 offline RL baseline、以及兼容 slime 的离线微调桥接。

仓库地址：https://github.com/MING-ZCH/OpenClaw-RL-Offline

## X/Twitter Thread

### English Thread

1. We released OpenClaw-RL-Offline, an offline-oriented fork built on top of OpenClaw-RL. The focus is simple: make offline agent RL workflows around benchmark trajectories easier to validate and extend.

2. Instead of treating offline support as scattered scripts, this repo makes the path explicit: collect benchmark trajectories, replay them through a reusable data layer, train lightweight offline RL baselines, and feed the data back into slime-based fine-tuning.

3. The current stack includes TrajectoryStore, ReplayBuffer, prioritized replay, IQL, CQL, AWAC, and Off-Policy GRPO. When replay data includes behavior-policy log-probs, GRPO uses them directly for a more faithful off-policy ratio.

4. The repo also includes mock validation paths for OSWorld, AndroidWorld, WebArena, and AlfWorld, plus Windows-friendly collection and launch wrappers for local development.

5. We are careful not to overclaim: the lightweight baselines are for CPU-friendly validation and research iteration, while full large-scale multimodal training still depends on the upstream slime runtime and benchmark environments.

6. If you work on offline agent RL around the OpenClaw ecosystem, this should be a cleaner starting point for replay-based experiments and publication-facing documentation.

Repo: https://github.com/MING-ZCH/OpenClaw-RL-Offline

### 中文线程

1. 我们发布了 OpenClaw-RL-Offline。这是一个基于 OpenClaw-RL 的 offline 分支，目标很直接：让围绕 benchmark 轨迹的 offline agent RL 工作流更容易验证和扩展。

2. 它不再把 offline 支持散落成若干脚本，而是把链路明确拆开：采集 benchmark 轨迹，进入可复用的数据层，训练轻量 offline RL baseline，再把数据接回 slime 离线微调。

3. 当前栈已经包含 TrajectoryStore、ReplayBuffer、优先级采样、IQL、CQL、AWAC、Off-Policy GRPO。若数据里有行为策略 log-prob，GRPO 会优先使用真实离线策略概率来计算更可信的 off-policy ratio。

4. 仓库还提供了 OSWorld、AndroidWorld、WebArena、AlfWorld 的 mock 验证路径，以及更友好的 Windows 采集与启动脚本入口。

5. 我们也明确控制表述边界：这些轻量 baseline 主要服务于 CPU 友好的验证和研究迭代，而不是替代 upstream slime runtime 的完整大规模多模态训练。

6. 如果你在 OpenClaw 生态里做 offline agent RL、回放训练或论文整理，这个仓库会是一个更清晰的起点。

仓库地址：https://github.com/MING-ZCH/OpenClaw-RL-Offline

## Chinese Community Long Post

### 中文长文案

最近我们把一套更偏 offline agent RL 的工作整理成了 OpenClaw-RL-Offline，并对外开源。

这个项目不是凭空新起一个框架，而是建立在 OpenClaw-RL 之上的离线扩展分支。我们想解决的问题也比较明确：在 OpenClaw 这类 agent RL 工作流里，offline 数据采集、轨迹回放、baseline 验证，以及最终接回原始训练栈做离线微调，过去往往分散在不同目录、不同脚本和不同隐含假设里，结果就是别人很难快速判断“到底哪些东西是真的做了，哪些只是占位”。

OpenClaw-RL-Offline 这次重点把这部分链路做成了更容易审查的形态。

第一层是离线数据层。仓库里提供了可回放的 JSONL 轨迹存储、ReplayBuffer、优先级采样，以及兼容 slime 的 replay 数据源接口。

第二层是轻量 offline RL baseline。现在已经可以运行 IQL、CQL、AWAC、Off-Policy GRPO。这里我们刻意把算法实现做成了 CPU 也能验证的版本，方便先检查数据和训练链路是否合理，而不是一开始就把实验门槛抬到大规模多卡环境。

第三层是 offline bridge。openclaw-offline 会把预采集轨迹重新送回原始 slime 训练接口，所以这个仓库并不是自己再造一个玩具 trainer，而是把离线数据真正接回 OpenClaw 体系本身。

这次我们还专门补强了一点比较关键的真实性问题：Off-Policy GRPO 不再只依赖 current/ref policy 的近似比值，当回放数据里存在行为策略 log-prob 时，会优先使用真实行为策略概率来做 off-policy correction。这样它在“离线”这个定义上会更站得住。

为了降低理解成本，仓库现在也补了比较完整的中英文文档，包括：

- 根 README 的工作流图和入口选择表；
- offline-rl 包的数据流图与 replay 数据决策表；
- implementation status 文档，专门区分真实实现与有意保持轻量的部分；
- 一份可直接复用的宣传/发布文案草稿。

当然，我们也没有把话说满。这里的轻量 baseline 不是完整 Qwen3-VL 训练本体，mock benchmark 适配器也不等于真实 benchmark fidelity。真正的大规模多模态训练仍然依赖 upstream slime runtime、模型权重和更完整的环境配置。

如果你更关心的是以下问题：

- 如何把 agent benchmark 数据整理成真正可回放的数据集；
- 如何先在 CPU 上验证 offline RL 思路再进入重训练；
- 如何把离线轨迹重新接回 OpenClaw 的训练栈；
- 如何把一个 fork 仓库写成更适合公开发布和论文复现的形态；

那 OpenClaw-RL-Offline 会比一个笼统的“plus”分支更容易理解，也更方便继续做实验和写材料。

仓库地址：https://github.com/MING-ZCH/OpenClaw-RL-Offline

### English Adaptation

We released OpenClaw-RL-Offline as an offline extension branch built on top of OpenClaw-RL.

The motivation is straightforward: offline data collection, replay-based optimization, baseline validation, and integration back into the original training stack are often spread across scripts and implicit assumptions. That makes it hard for readers to tell which parts are real, tested, and reusable.

This release turns that offline path into a clearer structure:

- a replayable JSONL data layer with TrajectoryStore, ReplayBuffer, prioritized replay, and slime-compatible data access;
- lightweight but executable offline RL baselines including IQL, CQL, AWAC, and Off-Policy GRPO;
- and an offline bridge that routes replayed trajectories back into the original slime training interfaces.

We also improved the authenticity of the replay-based GRPO path. When datasets include behavior-policy log-probs, the trainer now uses those values directly instead of relying only on a current-versus-reference approximation.

The repository now includes bilingual documentation, workflow diagrams, entry-point tables, implementation-boundary notes, and reusable promotion materials.

We still draw a clear boundary: these lightweight baselines are meant for CPU-friendly validation and research iteration, not as a claim that they replace full large-scale multimodal training. That larger training path still depends on the upstream slime runtime, model checkpoints, and proper benchmark environments.

## Figure Caption Pack

### Figure 1

- English: From benchmark trajectory collection to replay-based offline fine-tuning in the OpenClaw stack.
- 中文：从 benchmark 轨迹采集到 OpenClaw 体系内的回放式离线微调。

### Figure 2

- English: What is real today: offline data plane, lightweight baselines, and slime bridge.
- 中文：当前已经真实实现的部分：离线数据层、轻量 baseline 与 slime 桥接。

### Figure 3

- English: Choosing the right entry point for collection, baseline training, weight generation, and full offline tuning.
- 中文：如何在采集、baseline 训练、权重生成与完整离线微调之间选择正确入口。

### Figure 4

- English: Replay-aware Off-Policy GRPO prefers behavior-policy log-probs when datasets provide them.
- 中文：当数据集提供行为策略 log-prob 时，回放式 Off-Policy GRPO 会优先使用这些真实概率。

## Tagline Options

### English

- Offline replay made explicit for the OpenClaw ecosystem.
- A clearer offline data-and-fine-tuning layer on top of OpenClaw-RL.
- Benchmark-aware offline agent RL built around replay, validation, and reuse.

### 中文

- 面向 OpenClaw 生态的离线回放训练层。
- 基于 OpenClaw-RL 的更清晰 offline 数据与微调工作流。
- 围绕回放、验证与复用构建的 benchmark-aware offline agent RL。

## Suggested Hashtags

- English: #OfflineRL #AgentRL #LLM #ReinforcementLearning #OpenClaw #Benchmarking
- 中文: #OfflineRL #AgentRL #强化学习 #大模型 #智能体 #OpenClaw