# OpenClaw-RL-Offline

OpenClaw-RL-Offline 是基于 OpenClaw-RL 的一个离线强化学习分支，核心目标不是重写 upstream，而是把原来分散、隐含的 offline 数据收集、回放训练和基于 slime 的离线微调链路整理成一个更容易验证和发布的仓库版本。

英文总览见 [README.md](./README.md)。离线实现边界说明见 [offline-rl/docs/implementation_status.md](./offline-rl/docs/implementation_status.md)。

## 这个仓库当前真实实现了什么

| 模块 | 当前状态 | 说明 |
|---|---|---|
| `offline-rl` 数据层 | 完整可用 | 包含 `TrajectoryStore`、`ReplayBuffer`、优先级采样、slime 兼容数据源。 |
| `IQL` / `CQL` / `AWAC` | 真实实现 | 都有可运行训练循环、指标输出和 CPU 测试。 |
| `Off-Policy GRPO` | 真实实现 | 在数据里提供行为策略 log-prob 时，会直接使用这些离线概率；旧数据则回退到 ref-policy 近似。 |
| `openclaw-offline` | 真实 bridge | 会把离线轨迹重放回原始 slime 训练接口，而不是单独做一个玩具 trainer。 |
| 多 benchmark 采集包装 | 已实现 | OSWorld、AndroidWorld、WebArena、AlfWorld 都有 mock 适配与脚本。 |

## 这个仓库没有夸大什么

- 轻量 offline baseline 使用的是 CPU 友好的小型文本编码器，不是完整 Qwen3-VL 主干。
- mock benchmark 适配器主要用于 repo 级验证，不等价于真实 benchmark 环境。
- PowerShell 启动脚本只是 Windows 宿主机入口，真正的大规模训练仍然依赖 Linux 风格运行时与 slime 栈。

## 推荐阅读顺序

1. 先看 [README.md](./README.md) 了解英文版整体定位和快速开始。
2. 再看 [offline-rl/README.md](./offline-rl/README.md) 了解离线采集、算法和数据格式。
3. 然后看 [openclaw-offline/README.md](./openclaw-offline/README.md) 了解如何把离线数据接回 slime 训练。

## 快速开始

### 1. 采集离线轨迹

```bash
cd offline-rl

python scripts/collect_from_benchmark.py --env osworld --n 100 --output data/osworld_trajs.jsonl
python scripts/collect_from_benchmark.py --env androidworld --n 100 --output data/androidworld_trajs.jsonl
python scripts/collect_from_benchmark.py --env webarena --n 100 --output data/webarena_trajs.jsonl
python scripts/collect_from_benchmark.py --env alfworld --n 100 --output data/alfworld_trajs.jsonl
```

### 2. 训练轻量 offline baseline

```bash
python scripts/train_offline.py --algo iql --data data/osworld_trajs.jsonl --steps 500
python scripts/train_offline.py --algo cql --data data/webarena_trajs.jsonl --steps 500
python scripts/train_offline.py --algo awac --data data/alfworld_trajs.jsonl --steps 500
python scripts/train_offline.py --algo grpo --data data/osworld_trajs.jsonl --steps 200 --n-policy-updates 2
```

如果你的数据里保存了行为策略 log-prob，GRPO 会优先使用真实离线策略概率，而不是只拿 ref-policy 近似。支持的字段说明在 [offline-rl/README.md](./offline-rl/README.md) 里。

### 3. 可选地计算 advantage 权重

```bash
cd ../openclaw-offline

python compute_weights.py \
  --data ../offline-rl/data/osworld_trajs.jsonl \
  --output ../offline-rl/data/osworld_iql_weights.json \
  --algo iql \
  --train-steps 500 \
  --beta 3.0
```

### 4. 启动基于 slime 的离线微调

```bash
bash run_qwen35_4b_osworld_offline_rl.sh
bash run_qwen35_4b_androidworld_offline_rl.sh
bash run_qwen35_4b_webarena_offline_rl.sh
bash run_qwen35_4b_alfworld_offline_rl.sh
```

## 致谢

OpenClaw-RL-Offline 构建在 Gen-Verse 的 OpenClaw-RL 之上。这个分支重点强化的是 offline 数据、回放训练与可发布性，而不是重做 upstream 的方法组织。