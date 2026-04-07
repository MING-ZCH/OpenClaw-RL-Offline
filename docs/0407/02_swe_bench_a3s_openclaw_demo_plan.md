# OpenClaw-RL、PR #1、SWE-RL与SWE-Bench：接入方向分析与Demo计划

## 1. 先给结论

`puyuan1996/OpenClaw-RL` 的 PR #1 不是“小修小补”，而是把一整套面向代码代理的在线样本生成与RL训练代理层接进了OpenClaw-RL：它包含OpenAI兼容的RL proxy、a3s-code traffic driver、seed task与template workspace、异步rollout worker、运行脚本、以及一批runtime guard与测试。

但这条链路的本质不是“离线数据适配”，而是“让真实coding agent多轮会话持续产出训练样本”。因此它更接近在线或半在线agentic RL系统，而不是本仓库当前优先验证的offline demo。

所以对当前仓库最合理的策略不是直接复刻整套 `a3s-code-adapter + swe-rl live infra`，而是：

1. 先借鉴 `swe-rl` 的数据预处理与评测接口，做一个离线SWE demo。
2. 把 `a3s-code-adapter` 视为下一阶段的在线样本生产器，而不是当前第一阶段demo的前提。

## 2. PR #1 实际做了什么

从PR补丁内容看，PR #1 主要由5个补丁组成：

### PATCH 1

- 新增两个为4卡环境定制的 `toolcall-rl` 脚本。

### PATCH 2

- 正式引入 `a3s-code-adapter`。
- 新增16个文件，约4345行代码。
- 核心包括：
  - `code_rl_api_server.py`
  - `code_rl_rollout.py`
  - `a3s_code_agent_traffic_driver.py`
  - `run_a3s_code_rl_4gpu.sh`
  - `check_simulated_user_backends.py`
  - `seed_data/code_task_seeds.json`
  - `task_templates/mini_taskboard/...`

### PATCH 3 / PATCH 4

- 修复并增强 `a3s-code-adapter`：
  - 训练日志结构化
  - token统计与执行反馈记录
  - runtime guard
  - task template从单文件脚本升级为更完整的小package
  - 新增测试
  - 修复分布式与device相关bug

### PATCH 5

- 删除旧的 `task_templates_old`，收敛目录结构。

## 3. PR #1 最重要的工程含义

### 3.1 `a3s-code-adapter` 是“训练样本生产系统”，不是普通adapter

它的核心逻辑是：

1. 用 `seed_data` 与 `task_templates` 生成真实代码任务。
2. 用 `a3s_code_agent_traffic_driver.py` 驱动多轮agent会话。
3. 由 `code_rl_api_server.py` 作为OpenAI兼容代理，拦截模型请求、记录prompt/response/logprobs、组织session。
4. 把会话结果交给 `code_rl_rollout.py` 输送给 `slime` 异步训练器。

也就是说，这套系统不是简单“把SWE数据读成JSONL”，而是“让agent自己持续产出训练轨迹”。

### 3.2 它的依赖比当前offline demo重很多

它依赖至少以下几类额外基础设施：

- `a3s_code` SDK 可用。
- OpenAI兼容推理入口可用。
- PRM或外部judge接口可用。
- workspace template生成、隔离、清理逻辑可用。
- 多轮会话与session级状态跟踪可用。

这对当前offline优先的仓库来说，不适合直接作为第一步demo。

## 4. `swe-rl` 路径的训练与测试逻辑

从 `swe-rl/README.md` 与其远程训练脚本可以提炼出一条很清晰的SWE训练链路：

### 4.1 数据准备

`swe-rl` 支持把以下数据预处理成训练JSONL：

- `SumanthRH/SWE-Gym-Subset`
- `SumanthRH/SWE-bench_Verified`

这说明它不是从零定义SWE训练数据，而是复用现成benchmark并把其转成OpenClaw/Slime可消费的格式。

### 4.2 环境执行

`swe-rl` 使用远程Docker执行服务与环境池：

- exec server 通常暴露在 `:5000`
- env pool server 暴露在 `:18090`

这条链路让代码代理可以在隔离环境里做真实patch生成、测试执行与结果回收。

### 4.3 训练形态

其训练脚本本质上还是把“问题样本 -> 轨迹生成 -> 奖励计算 -> 异步训练”挂到 `slime` 上，通常使用GRPO风格训练，并支持：

- `custom_generate_function`
- `custom_reward_function`
- `dynamic_history`
- outcome reward
- 可选PRM step reward

### 4.4 评测产物

评测输出一般包含：

- `traj.json`
- `patch.diff`
- `meta.json`
- `summary.json`

这意味着它有比较清晰的“最终patch是否有效”的验证界面，而不是只看语言模型loss。

## 5. 对当前OpenClaw-RL-Offline仓库最有价值的借鉴点

真正值得借鉴的不是把 `swe-rl` 原样搬过来，而是以下四点。

### 5.1 借鉴它的数据预处理思路

`swe-rl` 已经证明，SWE任务可以通过预处理转成统一JSONL。对当前仓库来说，这非常关键，因为它说明：

- SWE-Bench并不一定要先通过完整在线环境才能接入。
- 可以先做“离线训练格式化 + 小规模评测”。

### 5.2 借鉴它的评测输出结构

当前离线demo里，最缺的往往不是训练，而是“如何判断demo成功”。

`traj.json / patch.diff / meta.json / summary.json` 这种输出分层很值得复用，因为它天然把：

- 轨迹
- patch
- 元信息
- 汇总结论

分开保存，后面无论做PRM、judge还是人工复审都方便。

### 5.3 把 `a3s-code-adapter` 理解成第二阶段能力

它最适合的角色是：

- 当已有基础离线SWE demo后，用它继续采更多真实coding-agent轨迹。
- 把静态benchmark扩展成持续数据生产。
- 让在线off-policy与离线数据回放真正接起来。

### 5.4 不要在第一阶段就复刻整套live infra

对当前仓库与当前机器，直接复刻完整 `swe-rl + a3s-code` 路径的问题是：

- 工程依赖过多。
- 失败点太多，难以判断是算法问题还是infra问题。
- 与当前offline优先目标不一致。

## 6. 推荐的SWE demo版本计划

下面给出一个更适合当前仓库、也更容易逐步验证的四阶段计划。

## 6.1 阶段0：只做SWE离线数据适配

目标：不碰在线环境，先让SWE数据能进入当前offline训练栈。

建议数据：

- 训练：SWE-Gym-Subset
- 评测：SWE-bench Verified 小子集

最小需要实现的字段：

- 问题描述
- 仓库上下文摘要
- 历史轨迹或中间步骤
- 最终patch或patch摘要
- 测试结果 / outcome label
- 可选的偏好对或judge score

成功标准：

- 能稳定转成当前offline-rl可读的统一JSONL
- 能进行train/val split
- 能在不启动live exec server的前提下跑通训练

## 6.2 阶段1：先做离线SWE训练demo

目标：在代码代理场景上，验证当前仓库已有32算法里最适合SWE的最小集合。

建议不要一口气上很多算法，而是先选3类信号各1个代表：

- 偏好/结果优化：DMPO 或 ORPO
- 轻量偏好基线：KTO 或 RRHF
- 价值学习：ILQL 或 AgentQ

原因：

- SWE数据天然更容易带有patch outcome、pass/fail、judge score。
- 先做偏好与结果优化，往往比先做复杂在线探索更稳。
- ILQL / AgentQ 可以补上“动作选择质量”的价值建模能力。

成功标准：

- 至少1个偏好法和1个价值法能稳定训练
- 能输出可比较的offline metrics
- 能在一个小验证子集上做基本效果比较

## 6.3 阶段2：做小规模SWE-Bench验证

目标：不是追求完整benchmark跑分，而是先做可信演示。

建议设置：

- 只选10到20个SWE-bench Verified样本
- 优先选择依赖简单、执行时间短的子集
- 把结果整理成：问题、生成patch、测试结果、成功/失败原因

建议输出：

- `traj.json`
- `patch.diff`
- `meta.json`
- `summary.json`

这样后续无论继续做人工分析，还是接PRM / judge，都不需要重设计结果格式。

## 6.4 阶段3：再考虑接入 `a3s-code-adapter`

这一阶段才建议考虑在线数据生产。

原因很明确：

- 只有在离线SWE demo证明“数据schema、奖励信号、算法选择”是对的之后，在线采样才值得投入。
- `a3s-code-adapter` 的真正价值是持续生成真实coding-agent轨迹，而不是替代第一阶段的数据适配。

这一阶段的目标才应该是：

- 让OpenClaw-RL能持续产出新的coding-agent训练样本。
- 把SWE静态数据与在线生成数据合并成统一replay池。
- 真正进入“离线初始化 + 在线off-policy继续改进”的完整闭环。

## 7. 当前最推荐的实施顺序

如果只选一条最短路径，我建议按下面顺序推进：

1. 在当前仓库里新增SWE数据adapter，而不是先搬 `a3s-code-adapter`。
2. 用SWE-Gym-Subset做训练集，用SWE-bench Verified小子集做验证。
3. 先跑DMPO / ORPO + ILQL / AgentQ这类小而稳的算法组合。
4. 把输出整理成 `traj.json + patch.diff + meta.json + summary.json` 风格。
5. 只有在上面这条链路稳定后，再考虑引入 `a3s-code-adapter` 做在线流量采样。

## 8. 为什么这条路径比“直接复刻PR #1”更优

因为它更符合当前目标和约束：

- 更贴合当前offline优先的仓库方向。
- 更适合8G CPU开发机先做数据和接口验证。
- 更容易定位问题来源。
- 更容易产出一个可信的demo，而不是一个依赖很多外部系统的半成品。

## 9. 最终建议

这轮如果只确定一个SWE方向的demo版本计划，建议定为：

> 第一版只做“离线SWE demo”，不做完整在线 `a3s-code + swe-rl` 复刻。

具体落点是：

- 数据：SWE-Gym-Subset + SWE-bench Verified小子集
- 算法：DMPO / ORPO + ILQL / AgentQ
- 输出：patch与验证结果结构化保存
- 目标：证明当前OpenClaw-RL-Offline能在代码代理任务上完成可解释的离线训练与小规模验证

等这一版稳定，再考虑把PR #1里的 `a3s-code-adapter` 接成“第二阶段在线样本生产器”。这比一步到位复刻整套live infra更稳，也更符合第一性原理。
