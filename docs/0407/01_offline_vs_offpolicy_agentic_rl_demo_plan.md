# Offline优先还是Off-Policy优先：Agentic RL第一性原理分析与Demo计划

## 1. 先给结论

当前仓库这轮优先集成offline相关算法，是合理的工程顺序；并且从算法形态上看，仓库其实已经同时覆盖了offline与off-policy，只是这两者在agentic场景下边界没有教科书里那么泾渭分明。

更准确地说：

- `offline RL` 关注“训练时不再与环境新增交互，只从固定日志数据学习”。
- `off-policy RL` 关注“学习目标可以使用不是当前策略产生的数据”，它既可以发生在固定数据集，也可以发生在在线训练的replay buffer里。
- 在agentic RL里，很多方法从工程上会落到“固定日志 + replay / preference / reward modeling”的混合形态，因此它们既带offline色彩，也带off-policy色彩。

所以真正的问题不是“到底选offline还是off-policy”，而是“在当前环境成本、数据条件、验证成本下，先把哪一层闭环打通最划算”。这一点上，offline-first是更短路径。

## 2. 第一性原理：为什么先做offline-first

### 2.1 agentic环境的核心成本不是loss，而是交互基础设施

在浏览器、桌面、代码仓库、工具调用这些agent场景里，训练成本主要来自：

- 环境启动慢且脆弱，例如浏览器、容器、桌面虚拟机、远程exec server。
- 行为具有副作用，失败动作会污染环境状态。
- 回放与复现困难，同一个任务会受页面、网络、文件系统、依赖版本影响。
- 评测通常不是单步reward，而是长程结果、patch是否可测、UI任务是否完成。

这意味着在工程上，最先需要验证的是：

- 轨迹schema是否够表达任务、历史、工具调用、结果。
- 奖励与偏好信号能否从日志中稳定提取。
- 算法是否能在固定数据上稳定训练，而不是一启动环境就把问题放大成“训练 + infra”双重不确定性。

### 2.2 offline-first最适合当前仓库与开发机条件

对当前8G CPU开发机来说，先做offline-first的优势非常直接：

- 可以先用JSONL/轨迹数据做smoke test，不必先拉起完整在线环境。
- 更容易控制变量，先验证数据接口、loss、batch构造、奖励聚合、pair mining是否正确。
- 更适合把32个算法统一到同一套dataset adapter与evaluation接口上。
- 更容易在本地先做多组数据模拟与单元测试，再决定哪些benchmark值得上重型环境。

### 2.3 off-policy在线迭代仍然重要，但它依赖稳定的live loop

在线off-policy的真正价值，在于：

- 新策略不断采样，replay buffer持续刷新。
- 可以把失败样本、hard negative、success buffer动态再利用。
- 能做真正的policy improvement，而不只是对历史行为做拟合或重排序。

但这要求：

- 已有稳定的环境池。
- 已有稳定的结果评判接口。
- 已有足够便宜的交互回路。

在这些条件没先打牢前，直接优先做在线off-policy，往往会把主要时间花在环境管理而不是算法验证上。

## 3. 当前仓库是否已经同时覆盖offline与off-policy

答案是“是，但要按训练信号去理解，而不是按教科书标签理解”。

根据当前 `offline-rl/offline_rl/algorithms/__init__.py` 的真实导出列表，32个算法大致可分为三层：

### 3.1 固定数据集offline值学习 / actor-critic

- IQL
- CQL
- AWAC
- TD3BC
- EDAC
- CRR
- BCQ
- DecisionTransformer

这一层最接近经典offline RL，核心目标是在固定日志数据上学习更稳健的Q、V、policy或序列模型。

### 3.2 带replay / success buffer / value modeling特征的agentic变体

- OffPolicyGRPO
- SORLOffPolicyGRPO
- ARPO
- OREO
- Retrospex
- WebRL
- GLIDER
- ArCHer
- DigiRL
- DigiQ
- AgentQ
- ILQL
- VEM
- REBEL

这一层是当前仓库最接近“agentic off-policy”的部分。它们很多并不是传统DQN式定义，但从工程视角看，都是在利用非当前策略实时生成的轨迹、成功缓冲区、回放样本、价值估计或结果模型来提升策略。

### 3.3 基于已记录轨迹做偏好或结果优化

- RWFineTuning
- KTO
- DPO
- DMPO
- IPO
- CPO
- SimPO
- ETO
- ORPO
- RRHF

这一层通常不直接写成“off-policy RL”，但在agentic场景中经常更实用。原因是很多真实数据天然是“日志轨迹 + 成败标签 + pair preference + outcome score”，而不是干净的在线reward MDP。

## 4. 这轮为什么说“优先集成offline算法”仍然是对的

如果把目标改写成一句更精确的话，那就是：

> 这轮优先实现面向日志轨迹的学习能力，而不是优先投入在线环境交互能力。

这样就没有概念混乱了。

因为从能力视角看，这轮优先补的是：

- 固定日志上的Q/V/policy学习。
- 结果分数、偏好对、成败标签的利用。
- success buffer / replay样本的再利用。
- 对浏览器、GUI、代码任务日志的统一适配。

这些能力恰好是将来做更强在线off-policy的前置条件。

## 5. 已经存在、且对agentic RL真正有价值的offline / replay论文方向

调研结果显示，agentic RL里“有效”的离线方法，常常不是纯粹沿用MuJoCo那套offline RL，而是以下三类：

### 5.1 价值学习 / Q学习驱动的agent方法

- DigiQ / DigiRL：面向数字或设备控制任务，把长程任务拆成更适合Q学习与价值建模的形式。
- Agent Q：把agent轨迹上的选择问题转成价值学习，适合工具调用或分支动作场景。
- ILQL：对语言模型和轨迹数据都相对自然，适合日志学习与离线价值修正。

### 5.2 回放 / 结果建模 / 轨迹改写方法

- Retrospex：强调从已有轨迹中做后验整理与反思式优化。
- WebRL：明确面向网页代理任务，强调任务结果与步骤价值的结合。
- GLIDER / OREO / VEM：更强调结果建模、环境价值建模、步骤层级归因。

### 5.3 偏好与结果优化方法

- DMPO、DPO、SimPO、ORPO、RRHF、KTO：更适合“日志里已有好坏示例、pair、结果分数”的现实数据形态。
- 在agentic RL中，这一类往往比硬套传统连续控制offline RL更容易先落地。

结论很明确：agentic RL领域里确实已经有比较有效的offline / replay实现方向，只是它们往往以“trajectory preference optimization”“outcome optimization”“replay-based improvement”这些名字出现，而不是都挂着传统offline RL标签。

## 6. 已确认可用于demo的数据集与测试任务

当前可以分成两类看。

### 6.1 已有或接近现成的离线日志数据

| 数据源 | 适用场景 | 当前状态 | 备注 |
| --- | --- | --- | --- |
| `offline-rl/data/wa_trajs.jsonl` | WebArena风格网页任务 | 本地已存在 | 最适合先做smoke test |
| WebLINX / weblinx-browsergym | 网页代理日志 | 公开可获取 | 比较适合先做公开demo |
| WebArena human / execution trajectories | 网页任务日志 | 官方资源存在 | 适合补充本地种子数据 |
| SWE-Gym-Subset | 代码代理 / SWE | `swe-rl` 已给出预处理思路 | 更适合第二阶段 |
| SWE-bench Verified | 代码代理 / SWE | `swe-rl` 已给出预处理思路 | 适合小规模验证子集 |

### 6.2 更像benchmark任务，而不是现成离线训练集

| 任务源 | 适用场景 | 当前状态 | 备注 |
| --- | --- | --- | --- |
| OSWorld / OSWorld-Verified | 桌面与操作系统代理 | 评测任务公开，完整离线日志不如Web/SWE充足 | 适合后续live或半离线方案 |
| AndroidWorld / AITW系任务 | 移动端代理 | 任务和交互研究很多，但统一离线日志适配还需额外整理 | 不建议作为第一批demo |

## 7. 推荐的demo顺序

### 阶段A：本地最小离线闭环demo

目标：先验证仓库里的offline训练栈、数据adapter、算法接口没有结构性问题。

建议数据：

- 直接使用本地 `wa_trajs.jsonl`

建议算法：

- IQL：做最基础的值学习基线
- AgentQ：做agentic Q-learning基线
- DMPO 或 KTO：做偏好/结果优化基线

成功标准：

- 数据可被统一读入
- 三类算法都能完成最小训练
- holdout日志上的reward / ranking指标有可解释输出
- 不要求先追求SOTA

### 阶段B：公开Web离线demo

目标：把本地种子数据升级成可复用、可展示的公开任务demo。

建议数据：

- WebLINX为主
- WebArena人类/执行轨迹为辅

建议算法：

- WebRL 或 Retrospex
- AgentQ 或 ILQL
- DMPO / SimPO / ORPO 三选一

评测建议：

- 先做离线reward / pair ranking / success label预测
- 再做10到20个小规模live网页任务抽样验证

### 阶段C：SWE离线demo

目标：把网页场景验证过的数据链路复用到代码代理场景。

建议数据：

- SWE-Gym-Subset 训练
- SWE-bench Verified 小验证集评测

建议算法：

- DMPO / ORPO / KTO 中选1到2个
- ILQL / AgentQ 中选1个

说明：

- 这一步已经比Web demo重很多，具体计划详见 `02_swe_bench_a3s_openclaw_demo_plan.md`。

### 阶段D：OSWorld / AndroidWorld扩展

目标：进入真正更重的GUI与设备代理场景。

建议：

- 不要作为第一批demo
- 先等Web和SWE链路跑顺，再复用数据schema与评测框架

## 8. 不建议的做法

当前阶段不建议：

- 一上来同时给32个算法都配完整公开benchmark。
- 一上来先做完整在线off-policy环境池。
- 一上来把OSWorld、AndroidWorld、SWE-Bench、WebArena全部并行接入。

这些做法都会把主要风险从“算法是否有效”变成“工程是否可控”。

## 9. 最终建议

如果目标是尽快做出一个可信、可演示、可继续扩展的agentic offline RL demo，那么最短路径是：

1. 先用本地 `wa_trajs.jsonl` 做最小闭环。
2. 再上WebLINX / WebArena公开日志做正式Web demo。
3. 再复用同一套adapter与评测思想，做SWE-Gym / SWE-Bench Verified离线demo。
4. 最后再考虑接入完整在线off-policy环境池与更重GUI任务。

这条路径既符合第一性原理，也符合当前仓库已有32算法的真实能力边界。
