# 后训练论文路线、曲线解读与调参

本文帮助读者从“能运行 SFT/RL”进阶到理解曲线、选择超参数，并了解后训练方法的发展。它整理已有论文，不声称穷尽当前最新成果。文末保留原始论文标题和链接，方便检索。

对 RailMind 这类领域 VLM 项目，更值得检查的是：模型是否学习了铁路视觉证据，还是只模仿回答风格？SFT 是否改善实际任务，还是只降低了偏置数据上的损失？RL 是否改进类别决策，还是形成过度自信的捷径？学习率、KL、奖励和输出长度是否支持选择当前 checkpoint？

## 后训练的三个层次

1. **SFT 学习回答分布。**模仿领域回答风格、格式、证据表达及常见任务分解。
2. **偏好优化学习相对偏好。**DPO/IPO/KTO/ORPO/SimPO 等方法比较较好与较差回答，通常无需在线环境交互。
3. **基于可验证或学习奖励的 RL 优化得分。**PPO/GRPO/DAPO/GSPO 等方法采样回答、评分并更新策略，同时控制偏移。

目标行为缺失或不稳定时先做 SFT；能收集优选/拒绝回答对时考虑偏好优化；奖励可靠、成本可控且单靠 SFT 难以学会时再考虑在线 RL。

## 发展脉络

| 时期 | 代表论文 | 核心思路 | 意义 |
|---|---|---|---|
| 2017–2019 | PPO；Fine-Tuning Language Models from Human Preferences | 约束策略更新，依据人类判断学习奖励 | RLHF 优化基础 |
| 2021–2022 | FLAN、LoRA、InstructGPT、Self-Instruct、Constitutional AI | 指令微调、低秩适配、RLHF、合成指令、AI 反馈 | 从基座模型到指令遵循 |
| 2023 | LLaVA、InstructBLIP、QLoRA、LIMA、DPO、RRHF | 视觉指令、量化适配、数据质量优先、直接偏好损失 | 降低训练与复现成本 |
| 2024 | KTO、ORPO、SimPO、DeepSeekMath/GRPO | 简化或无参考偏好优化，组相对 RL | 降低工程开销、推进推理任务 |
| 2025–2026 | Qwen2.5-VL/Qwen3-VL 报告、DeepSeek-R1、DAPO、GSPO、GRPO 理论研究 | 大规模 VLM 训练、可验证奖励、序列级/组策略优化 | 从对话对齐走向推理及多模态领域任务 |

## 重点论文怎么读

### SFT 与指令微调

1. **FLAN**：理解多任务指令混合如何改善零样本表现。RailMind 的教师生成领域任务可从这个角度理解。
2. **Self-Instruct**：理解合成指令；铁路场景需要更强教师和更严格审核，关注标签与视觉定位的实际后果。
3. **LIMA**：关注高质量精选数据的价值，不盲目追求生成数量。
4. **LoRA、QLoRA**：分别理解可训练参数范围与量化后的显存成本。二者都不能修复错误数据。
5. **LLaVA、InstructBLIP**：理解视觉指令微调为何需要精心设计图文任务，而非只做文本 SFT。
6. **Qwen2.5-VL、Qwen3-VL 技术报告**：了解基座系列的图像分辨率、视觉 token、多语言/多模态能力及后训练假设。

### RLHF、偏好优化与可验证奖励 RL

1. **PPO**：理解裁剪、策略比率与稳定更新，不必重新实现，但应知道更新为何可能崩溃。
2. **Fine-Tuning Language Models from Human Preferences**：理解先学习奖励、再优化语言模型策略的流程。
3. **InstructGPT**：学习 SFT、偏好数据、奖励模型、PPO、KL 控制与评测组成的 RLHF 流程。
4. **Constitutional AI**：理解由规则指导的 AI 反馈。它能减少人审成本，但不取消验证。
5. **DPO**：理解如何从偏好对直接训练策略，避免显式奖励模型和在线 PPO 循环。
6. **RRHF、IPO、KTO、ORPO、SimPO**：了解简化偏好优化的分支；PPO 成本或稳定性不合适时可研究。
7. **DeepSeekMath/GRPO**：理解同提示多回答的组内比较，以及如何减少对独立价值模型的依赖。
8. **DeepSeek-R1**：关注可验证奖励和分阶段推理训练；奖励设计与任务选择决定是否存在可靠学习信号。
9. **DAPO、GSPO**：理解动态采样、解耦裁剪、序列级策略优化等大规模训练工程。
10. **GRPO 理论研究**：有实现经验后再看，重点理解估计量、偏差、噪声与长度敏感性。

## 如何看 SFT 曲线

训练损失下降只说明模型更善于预测监督 token，不能证明推理、定位或分类更好。

| 信号 | 较健康的表现 | 警示表现 | 常见排查方向 |
|---|---|---|---|
| 训练损失 | 平滑下降后趋稳 | 快速下降但验证不改善 | 降学习率、减 epoch、增加多样性 |
| 验证损失 | 下降或接近训练损失 | 训练早期后开始上升 | 早停、减 epoch、检查划分 |
| 格式正确率 | 较快提高 | 波动或退化 | 补格式样例、降学习率 |
| 类别 macro-F1 | 验证集上改善 | 只有高频类提高 | 数据平衡、按类采样 |
| 输出长度 | 接近目标且稳定 | 变长但指标不变 | 检查长度约束、缩短监督、审计奖励 |
| 幻觉类别 | 减少 | 预测生成目标而非可见类别 | 严格审核、移除未验证合成数据 |

RailMind 选择 checkpoint 时综合验证 macro-F1、格式正确率、各类精确率/召回率，以及人工检查的失败案例，不能只看 SFT loss。

## 如何控制 SFT 超参数

领域 VLM 通常先受数据质量限制，再受优化器限制。以下仅是探索起点，需按模型、训练方式与验证结果调整。

| 参数 | 保守探索起点 | 过高的风险 | 过低的风险 |
|---|---|---|---|
| 学习率 | LoRA 可探索 `5e-6`–`1e-5`，全量微调常需更低 | 格式漂移、遗忘、损失不稳 | 学得慢、领域术语欠拟合 |
| epoch | 高质量重复数据可先试 1–3 | 记忆化、验证退化、表达重复 | 格式学习不足 |
| LoRA rank | 可先试 8–32 | 显存增加、小数据过拟合 | 领域适配容量不足 |
| warmup 比例 | 可先试 0.03–0.10 | 过长浪费更新步数 | 过短易有初期尖峰 |
| 权重衰减 | 小值或框架默认 | 过强可能妨碍适配器学习 | 正则化不足 |
| 最大长度/截断长度 | 容纳图像提示与最终格式 | 成本增加、学到冗长回答 | 证据或最终 JSON 被截断 |
| 图像像素预算 | 先匹配基座配置 | 显存激增 | 小缺陷丢失 |

一次只改一个主要变量：先用小实验找稳定学习率，固定学习率比较数据方案，再比较 rank/epoch，最后扩展 GPU 与批量。第一个实验先确认设置是否合理。

## 如何看 RL 曲线

奖励改善时真实质量也可能变差，因此联合观察：

| 信号 | 较健康的表现 | 警示表现 | 可能原因 |
|---|---|---|---|
| 平均奖励 | 上升后趋稳 | 一开始就跳到高值 | 奖励太简单或发生泄漏 |
| 完全匹配率 | 上升 | 奖励上涨但匹配不动 | 超集/格式漏洞 |
| 超集比例 | 受控 | 持续上涨 | 多报类别 |
| 相对参考模型的 KL | 非零且有界 | 近零或爆炸 | 可能未学习或策略漂移 |
| 熵 | 缓慢下降 | 早期塌缩 | 模式坍塌 |
| 输出长度 | 稳定且符合任务 | 为得分持续拉长 | 奖励未约束冗长 |
| 无效 JSON 比例 | 下降 | 持续偏高 | SFT 尚未准备好 |
| 各类召回 | 改善且精确率不崩溃 | 稀有类消失 | 奖励/数据不平衡 |

类别 RL 要在相同验证集画出完全匹配、超集、无效输出和 macro-F1。总多预测缺陷的模型可能拿到部分奖励，却更不适合巡检。上述信号都是排查线索，不能凭单条曲线判断原因。

## 如何控制 RL 超参数

RL 应谨慎更新。如果单次运行使格式或行为剧烈变化，需检查更新强度。

| 参数 | 含义 | 过高 | 过低 |
|---|---|---|---|
| RL 学习率 | 策略更新强度 | KL 尖峰、格式漂移、奖励投机 | 改善不明显 |
| KL 系数 | 靠近参考模型的约束 | 阻碍学习 | 漂移或塌缩 |
| 裁剪范围 | 限制策略比率更新 | 更新不稳定 | 学习缓慢 |
| 组大小/rollout 数 | 组内奖励比较质量 | 成本高，可能偏向偶然成功的长样本 | 优势估计噪声大 |
| 最大回答长度 | 搜索预算 | 冗长和成本增加 | 答案截断 |
| 温度 | 探索程度 | 低质量噪声样本增多 | 探索不足 |
| 奖励权重 | 优化行为的分配 | 利用容易得分的项 | 信号太弱 |

建议流程：先确认 SFT 格式正确率；跑极小 RL 实验并看原始生成；每隔少量步数画奖励、KL、长度和无效率；若靠变长或多报提高奖励则尽早停止；所有 checkpoint 都用实际任务指标验证。

## RailMind 的实验梯度

- 不理解铁路类别：先改善 SFT 数据。
- 理解类别但多报/少报：再探索类别 RL。
- 定位差：检查 GroundingDINO 和框标注，类别 RL 不会直接修复坐标。
- 验证数字改善但人工观感退化：审计指标、奖励和标签。

建议逐级实验：

1. `SFT-small`：200–500 条审核样例，单卡短程运行，检查格式与可见证据。
2. `SFT-domain`：完整审核描述数据，保守学习率，1–3 个 epoch。
3. `SFT-ablation`：比较仅真实数据与真实加审核合成数据。
4. `RL-smoke`：100–300 条审核合成提示，低学习率、短程检查。
5. `RL-full`：固定 SFT checkpoint，使用完整审核 RL 数据，画完全匹配/超集/KL/长度。
6. `Joint-eval`：相同私有 820 基准，比较 SFT、GRPO、GSPO 与检测器变体。

保存精确数据哈希、提示词哈希、配置、种子、框架版本、日志和生成抽样表，才能让曲线对应可追溯的实验。所有样本与数据清单保留在私有环境。

## 论文与原始链接

基础论文：

- PPO: Proximal Policy Optimization Algorithms - https://arxiv.org/abs/1707.06347
- Fine-Tuning Language Models from Human Preferences - https://arxiv.org/abs/1909.08593
- FLAN: Finetuned Language Models Are Zero-Shot Learners - https://arxiv.org/abs/2109.01652
- LoRA: Low-Rank Adaptation of Large Language Models - https://arxiv.org/abs/2106.09685

指令微调与 VLM SFT：

- InstructGPT: Training Language Models to Follow Instructions with Human Feedback - https://arxiv.org/abs/2203.02155
- Self-Instruct: Aligning Language Models with Self-Generated Instructions - https://arxiv.org/abs/2212.10560
- LLaVA: Visual Instruction Tuning - https://arxiv.org/abs/2304.08485
- InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning - https://arxiv.org/abs/2305.06500
- QLoRA: Efficient Finetuning of Quantized LLMs - https://arxiv.org/abs/2305.14314
- LIMA: Less Is More for Alignment - https://arxiv.org/abs/2305.11206
- Qwen2.5-VL Technical Report - https://arxiv.org/abs/2502.13923
- Qwen3-VL Technical Report - https://arxiv.org/abs/2511.21631

偏好优化与强化学习：

- Constitutional AI: Harmlessness from AI Feedback - https://arxiv.org/abs/2212.08073
- RRHF: Rank Responses to Align Language Models with Human Feedback without tears - https://arxiv.org/abs/2304.05302
- DPO: Direct Preference Optimization - https://arxiv.org/abs/2305.18290
- IPO / general preference learning theory - https://arxiv.org/abs/2310.12036
- KTO: Model Alignment as Prospect Theoretic Optimization - https://arxiv.org/abs/2402.01306
- ORPO: Monolithic Preference Optimization without Reference Model - https://arxiv.org/abs/2403.07691
- SimPO: Simple Preference Optimization with a Reference-Free Reward - https://arxiv.org/abs/2405.14734
- DeepSeekMath and GRPO - https://arxiv.org/abs/2402.03300
- DeepSeek-R1 - https://arxiv.org/abs/2501.12948
- DAPO: An Open-Source LLM Reinforcement Learning System at Scale - https://arxiv.org/abs/2503.14476
- GSPO: Group Sequence Policy Optimization - https://arxiv.org/abs/2507.18071
- Demystifying GRPO: Its Policy Gradient is a U-Statistic - https://arxiv.org/abs/2603.01162


## 一周阅读顺序

1. FLAN、LoRA、QLoRA
2. InstructGPT
3. LLaVA 或 InstructBLIP
4. DPO
5. DeepSeekMath/GRPO
6. DeepSeek-R1
7. DAPO 或 GSPO

每篇读完写一句：“这篇论文会让我在 RailMind 中调整____，原因是____。”把论文阅读落实为工程判断，而不只是记住算法名称。
