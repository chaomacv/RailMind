# RailMind：铁路图像分析、缺陷定位与训练复现

本仓库围绕论文 **RailMind: Multimodal Reasoning for Railway Infrastructure Security**，将服务器上的研究代码整理为一个独立项目。

核心分工：**Qwen3-VL 负责识别缺陷类别、解释可见证据；GroundingDINO 负责精确定位。** Gemini 等教师模型生成领域监督数据，NanoBanana 根据背景图和缺陷参考图增强长尾数据；SFT 后可选择 GRPO/GSPO 强化类别一致性。

## 从这里开始

第一次阅读本项目，建议先看 [辅助阅读指南：从目录到完整工作流](docs/00_reading_guide.md)，再进入各阶段教程。

在仓库根目录，用 Python 3.10 及以上版本：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m railmind demo --output outputs/demo
python -m unittest discover -s tests -v
python scripts/check_release.py
```

这个样例不需要 GPU、模型权重或 API Key。它验证图像清单、审核、SFT/RL 数据导出、增强计划和评测流程。图像是程序生成的示意样例，输出指标不是论文结果。

## 完整研究流程

| 阶段 | 输入 | 输出 | 文档 |
|---|---|---|---|
| 数据准备 | 图像、COCO 人工标注 | 带 split 和 SHA256 的统一 manifest | [数据规范](docs/02_data.md) |
| 教师数据引擎 | 训练图像、铁路知识提示词 | 场景、类别、可见证据、风险描述 | [大模型标注](docs/03_data_engine.md) |
| 数据增强 | 同场景背景与缺陷参考图 | 合成图像、生成记录、待审核标签 | [NanoBanana](docs/04_augmentation.md) |
| 审核与训练 | 通过审核的文本或框标注 | Qwen SFT 数据、DINO ODVG/COCO 数据 | [SFT 与检测器](docs/05_training.md) |
| 可选 RL | 合成图像的人工确认类别 | GRPO/GSPO 后训练模型 | [强化学习](docs/06_rl.md) |
| 联合推理 | 图像、Qwen 类别输出 | GroundingDINO 像素坐标检测框 | [推理与评测](docs/07_evaluation.md) |
| 实验比较 | 固定测试集与协议 | 分类/检测、消融和失败案例结果 | [复现边界](docs/08_reproducibility.md) |
| 知识板块 | SFT/RL 后训练、社区工具链、常见风险 | 面向复用者的学习路线与工程清单 | [SFT/RL 技能图谱](docs/09_sft_rl_skills_and_community.md) |
| 论文路线 | 经典与最新后训练论文、曲线解读、调参方法 | 从“会跑”到“会判断训练”的补课路径 | [后训练论文路线](docs/10_post_training_paper_roadmap.md) |

已有训练图像可以直接进入教师标注流程；已有人工框标注可以进入检测器训练。合成图像先审核，再进入 SFT/RL。生成时指定的目标类别不自动等于真实标签。

## 810 与 820 已核对

服务器的 `810_test_modified` 目录实际已有 **820 张图像、820 条 COCO 图像记录、2,083 个实例标注**，所有图像引用均存在。六个场景的数量与论文一致，最后增加的 10 张是接触网杆图像。

因此本仓库按实测的 **820 版本**记录当前基准，同时说明如何显式派生原 810 ID 子集。没有改动服务器原始数据，也不根据文件夹名称隐式删掉 10 张图像。证据在 [source_inventory.json](docs/source_inventory.json)。

## 两套 SFT 实验不要混用

- 论文：2,264 + 5,497 + 13,170 = **20,931** 条描述监督数据。
- 服务器后续混合训练：额外加入 **3,334** 条检测框输出样本，总计 **24,265**。
- GroundingDINO：以 3,334 张 train2 图像为主，保留实际源码和配置。
- RL：论文写明 3,134 个提示、16 个 rollout；当前没有恢复到原铁路专用启动脚本，发布的是明确标记的重建配置与类别奖励插件。

## 发布内容与使用条件

本仓库提供新的统一 CLI、教程、配置、提示词、必要训练框架源码快照及测试。原始图像、权重、加密文件、聊天记录、网站账号和 API Key 不进入源代码发布包。需要真实训练数据时，另行获取授权的数据或已有的私有复现包。

API 和训练命令默认只检查和展示参数，加 `--execute` 才实际调用。CPU 样例与回归测试已作为发布验证目标；GPU 训练、真实增强 API 和论文指标复现需在相应环境中执行，不能由离线样例代替。

旧 `network_app0.11` 网站不是论文训练主线，也没有搬入包含账号和密钥的旧目录；远端调用通过新的环境变量配置接入。`bank`、`huawei` 等独立项目不属于本仓库。

阅读顺序：[辅助阅读指南](docs/00_reading_guide.md) → [流程](docs/01_workflow.md) → [数据](docs/02_data.md) → [标注](docs/03_data_engine.md) → [增强](docs/04_augmentation.md) → [训练](docs/05_training.md) → [RL](docs/06_rl.md) → [评测](docs/07_evaluation.md) → [SFT/RL 技能图谱](docs/09_sft_rl_skills_and_community.md) → [后训练论文路线](docs/10_post_training_paper_roadmap.md)。

## 数据发布边界

训练集和评测集均不上传，包括图像、标注、教师监督文本、SFT/RL 样本、manifest 和划分/子集 ID 清单。本仓库仅发布方法代码、通用提示词、配置模板及文档；离线示例在运行时生成虚构样例。
