# SFT 与 GroundingDINO 训练

Qwen、RL 和旧版检测器分别使用独立的 Python/CUDA 环境。CPU 工具包无需 PyTorch。离线测试不能证明 GPU 依赖兼容，不建议将所有训练框架装入同一个环境。

## Qwen SFT

先用 `export-sft` 导出审核后的数据，生成 `train.json` 及配套的 `dataset_info.json`。在兼容的 GPU 环境中安装对应框架：

```bash
python -m pip install -e .
python -m pip install -e third_party/llamafactory
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
python -m railmind train sft --recipe configs/sft/custom_lora.json \
  --model models/Qwen3-VL-8B-Instruct --dataset data/sft/train.json \
  --output outputs/sft_custom
```

默认仅打印解析后的命令。模型、数据和 GPU 环境准备完成后加 `--execute`。启动器设置 `FORCE_TORCHRUN=1`，生成绝对路径配置，并拒绝非空输出目录。生成的 `.yaml` 使用 JSON 语法；JSON 是有效 YAML，框架的 YAML 解析器可以读取。

配置选择：

- `custom_lora.json`：自己的审核数据，不声称复现论文结果。
- `paper_20931.json`：论文数据量，3 个 epoch，名义有效批量 32，学习率 5e-6。精确历史配置不完整；全参数微调是明确记录的假设，不是原始方法的恢复证据。
- `server_lora_24265.json`：离线包中的后续配置，rank 8，3 个 epoch，学习率 1e-5，有效批量 32，截断长度 8192。包含额外的直接框输出样本，与论文描述监督配置分开报告。

批量检查为 `GPU 数 × 单卡批量 × 梯度累积步数`。改变 GPU 数时显式复制并调整配置。框架模板必须支持 `qwen3_vl`；服务器旧顶层 LLaMA-Factory 的模板集合不同。基座模型从官方分发渠道获取，保留其许可证。

保存解析后的配置、数据/提示词哈希、随机种子、实际框架版本、基座模型、硬件、日志和 checkpoint。若 RL 使用完整 SFT 模型，先按所选框架合并 LoRA；适配器形式的 RL 需单独记录训练方案。

## 人工框标注

合成图像需重新检查并标框。以下为审核格式示例：

```json
[{"id":"sample-id", "sha256":"EXACT_IMAGE_SHA256", "approved":true, "reviewer":"reviewer",
  "verified_labels":["noclip"], "instances":[{"label":"noclip","bbox":[100,200,160,240]}]}]
```

`bbox` 使用 **xyxy 像素坐标**。`review` 检查边界和已确认类别的一致性。已有人工 COCO 标注可先导入清单再审核，不需要先生成教师文本。

```bash
python -m railmind export-dino --records data/train/reviewed_boxes.jsonl \
  --validation data/val/manifest.jsonl --train-root data/train/images \
  --val-root data/val/images --output-dir data/dino
```

输出 ODVG 训练 JSONL、COCO 验证 JSON、类别映射及含本地绝对路径的 `datasets.json`，并检查训练/验证哈希和增强来源关系。迁移机器后重新生成配置。这些数据产物不上传仓库。

## 与源码匹配的检测器

仓库内保存的是 **Open-GroundingDino**，采用 `models/GroundingDINO` 结构，与官方仅推理包的目录结构不同。在兼容环境中安装依赖、编译算子：

```bash
python -m pip install -r third_party/groundingdino/requirements.txt
cd third_party/groundingdino/models/GroundingDINO/ops
python setup.py build install
python test.py
cd ../../../../..
```

编译前先安装匹配 CUDA 的 PyTorch/torchvision，参考上游快照 README 中的历史环境基线。本发布及私有复现包都不是完整的已验证容器或历史环境锁。不要默认服务器 Python 3.14 可直接用于 GPU 训练。

保留的 `cfg_railway.py` 包含 Swin-T、BERT 冻结、30 个 epoch 及实际优化器/调度配置。预训练 Swin-T 和 BERT 权重另行获取。

```bash
python -m railmind train dino --recipe configs/dino/recipe.json \
  --framework-root third_party/groundingdino \
  --checkpoint weights/groundingdino_swint_ogc.pth \
  --text-encoder weights/bert-base-uncased \
  --dataset-config data/dino/datasets.json --output outputs/dino_train2
```

加 `--execute` 才训练。启动器使用当前解释器，不依赖旧 `zcy` Conda 路径。论文“batch size 4”与源码“每进程 batch 4”不能直接当作相同全局批量，需记录实际进程数。
