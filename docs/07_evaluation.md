# 推理、评测与消融实验

## 语义预测

设置 `RAILMIND_BASE_URL`、`RAILMIND_API_KEY`、`RAILMIND_MODEL`，另行启动兼容模型服务。本仓库提供工作流工具，不提供推理服务端。

```bash
python -m railmind semantic --manifest data/test/manifest.jsonl \
  --image-root data/test/images --output outputs/semantic.jsonl
```

加 `--execute` 才调用。请求不发送基准标签、文件名或来源元数据。基础 URL 以 `/v1` 结尾，而非 `/chat/completions`。保持 HTTPS 校验，确认端点返回模型 JSON，而非存储网关响应。

## 缺陷定位

在匹配源码的检测器环境中先编译 CUDA 算子：

```bash
python -m railmind ground --manifest data/test/manifest.jsonl \
  --semantic outputs/semantic.jsonl --image-root data/test/images \
  --checkpoint weights/dino_finetuned.pth --text-encoder weights/bert-base-uncased \
  --output outputs/detections.jsonl
```

加 `--execute` 才推理。默认类别提示用句点分隔；图像短边缩放到 800，长边上限 1333；模型归一化框转换为原图 **xyxy 像素坐标**；框/文本阈值均为 0.35。阈值应在验证集选择并记录，不能使用测试集调参。

文本提示使用预测类别，不使用真值类别。预测类别为空时输出空检测。检测器解码短语必须唯一映射到一个请求的标准类别；歧义短语计入 `unmapped_phrases`，不随意分配。部分 token 回答可能因映射失败降低召回，做研究结论前需检查计数并验证映射。配置与 checkpoint 键不匹配会报错，不静默使用部分加载的模型。

适配器直接调用匹配的 `models/GroundingDINO` 实现，不使用引用另一套官方包结构的 `groundingdino/util/inference.py` 辅助代码。

## 指标

```bash
python -m railmind evaluate --manifest data/test/manifest.jsonl \
  --predictions outputs/detections.jsonl --iou 0.5 --output outputs/metrics.json
```

分类可只使用 `semantic.jsonl` 评测，将预测类别集合与图像级真值集合比较。检测按同类别、分数降序、指定 IoU 阈值进行一对一贪心匹配；重复预测计为假阳性。默认缺失预测会报错；`--allow-missing` 才显式按空预测处理。未知图像 ID 或类别会报错。

输出分别报告：

- 13 类分类宏平均 PR/RC/F1；macro-F1 是各类别 F1 的算术平均。
- 13 类检测宏平均 PR/RC/F1。
- 各场景检测 PR/RC/F1、已出现的场景平均；六个场景齐全时才计算六场景平均。
- 汇总实例计数后的检测微平均 PR/RC/F1。

数值范围为 `[0,1]`，无真值支持类别记为零。汇总精确率/召回率的调和平均**不能替代**类别 F1 的算术宏平均。固定 IoU 下的精确率**不是** COCO AP/mAP；需要 COCO AP 时，另行记录 pycocotools 协议及预测分数/真值转换规则。

发布匹配与汇总协议名为 `railmind-release-v1`。比较论文数字前需核对历史结果脚本。论文表 I 使用场景平均，其他表的类别宏平均与调和汇总表述存在不一致，不要将这些输出字段混在同一张表中。

## 建议报告的实验

1. 相同模型规模、提示词及图像级类别协议下，比较 Qwen 基座、描述 SFT、SFT+GRPO、SFT+GSPO。
2. 比较大模型直接输出框与理解引导定位。直接框输出需明确解析与坐标适配，不能直接当作语义预测。
3. 相同检测器与测试清单下，比较原始训练数据与同场景增强。
4. 比较预测类别提示与全类别提示。真值类别提示属于理想上界，不作为主结果。
5. 固定模型输出，扫描 IoU 0.1–0.5；置信度阈值仅在验证集选择。
6. 报告类别/场景及长尾结果、缺失输出率、无法映射短语、延迟、显存和小目标/遮挡失败案例。

820 版本中的接触网杆仅 11 张，需报告样本量与不确定性。稀疏标注可能将视觉上合理的额外检测计为假阳性，应审计案例，不要看完模型结果后修改测试标注。
