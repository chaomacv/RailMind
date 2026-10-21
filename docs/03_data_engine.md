# 大模型教师标注

教师模型将已有铁路图像转换为场景、类别、证据和风险描述监督，不替代检测器需要的人工框标注。

## 配置兼容的教师服务

```bash
cp .env.example .env
# 在本地填写 .env 后加载环境变量
source .env
python -m railmind annotate --manifest data/train/manifest.jsonl \
  --image-root data/train/images --output outputs/teacher.jsonl
```

以上默认只检查并展示请求。设置 `RAILMIND_TEACHER_BASE_URL`、`RAILMIND_TEACHER_API_KEY`、`RAILMIND_TEACHER_MODEL`，加 `--execute` 才实际调用。基础 URL 以 `/v1` 结尾，适配器追加 `/chat/completions`。Gemini 通过 **OpenAI 兼容网关**访问，不使用 Google 原生 API 协议。调用前确认提供方模型名称及多模态支持。

请求只包含图像字节与提示词，不发送真值类别、文件名、生成目标或场景元数据。`teacher.txt` 提供类别定义与部件功能背景，要求谨慎描述可见证据。模型生成坐标不能直接作为真值。

回答必须包含具有标准 `anomaly_classes` 的 JSON 对象。无效回答会报错，不会静默变为负样本。每成功处理一张图像就保存进度；续接时检查图像、模型、端点和提示词哈希。更换实验设置时使用新的输出文件。HTTP 429 仅有限重试，生成结果不确定时不盲目重复提交。

## 训练前审核

使用 manifest 中的 SHA256 填写审核文件，以下为格式示例：

```json
[{"id":"example-01", "sha256":"EXACT_IMAGE_SHA256", "approved":true,
  "verified_labels":["noclip"], "reviewer":"reviewer-name"}]
```

```bash
python -m railmind review --records outputs/teacher.jsonl \
  --decisions data/reviews.json --image-root data/train/images \
  --output outputs/reviewed_teacher.jsonl
python -m railmind export-sft --records outputs/reviewed_teacher.jsonl \
  --image-root data/train/images --output data/sft/train.json
```

只导出审核通过的记录。修改类别时，需添加 `corrected_teacher` 对象，包含修正后的最终类别及一致证据；只改标签会留下误导性的教师描述。审核者还需检查幻觉、风险表述、类别覆盖和负样本。

历史数据中 `fullAnalysis.raw_output` 的 JSON 不会自动视为已审核，需要转换格式并审核。私有图像路径、原始回答和训练监督文件不随源码发布。
