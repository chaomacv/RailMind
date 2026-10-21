# NanoBanana 同场景数据增强

服务器实现通过 GRSAI 风格的异步 NanoBanana 网关提交 `/v1/draw/nano-banana`，轮询 `/v1/draw/result`，再下载图像。这是**特定提供方的适配器**，不是通用 Google SDK。付费调用前需核对提供方的端点与模型可用性。

## 准备背景池和参考池

两个池都只使用训练集来源，先构建清单，再用 `review` 写入视觉审核结果。增强计划要求审核通过且图像哈希匹配；正常背景也需要显式填写空的已确认标签列表。背景应不含目标缺陷，参考图应清晰展示缺陷，并保持场景一致。例如，弹条断裂应合成在轨道弹条上，而不是声屏障面板上。多缺陷任务要求参考图的已确认类别覆盖目标组合。

运行前确定生成配额。可按真实训练集类别数量优先补充长尾类，例如用 `1 / max(class_count, 1)` 作权重，再分配并审核整数配额。计划器使用明确目标数量，不隐藏采样决策。

```bash
python -m railmind plan-augment --backgrounds data/backgrounds/reviewed.jsonl \
  --references data/references/reviewed.jsonl \
  --targets configs/augmentation_targets.example.json --seed 42 \
  --output outputs/augmentation_tasks.jsonl
python -m railmind augment --tasks outputs/augmentation_tasks.jsonl \
  --background-root data/backgrounds/images --reference-root data/references/images \
  --output-dir outputs/generated
```

计划器拒绝非训练集、场景不兼容、背景与参考相同、参考缺少目标类别的任务。随机种子使选图过程可重复，标注质量仍需数据整理者确认。

## 生成与来源追溯

设置 `NANO_BANANA_BASE_URL`、`NANO_BANANA_API_KEY`、`NANO_BANANA_MODEL`，用 `pip install -e '.[images]'` 安装图像支持。第二条命令加 `--execute` 才调用。

从来源代码整理出的 13 类提示词要求：缺陷变化符合物理常识且范围有限，保持透视、光照和部件几何，去除参考图上的标注框与文字。多类别任务组合相应约束；仍需检查冲突，不能假定结果遵守全部指令。

每个任务保存来源哈希、提示词、模型、提供方和已受理的任务 ID。中断后从保存的 ID 继续轮询，续接时检查已下载图像。如果提供方受理后、ID 落盘前崩溃，任务状态仍不确定，重跑前先检查提供方控制台。

```bash
python -m railmind collect-augment --output-dir outputs/generated \
  --output data/synthetic/manifest.jsonl
```

图像留在 `outputs/generated`，后续标注和审核将它作为 `--image-root`。清单保存相对路径、实际尺寸和来源关系，包含 `target_labels`，不将生成目标自动视为 `gt_labels`。

## 视觉质量审核

逐图检查实际缺陷、尺度、几何、真实感、额外缺陷、迁移伪影及残留标注。用 `review` 格式写入已确认类别。SFT 还需审核过的教师证据；RL 需审核并确认类别的合成图像；检测器训练需要审核过的**实际像素框**，不能在生成式编辑后沿用原框。

记录审核者、图像哈希与通过/拒绝结果。失败样本和原因留在本地供审计，拒绝样本不进入训练。训练前对所有验证/测试清单执行来源泄漏检查。生成图像和监督样本属于训练数据，不上传仓库。
