# 数据规范与数据集划分

## 标准类别

类别标识符保持英文，供代码、提示词和评测统一使用。

| 标识符 | 含义 |
|---|---|
| nest | 接触网杆或电力塔上的鸟巢 |
| nonut | 钢桥螺母/螺栓缺失 |
| rustyfence | 桥梁护栏锈蚀 |
| rustypaint | 钢桥涂层锈蚀 |
| peeling | 钢桥涂层剥落 |
| noclip | 钢轨扣件弹条缺失 |
| snappedclip | 钢轨扣件弹条断裂 |
| debris | 环境杂物堆积 |
| plastic | 环境塑料薄膜 |
| rustypole | 声屏障立柱锈蚀 |
| agedcement | 声屏障砂浆老化 |
| rustyplate | 声屏障面板锈蚀 |
| loosenut | 塔架天线夹具螺母松动 |

六类场景为 `steel_bridge`（钢桥）、`rail_track`（轨道）、`environment`（周边环境）、`catenary_pole`（接触网杆）、`noise_barrier`（声屏障）、`power_tower`（电力塔）。服务器中文目录名会显式归一化。COCO 类别 ID 按 `name` 解释，不假设不同数据集使用相同编号。

## 构建图像清单（manifest）

将解码后的图像按场景存放。COCO 图像记录需要 `id`、`file_name`、`width`、`height`；实例标注需要图像 ID、类别 ID 和 **xywh 像素坐标**。清单将框转换为 **xyxy 像素坐标**。

```bash
python -m railmind manifest --coco data/train/annotations.coco.json \
  --image-root data/train/images --split train --output data/train/manifest.jsonl
python -m railmind manifest --coco data/test/annotations.coco.json \
  --image-root data/test/images --split test --output data/test/manifest.jsonl
python -m railmind verify --manifest data/train/manifest.jsonl --image-root data/train/images
python -m railmind leakage data/train/manifest.jsonl data/test/manifest.jsonl
```

未标注图像可使用 `annotations` 为空数组的 COCO 文件。导入后标签为空只表示**没有标注**，不能视为已审核的负样本；审核前不能用作已确认的正常背景或参考图。增强计划需要有人工标注依据的背景/参考清单。负样本进入 SFT 前同样需要人工确认。

以下为虚构格式示例，不是项目数据样本：

```json
{"id":"example-01", "image":"rail_track/example.jpg", "scene":"rail_track", "split":"train", "source":"real", "sha256":"SHA256_OF_IMAGE", "width":1920, "height":1080, "gt_labels":["noclip"], "instances":[{"label":"noclip", "bbox":[100,200,160,240]}]}
```

图像路径相对于指定根目录，不得使用 `..` 或符号链接越界。导出时按当前机器解析路径；迁移数据后重新导出。生成的绝对训练路径属于本地运行产物，不提交到仓库。

## 泄漏检查与来源关系

按航次、线路、序列或来源分组划分，避免近重复图像跨集合。哈希检查能发现完全相同图像，以及已记录的增强背景/参考来源跨集合问题；不能识别视觉相似裁剪，也不能补全缺失的采集元数据。先划分来源组，再生成训练衍生数据。

不得用评测图像作为增强背景或参考。合成基准扩展应独立标识，在模型选择前固定。820 条记录本身不能证明新增图像全部真实或全部合成，需保存来源证据。

## 基准版本

审核汇总见[来源清单](source_inventory.json)。当前来源 COCO 完整版本有 820 条图像记录。比较用的 810 子集需在对应私有标注文件中显式筛选图像及标注，记录来源文件 SHA256 和子集 ID。不要将该版本的筛选规则套用于其他数据集；历史原始 810 标注文件尚未独立找回。

约 62 GiB 私有复现包包含更完整的训练材料，不属于 GitHub 发布内容，也不是运行 CPU 示例的前提。

训练集、评测集、manifest 和子集 ID 清单均不发布。有授权的本地数据使用者，可在仓库外准备私有 ID 清单：

```bash
python -m railmind subset --manifest data/test/manifest.jsonl \
  --ids /path/to/private/subset_ids.json --output data/test/manifest_810.jsonl
```

子集预测必须恰好包含对应 ID，模型输出也需显式筛选。文档默认完整基准仍指私有数据的 820 版本，不代表仓库包含该数据集。
