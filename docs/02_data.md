# Data conventions and dataset splits

[English](../README.md) · [简体中文](zh-CN/02_data.md)


## Standard classes

English identifiers are shared by code, prompts and evaluation.

| Identifier | Meaning |
|---|---|
| nest | Bird nest on a catenary pole or power tower |
| nonut | Missing steel-bridge nut/bolt |
| rustyfence | Corroded bridge guardrail |
| rustypaint | Corroded steel-bridge coating |
| peeling | Peeling steel-bridge coating |
| noclip | Missing rail fastening clip |
| snappedclip | Broken rail fastening clip |
| debris | Accumulated environmental debris |
| plastic | Environmental plastic film |
| rustypole | Corroded noise-barrier column |
| agedcement | Deteriorated noise-barrier mortar |
| rustyplate | Corroded noise-barrier panel |
| loosenut | Loose tower antenna-clamp nut |

The six scenes are `steel_bridge`, `rail_track`, `environment`, `catenary_pole`, `noise_barrier` and `power_tower`. Legacy Chinese directory aliases are explicitly normalized. Interpret COCO categories by `name`; do not assume consistent numeric category IDs across datasets.

## Build an image manifest

Store decoded images by scene. COCO image entries require `id`, `file_name`, `width` and `height`; annotations require image/category IDs and **xywh pixel boxes**. The manifest converts them to **xyxy pixel boxes**.

```bash
python -m railmind manifest --coco data/train/annotations.coco.json \
  --image-root data/train/images --split train --output data/train/manifest.jsonl
python -m railmind manifest --coco data/test/annotations.coco.json \
  --image-root data/test/images --split test --output data/test/manifest.jsonl
python -m railmind verify --manifest data/train/manifest.jsonl --image-root data/train/images
python -m railmind leakage data/train/manifest.jsonl data/test/manifest.jsonl
```

Unlabeled images may use an empty COCO `annotations` array. Empty imported labels mean **unannotated**, not reviewed negatives. Until reviewed, such images cannot serve as confirmed normal backgrounds or references. Augmentation pools require manually supported annotations; negative SFT samples also require review.

The following is a fictional format example, not a project sample:

```json
{"id":"example-01", "image":"rail_track/example.jpg", "scene":"rail_track", "split":"train", "source":"real", "sha256":"SHA256_OF_IMAGE", "width":1920, "height":1080, "gt_labels":["noclip"], "instances":[{"label":"noclip", "bbox":[100,200,160,240]}]}
```

Image paths are relative to the supplied root; `..` and escaping symlinks are prohibited. Exports resolve paths on the current machine, so re-export after moving data. Generated absolute training paths are local artifacts and must not be committed.

## Leakage and provenance

Group splits by flight, route, sequence or provenance to prevent near duplicates crossing splits. Hash checks detect identical images and recorded augmentation-background/reference leakage. They cannot recognize visually similar crops or reconstruct missing acquisition metadata. Split source groups before producing training derivatives.

Never use evaluation images as augmentation backgrounds or references. Label synthetic benchmark extensions separately and fix them before model selection. A count of 820 alone does not establish whether additional images are real or synthetic; preserve provenance evidence.

## Benchmark versions

See the [inventory](source_inventory.json). The complete audited source COCO has 820 image records. Derive the 810 comparison subset explicitly from the corresponding private annotations, recording the source SHA256 and subset IDs. Do not apply its filtering rule to other datasets. The historical original 810 annotation file has not been independently recovered.

The approximately 62 GiB private reproduction package contains additional training materials. It is not part of the GitHub release and is not required for the CPU demo.

Training/evaluation datasets, manifests and subset IDs are not distributed. Authorized local users can prepare a private ID list outside the repository:

```bash
python -m railmind subset --manifest data/test/manifest.jsonl \
  --ids /path/to/private/subset_ids.json --output data/test/manifest_810.jsonl
```

Subset predictions must contain exactly the selected IDs; filter model outputs explicitly. The default complete benchmark refers to the private 820-image version, not an included dataset.
