# Data Contracts And Splits

## Canonical Categories

| Name | Meaning |
|---|---|
| nest | Nest on catenary pole or power tower |
| nonut | Missing steel bridge nut/bolt |
| rustyfence | Bridge guardrail corrosion |
| rustypaint | Steel bridge coating corrosion |
| peeling | Steel bridge coating peeling |
| noclip | Missing rail fastener clip |
| snappedclip | Fractured rail fastener clip |
| debris | Accumulated environmental debris |
| plastic | Environmental plastic film |
| rustypole | Noise barrier column corrosion |
| agedcement | Noise barrier mortar degradation |
| rustyplate | Noise barrier panel corrosion |
| loosenut | Loose tower antenna clamp nut |

Scenes: `steel_bridge`, `rail_track`, `environment`, `catenary_pole`, `noise_barrier`, `power_tower`. Chinese folder names from the server are normalized explicitly. COCO category IDs are interpreted via their `name`, never assumed to have one universal ordering.

## Build A Manifest

Place decoded images under scene folders. Each COCO image needs `id`, `file_name`, `width`, `height`; each annotation needs an image/category ID and an **xywh pixel** box. The manifest converts boxes to **xyxy pixels**.

```bash
python -m railmind manifest --coco data/train/annotations.coco.json \
  --image-root data/train/images --split train --output data/train/manifest.jsonl
python -m railmind manifest --coco data/test/annotations.coco.json \
  --image-root data/test/images --split test --output data/test/manifest.jsonl
python -m railmind verify --manifest data/train/manifest.jsonl --image-root data/train/images
python -m railmind leakage data/train/manifest.jsonl data/test/manifest.jsonl
```

Unannotated images can use COCO with an empty `annotations` array. An empty imported label list then means **no annotation**, not a reviewed negative image; do not use it as verified background/reference data until reviewed. The planning command expects human annotation-backed background/reference manifests. Negative SFT examples require explicit human review.

Example record:

```json
{"id":"example-01", "image":"rail_track/example.jpg", "scene":"rail_track", "split":"train", "source":"real", "sha256":"SHA256_OF_IMAGE", "width":1920, "height":1080, "gt_labels":["noclip"], "instances":[{"label":"noclip", "bbox":[100,200,160,240]}]}
```

Paths must be relative to the supplied image root, without `..` or symlink escapes. Data exports resolve image paths on the current machine; rebuild exports after moving data. Generated training paths are local run artifacts, not committed configs.

## Leakage And Lineage

Split by flight/route/sequence or source group to avoid near-duplicate leakage. The hash checker catches exact duplicate images and recorded background/reference lineage across splits; it does **not** detect visually similar crops or missing acquisition metadata. Make the group split before generating training derivatives.

Never use benchmark images as augmentation backgrounds or references. Synthetic benchmark extensions must be separately identified and frozen before model selection. The current 820 annotation count does not by itself establish whether every added image is real or synthetic; keep source provenance for those ten images.

## Benchmark Versions

See the audited counts in [source inventory](source_inventory.json). Default: all 820 records in the current source COCO. For a comparison-only 810 subset, explicitly retain IDs 1-810 in this specific annotation file, filter its annotations to those IDs, and record the source annotation SHA256 plus subset IDs. Do not apply this ID rule to another dataset. The old historical annotation file has not been independently recovered.

The private 62 GiB reproduction bundle contains full training material. It is not required for the public CPU demo, and is not embedded in this GitHub payload.

Dataset manifests and subset ID lists are withheld. If you have authorized local data, create a private subset ID list outside the repository and pass its path:

```bash
python -m railmind subset --manifest data/test/manifest.jsonl \
  --ids /path/to/private/subset_ids.json --output data/test/manifest_810.jsonl
```

Evaluate that subset with predictions containing exactly those IDs; filter model outputs explicitly as well. The full 820 remains the default benchmark in this release.
