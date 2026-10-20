import json
import struct
import zlib
from pathlib import Path

from . import augment, data, evaluation
from .reward import category_reward
from .schema import save


def fixture_png(path, anomaly):
    """Procedural schematic fixture; no real railway image is redistributed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    def chunk(kind, value):
        return struct.pack(">I", len(value)) + kind + value + struct.pack(">I", zlib.crc32(kind + value) & 0xffffffff)
    scanlines = bytearray()
    for y in range(32):
        scanlines.append(0)
        for x in range(48):
            color = (235, 241, 238)
            if x in (10, 11, 35, 36) or y in (12, 13, 22, 23):
                color = (95, 103, 107)
            if anomaly and 14 <= x < 22 and 16 <= y < 24:
                color = (183, 55, 62)
            scanlines.extend(color)
    header = struct.pack(">IIBBBBB", 48, 32, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))


def demo(output):
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Demo output is not empty; select a fresh output directory")
    image_root = root / "images"
    fixture_png(image_root / "steel_bridge/defect.png", True)
    fixture_png(image_root / "steel_bridge/background.png", False)
    coco = {"images": [{"id": "defect", "file_name": "steel_bridge/defect.png", "width": 48, "height": 32},
                       {"id": "background", "file_name": "steel_bridge/background.png", "width": 48, "height": 32}],
            "categories": [{"id": 0, "name": "nonut"}],
            "annotations": [{"id": 1, "image_id": "defect", "category_id": 0, "bbox": [14, 16, 8, 8]}]}
    save(root / "coco.json", coco)
    data.coco_manifest(root / "coco.json", image_root, "train", root / "manifest.jsonl")
    from .schema import load
    rows = load(root / "manifest.jsonl")
    decisions = []
    for row in rows:
        row["teacher"] = {"scene": row["scene"], "anomaly_classes": row["gt_labels"],
                          "evidence": "Synthetic schematic fixture only, not a teacher model result.",
                          "risk": "not_assessed"}
        decisions.append({"id": row["id"], "sha256": row["sha256"], "approved": True,
                          "verified_labels": row["gt_labels"], "reviewer": "offline-demo-fixture"})
    save(root / "teacher.jsonl", rows)
    save(root / "decisions.json", decisions)
    data.apply_review(rows, decisions, image_root, root / "reviewed.jsonl")
    data.export_training(rows, image_root, root / "sft/train.json", "sft")
    augmented = [dict(rows[0], source="synthetic", lineage_sha256=[])]
    data.export_training(augmented, image_root, root / "rl/train.jsonl", "rl")
    augment.plan([rows[1]], [rows[0]], [{"labels": ["nonut"], "count": 2}], 42, root / "augmentation_tasks.jsonl")
    predictions = [{"id": row["id"], "sha256": row["sha256"], "anomaly_classes": row["gt_labels"],
                    "detections": [{**box, "score": 0.99} for box in row["instances"]]} for row in rows]
    save(root / "predictions.jsonl", predictions)
    result = evaluation.evaluate(rows, predictions, root / "metrics.json")
    reward = category_reward(json.dumps({"anomaly_classes": ["nonut"]}), json.dumps({"anomaly_classes": ["nonut"]}))
    assert result["detection"]["micro"]["f1"] == 1.0 and reward == 1.0
    summary = {"offline_demo": "PASS", "images": 2, "reward": reward, "output": str(root),
               "note": "Fixture metrics are not paper results. No API calls, GPUs or training were used."}
    save(root / "summary.json", summary)
    return summary
