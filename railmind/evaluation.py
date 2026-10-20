from collections import defaultdict
import math

from .schema import CLASSES, SCENES, bbox, labels, load, save, unique


def metrics(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
            "f1": 2*p*r/(p+r) if p+r else 0.0}


def mean_metrics(items):
    return {key: sum(item[key] for item in items) / len(items) if items else 0.0
            for key in ("precision", "recall", "f1")}


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union else 0.0


def evaluate(gt, predictions, output, threshold=0.5, allow_missing=False):
    if not 0 < threshold <= 1:
        raise ValueError("IoU must be in (0,1]")
    if not gt:
        raise ValueError("Empty evaluation set")
    unique(gt)
    unique(predictions)
    by_id = {r["id"]: r for r in predictions}
    ids = {r["id"] for r in gt}
    if set(by_id) - ids:
        raise ValueError("Prediction IDs outside the evaluated split")
    missing = ids - set(by_id)
    if missing and not allow_missing:
        raise ValueError(f"Missing predictions for {len(missing)} images; use --allow-missing to count them as empty")
    class_counts = {c: [0, 0, 0] for c in CLASSES}
    detection_counts = {c: [0, 0, 0] for c in CLASSES}
    scene_counts = {s: [0, 0, 0] for s in SCENES}
    has_detection = any("detections" in r for r in predictions)
    coverage = defaultdict(int)
    for row in gt:
        coverage[row["scene"]] += 1
        pred = by_id.get(row["id"], {"id": row["id"], "anomaly_classes": [], "detections": []})
        if "prediction" in pred:
            pred = {**pred, **pred["prediction"]}
        if "sha256" in pred and pred["sha256"] != row["sha256"]:
            raise ValueError("Prediction references a different image")
        pset, gset = set(labels(pred["anomaly_classes"])), set(labels(row["gt_labels"]))
        for c in CLASSES:
            class_counts[c][0] += int(c in pset and c in gset)
            class_counts[c][1] += int(c in pset and c not in gset)
            class_counts[c][2] += int(c not in pset and c in gset)
        if not has_detection:
            continue
        if "detections" not in pred:
            raise ValueError("Mixed classification/detection prediction formats")
        ground = row.get("instances")
        if ground is None:
            raise ValueError("Detection evaluation requires box GT for every image")
        for item in ground:
            labels([item["label"]])
            bbox(item["bbox"], row["width"], row["height"])
        matched = set()
        detections = pred["detections"]
        for item in detections:
            labels([item["label"]])
            bbox(item["bbox"], row["width"], row["height"])
            if not math.isfinite(float(item["score"])) or not 0 <= float(item["score"]) <= 1:
                raise ValueError("Detection score must be in [0,1]")
        for item in sorted(detections, key=lambda x: x["score"], reverse=True):
            candidates = [(iou(item["bbox"], box["bbox"]), j) for j, box in enumerate(ground)
                          if j not in matched and box["label"] == item["label"]]
            overlap, index = max(candidates, default=(0.0, -1))
            correct = overlap >= threshold and index >= 0
            if correct:
                matched.add(index)
            column = 0 if correct else 1
            detection_counts[item["label"]][column] += 1
            scene_counts[row["scene"]][column] += 1
        for index, box in enumerate(ground):
            if index not in matched:
                detection_counts[box["label"]][2] += 1
                scene_counts[row["scene"]][2] += 1
    classification = {c: metrics(*counts) for c, counts in class_counts.items()}
    result = {"protocol": "railmind-release-v1", "images": len(gt), "missing_predictions": len(missing),
              "scene_image_counts": dict(coverage), "classification": {"per_category": classification,
              "macro_13_categories": mean_metrics(list(classification.values()))},
              "notes": ["Metrics are fractions, not percentages; unsupported categories score zero.",
                        "Classification uses final category sets; detection uses score-ordered one-to-one greedy matching.",
                        "These are fixed-threshold PR/RC/F1, not COCO AP or mAP."]}
    if has_detection:
        dc = {c: metrics(*counts) for c, counts in detection_counts.items()}
        ds = {s: metrics(*counts) for s, counts in scene_counts.items()}
        result["detection"] = {"iou": threshold, "per_category": dc, "per_scene": ds,
            "macro_13_categories": mean_metrics(list(dc.values())),
            "mean_present_scenes": mean_metrics([ds[s] for s in SCENES if coverage[s]]),
            "mean_6_scenes": mean_metrics(list(ds.values())) if all(coverage[s] for s in SCENES) else None,
            "micro": metrics(*(sum(x[i] for x in detection_counts.values()) for i in range(3)))}
    save(output, result)
    return result
