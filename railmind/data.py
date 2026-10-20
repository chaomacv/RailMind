from collections import Counter, defaultdict
from pathlib import Path

from .schema import bbox, checked_path, digest, labels, load, save, scene, train_only, unique


def coco_manifest(coco_path, image_root, split, output):
    coco = load(coco_path)
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    labels(list(categories.values()))
    instances = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            raise ValueError("Crowd annotations require a separate COCO evaluation protocol")
        instances[ann["image_id"]].append(ann)
    rows = []
    for image in coco["images"]:
        filename = image["file_name"]
        path = checked_path(image_root, filename)
        detected = []
        for ann in instances[image["id"]]:
            x, y, w, h = ann["bbox"]
            detected.append({"label": categories[ann["category_id"]],
                             "bbox": bbox([x, y, x+w, y+h], image["width"], image["height"])})
        rows.append({
            "id": str(image["id"]), "image": filename,
            "scene": scene(image.get("scene", Path(filename).parts[0])),
            "split": split, "source": "real", "sha256": digest(path),
            "width": image["width"], "height": image["height"],
            "gt_labels": sorted({x["label"] for x in detected}), "instances": detected,
        })
    unique(rows)
    if set(instances) - {x["id"] for x in coco["images"]}:
        raise ValueError("Orphan annotations in COCO input")
    save(output, rows)
    return {"images": len(rows), "instances": sum(len(x["instances"]) for x in rows),
            "scenes": dict(Counter(x["scene"] for x in rows)), "manifest": str(output)}


def verify_manifest(rows, image_root):
    unique(rows)
    for row in rows:
        path = checked_path(image_root, row["image"])
        if digest(path) != row["sha256"]:
            raise ValueError(f"Image changed: {row['id']}")
        scene(row["scene"])
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("Unknown split")
        labels(row.get("gt_labels", []))
    return {"images": len(rows), "verified": True}


def check_leakage(manifests):
    seen = {}
    for manifest in manifests:
        for row in load(manifest):
            for sha in [row["sha256"], *row.get("lineage_sha256", [])]:
                previous = seen.get(sha)
                if previous and previous != row["split"]:
                    raise ValueError(f"Image/augmentation lineage leaks across {previous} and {row['split']}")
                seen[sha] = row["split"]
    return {"checked_manifests": len(manifests), "unique_hashes": len(seen), "leakage": False}


def apply_review(records, reviews, image_root, output):
    unique(records)
    unique(reviews)
    decisions = {r["id"]: r for r in reviews}
    if set(decisions) - {r["id"] for r in records}:
        raise ValueError("Unknown review IDs")
    for row in records:
        decision = decisions.get(row["id"])
        if not decision:
            continue
        if decision.get("approved") not in (True, False) or not isinstance(decision.get("approved"), bool):
            raise ValueError("approved must be a boolean")
        current = digest(checked_path(image_root, row["image"]))
        if current != row["sha256"] or decision.get("sha256") != current:
            raise ValueError("Review must identify the exact image SHA256")
        if not decision.get("reviewer"):
            raise ValueError("Reviewer is required")
        reviewed_labels = labels(decision["verified_labels"])
        row["review"] = {"approved": decision["approved"], "reviewer": decision["reviewer"],
                         "verified_labels": reviewed_labels, "sha256": current}
        if decision["approved"]:
            if "gt_labels" in row and row["gt_labels"] != reviewed_labels:
                row["original_gt_labels"] = row["gt_labels"]
            row["gt_labels"] = reviewed_labels
        if "instances" in decision:
            boxes = decision["instances"]
            for item in boxes:
                labels([item["label"]])
                bbox(item["bbox"], row["width"], row["height"])
            if sorted({item["label"] for item in boxes}) != reviewed_labels:
                raise ValueError("Reviewed boxes and verified labels disagree")
            row["instances"] = boxes
        if decision["approved"] and "teacher" in row:
            teacher = row["teacher"].copy()
            teacher["anomaly_classes"] = reviewed_labels
            if teacher != row["teacher"] and not decision.get("corrected_teacher"):
                raise ValueError("Changed teacher labels require corrected_teacher with consistent evidence")
            if decision.get("corrected_teacher"):
                from .schema import parse_answer
                import json
                teacher = parse_answer(json.dumps(decision["corrected_teacher"]))
                if teacher["anomaly_classes"] != reviewed_labels:
                    raise ValueError("Corrected teacher labels do not match verified_labels")
            row["teacher"] = teacher
    save(output, records)
    return {"records": len(records), "approved": sum(r.get("review", {}).get("approved", False) for r in records)}


def export_training(records, image_root, output, mode):
    train_only(records)
    verify_manifest(records, image_root)
    exported = []
    for row in records:
        review = row.get("review", {})
        if not review.get("approved"):
            continue
        if review.get("sha256") != row["sha256"]:
            raise ValueError("Stale review")
        verified = labels(review["verified_labels"])
        image = str(checked_path(image_root, row["image"]))
        if mode == "sft":
            if "teacher" not in row or row["teacher"]["anomaly_classes"] != verified:
                raise ValueError("SFT requires reviewed, label-consistent teacher supervision")
            import json
            exported.append({"messages": [{"role": "user", "content": "<image>Analyze railway anomalies. Return scene, anomaly_classes, evidence and risk as JSON."},
                                          {"role": "assistant", "content": json.dumps(row["teacher"], ensure_ascii=False)}],
                             "images": [image]})
        else:
            if row.get("source") != "synthetic":
                raise ValueError("RL export requires reviewed synthetic samples; separate real-data experiments explicitly")
            import json
            exported.append({"messages": [{"role": "user", "content": "<image>Identify railway anomalies. Return JSON with anomaly_classes."}],
                             "images": [image], "solution": json.dumps({"anomaly_classes": verified})})
    if not exported:
        raise ValueError("No approved training samples")
    save(output, exported)
    if mode == "sft":
        save(Path(output).parent / "dataset_info.json", {"railmind_sft": {"file_name": Path(output).name,
            "formatting": "sharegpt", "columns": {"messages": "messages", "images": "images"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}}})
    return {"exported": len(exported), "mode": mode, "output": str(output)}


def export_dino(records, validation, train_root, val_root, output_dir):
    from .schema import CLASSES
    train_only(records)
    verify_manifest(records, train_root)
    if not validation or any(r["split"] not in {"val", "test"} for r in validation):
        raise ValueError("DINO validation requires an explicit val/test split")
    verify_manifest(validation, val_root)
    train_hashes = {h for r in records for h in [r["sha256"], *r.get("lineage_sha256", [])]}
    val_hashes = {h for r in validation for h in [r["sha256"], *r.get("lineage_sha256", [])]}
    if train_hashes & val_hashes:
        raise ValueError("Training and validation share image or lineage hashes")
    odvg = []
    for r in records:
        review = r.get("review", {})
        if not review.get("approved"):
            continue
        if review.get("sha256") != r["sha256"] or "instances" not in r:
            raise ValueError("DINO export requires reviewed boxes for this exact image")
        if sorted({x["label"] for x in r["instances"]}) != labels(review["verified_labels"]):
            raise ValueError("Reviewed boxes do not match verified_labels")
        instances = [{"bbox": bbox(x["bbox"], r["width"], r["height"]),
                      "label": CLASSES.index(x["label"]), "category": x["label"]} for x in r["instances"]]
        odvg.append({"filename": r["image"], "width": r["width"], "height": r["height"], "detection": {"instances": instances}})
    if not odvg:
        raise ValueError("No approved box-supervised samples")
    out = Path(output_dir).resolve()
    save(out / "train.odvg.jsonl", odvg)
    save(out / "label_map.json", {str(i): c for i, c in enumerate(CLASSES)})
    coco = {"images": [], "annotations": [], "categories": [{"id": i, "name": c} for i, c in enumerate(CLASSES)]}
    for index, r in enumerate(validation):
        coco["images"].append({"id": index, "file_name": r["image"], "width": r["width"], "height": r["height"]})
        for item in r["instances"]:
            x1, y1, x2, y2 = bbox(item["bbox"], r["width"], r["height"])
            coco["annotations"].append({"id": len(coco["annotations"]), "image_id": index,
                "category_id": CLASSES.index(item["label"]), "bbox": [x1, y1, x2-x1, y2-y1],
                "area": (x2-x1)*(y2-y1), "iscrowd": 0})
    save(out / "val.coco.json", coco)
    save(out / "datasets.json", {"train": [{"root": str(Path(train_root).resolve()),
        "anno": str(out / "train.odvg.jsonl"), "label_map": str(out / "label_map.json"), "dataset_mode": "odvg"}],
        "val": [{"root": str(Path(val_root).resolve()), "anno": str(out / "val.coco.json"), "label_map": None, "dataset_mode": "coco"}]})
    return {"training_images": len(odvg), "validation_images": len(validation), "dataset_config": str(out / "datasets.json")}
