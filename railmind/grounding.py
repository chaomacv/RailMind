import sys
from pathlib import Path

from .schema import bbox, checked_path, labels, load, save, unique
from .data import verify_manifest


def ground(manifest, semantic, image_root, model_config, checkpoint, output, code_root,
           box_threshold=0.35, text_threshold=0.35, device="cuda", execute=False, text_encoder=None):
    rows = load(manifest)
    verify_manifest(rows, image_root)
    semantics = load(semantic)
    unique(semantics)
    predictions = {r["id"]: r for r in semantics}
    if set(predictions) != {r["id"] for r in rows}:
        raise ValueError("Semantic predictions must cover exactly this manifest")
    for row in rows:
        if predictions[row["id"]]["sha256"] != row["sha256"]:
            raise ValueError("Semantic image hash mismatch")
        labels(predictions[row["id"]]["prediction"]["anomaly_classes"])
    if not execute:
        return {"dry_run": True, "images": len(rows), "checkpoint": str(checkpoint),
                "box_threshold": box_threshold, "text_threshold": text_threshold}
    if not 0 <= box_threshold <= 1 or not 0 <= text_threshold <= 1:
        raise ValueError("Detection thresholds must be in [0,1]")
    sys.path.insert(0, str(Path(code_root).resolve()))
    import torch
    from PIL import Image
    import datasets.transforms as transforms
    from models import build_model
    from util.slconfig import SLConfig
    from util.utils import clean_state_dict
    args = SLConfig.fromfile(str(model_config))
    args.device = device
    if text_encoder:
        args.text_encoder_type = str(Path(text_encoder).resolve())
    model = build_model(args)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(clean_state_dict(state["model"]), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("Checkpoint and model config do not match; select the source-matched detector")
    model.eval().to(device)
    transform = transforms.Compose([transforms.RandomResize([800], max_size=1333),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    results = []
    for row in rows:
        wanted = labels(predictions[row["id"]]["prediction"]["anomaly_classes"])
        detections = []
        dropped = 0
        if wanted:
            with Image.open(checked_path(image_root, row["image"])) as source:
                if source.size != (row["width"], row["height"]):
                    raise ValueError("Manifest dimensions differ from decoded image")
                image, _ = transform(source.convert("RGB"), None)
            caption = " . ".join(wanted) + " ."
            with torch.no_grad():
                prediction = model(image[None].to(device), captions=[caption])
            logits = prediction["pred_logits"].cpu().sigmoid()[0]
            boxes = prediction["pred_boxes"].cpu()[0]
            tokenized = model.tokenizer(caption)
            mask = logits.max(dim=1)[0] > box_threshold
            for box, logit in zip(boxes[mask].tolist(), logits[mask]):
                selected = [token for token, keep in zip(tokenized["input_ids"], (logit > text_threshold).tolist()) if keep]
                phrase = model.tokenizer.decode(selected, skip_special_tokens=True).replace(".", " ")
                score = float(logit.max())
                matches = [c for c in wanted if c in phrase.lower().split()]
                if len(matches) != 1:
                    dropped += 1
                    continue
                cx, cy, w, h = box
                coords = [max(0, (cx-w/2)*row["width"]), max(0, (cy-h/2)*row["height"]),
                          min(row["width"], (cx+w/2)*row["width"]), min(row["height"], (cy+h/2)*row["height"])]
                detections.append({"label": matches[0], "bbox": bbox(coords, row["width"], row["height"]), "score": score})
        results.append({"id": row["id"], "sha256": row["sha256"], "anomaly_classes": wanted,
                        "detections": detections, "unmapped_phrases": dropped,
                        "box_threshold": box_threshold, "text_threshold": text_threshold})
        save(output, results)
    return {"images": len(results), "unmapped_phrases": sum(r["unmapped_phrases"] for r in results), "output": str(output)}
