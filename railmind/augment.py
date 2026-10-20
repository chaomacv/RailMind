import json
import os
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .api import image_uri, request_json
from .schema import SCENE_CLASSES, checked_path, digest, labels, load, save, train_only


def plan(backgrounds, references, targets, seed, output):
    train_only(backgrounds)
    train_only(references)
    if any(not r.get("review", {}).get("approved") or r.get("review", {}).get("sha256") != r["sha256"]
           for r in backgrounds + references):
        raise ValueError("Background/reference pools require image-hash-matched review decisions")
    rng = random.Random(seed)
    tasks = []
    for target in targets:
        wanted = labels(target["labels"])
        if not wanted or not isinstance(target["count"], int) or target["count"] < 1:
            raise ValueError("Generation targets need positive counts and nonempty labels")
        pairs = [(bg, ref) for bg in backgrounds for ref in references
                 if bg["scene"] == ref["scene"]
                 and set(wanted).issubset(SCENE_CLASSES[bg["scene"]])
                 and set(wanted).issubset(ref.get("gt_labels", []))
                 and not set(wanted).intersection(bg.get("gt_labels", []))
                 and bg["sha256"] != ref["sha256"]]
        if not pairs:
            raise ValueError(f"No train-only, scene-aligned background/reference pair for {wanted}")
        for index in range(target["count"]):
            bg, ref = rng.choice(pairs)
            tasks.append({"id": f"{'_'.join(wanted)}-{len(tasks):05d}", "scene": bg["scene"],
                "split": "train", "target_labels": wanted, "background": bg["image"],
                "reference": ref["image"], "background_sha256": bg["sha256"],
                "reference_sha256": ref["sha256"], "seed": seed,
                "review_status": "pending"})
    save(output, tasks)
    return {"tasks": len(tasks), "seed": seed, "output": str(output)}


def run(tasks, background_root, reference_root, prompt_config, output_dir, execute=False):
    from .schema import unique
    unique(tasks)
    config = load(prompt_config)
    mapping = config["category_mapping"]
    out = Path(output_dir)
    for task in tasks:
        if task["split"] != "train" or not set(labels(task["target_labels"])).issubset(SCENE_CLASSES[task["scene"]]):
            raise ValueError("Invalid generation target scene/split")
        if Path(task["id"]).name != task["id"] or task["id"] in (".", ".."):
            raise ValueError("Invalid task ID")
        for field, root in [("background", background_root), ("reference", reference_root)]:
            if digest(checked_path(root, task[field])) != task[field + "_sha256"]:
                raise ValueError("Generation input changed since planning")
    if not execute:
        return {"dry_run": True, "tasks": len(tasks), "output": str(out),
                "note": "Provider adapter: legacy GRSAI async NanoBanana protocol; verify current provider before execution"}
    host = os.environ.get("NANO_BANANA_BASE_URL", "").rstrip("/")
    key = os.environ.get("NANO_BANANA_API_KEY", "")
    model = os.environ.get("NANO_BANANA_MODEL", "nano-banana-pro")
    if not host:
        raise ValueError("Set NANO_BANANA_BASE_URL")
    out.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        state = out / (task["id"] + ".state.json")
        prompt = "\n".join(mapping[c] for c in task["target_labels"])
        expected = {"task": task, "prompt": prompt, "model": model, "provider": host,
                    "prompt_config_sha256": digest(prompt_config)}
        if state.exists():
            saved = load(state)
            if any(saved.get(k) != v for k, v in expected.items()):
                raise ValueError("Generation resume configuration changed")
        else:
            # Persist the accepted task ID before polling, so reconnecting does not resubmit it.
            response = request_json(host + "/v1/draw/nano-banana", {"model": model, "prompt": prompt,
                "urls": [image_uri(checked_path(background_root, task["background"])),
                         image_uri(checked_path(reference_root, task["reference"]))],
                "aspectRatio": "auto", "imageSize": "4K", "webHook": "-1", "shutProgress": False}, key)
            if response.get("code") not in (0, 200):
                raise RuntimeError("NanoBanana submission rejected")
            task_id = response.get("data")
            if isinstance(task_id, dict):
                task_id = task_id.get("id")
            if not isinstance(task_id, (str, int)):
                raise ValueError("Provider did not return a task ID")
            saved = {**expected, "provider_task_id": task_id, "status": "submitted"}
            save(state, saved)
        if saved["status"] == "downloaded":
            image = out / saved["image"]
            if not image.is_file() or digest(image) != saved["sha256"]:
                raise ValueError("Previously generated image missing or changed")
            continue
        for _ in range(100):
            response = request_json(host + "/v1/draw/result", {"id": saved["provider_task_id"]}, key, timeout=30)
            info = response.get("data") or {}
            if info.get("status") == "failed":
                raise RuntimeError("Provider generation failed; inspect provider dashboard")
            if info.get("status") == "succeeded":
                download_url = info["results"][0]["url"]
                if urllib.parse.urlsplit(download_url).scheme != "https":
                    raise ValueError("Expected HTTPS generated-image URL")
                with urllib.request.urlopen(download_url, timeout=60) as response:
                    content = response.read(30 * 1024 * 1024 + 1)
                if len(content) > 30 * 1024 * 1024:
                    raise ValueError("Generated image exceeds 30 MiB")
                from PIL import Image
                import io
                with Image.open(io.BytesIO(content)) as im:
                    im.load()
                    im.convert("RGB").save(out / (task["id"] + ".png"))
                saved.update(status="downloaded", image=task["id"] + ".png",
                             sha256=digest(out / (task["id"] + ".png")))
                save(state, saved)
                break
            time.sleep(3)
        else:
            raise TimeoutError("Generation pending; rerun to resume polling the saved task ID")
    return {"tasks": len(tasks), "output": str(out)}


def collect(output_dir, output):
    from PIL import Image
    rows = []
    root = Path(output_dir)
    for state in sorted(root.glob("*.state.json")):
        record = load(state)
        if record["status"] != "downloaded":
            continue
        task = record["task"]
        path = checked_path(root, record["image"])
        if digest(path) != record["sha256"]:
            raise ValueError("Generated image hash mismatch")
        with Image.open(path) as image:
            width, height = image.size
        rows.append({"id": task["id"], "image": record["image"], "sha256": record["sha256"],
            "scene": task["scene"], "split": "train", "source": "synthetic", "width": width, "height": height,
            "target_labels": task["target_labels"], "lineage_sha256": [task["background_sha256"], task["reference_sha256"]],
            "generation_model": record["model"], "provider_task_id": record["provider_task_id"],
            "prompt_sha256": digest(state), "review_status": "pending"})
    if not rows:
        raise ValueError("No completed generation states")
    save(output, rows)
    return {"images": len(rows), "output": str(output), "gt_status": "pending visual review, not inferred from generation targets"}
