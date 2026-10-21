#!/usr/bin/env python3
"""在独立的 RailMind 发布目录中整理经审计的源码快照。"""
import argparse
import ast
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    source = Path(args.source).resolve()
    release = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(release))
    if release == source or not release.is_relative_to(source):
        raise ValueError("Release must be a separate child of the source workspace")
    snapshots = {
        "groundingdino": ("railmind_training_repro_20260815/groundingdino/code", ["models", "datasets", "util"],
            ["main.py", "engine.py", "requirements.txt", "LICENSE", "README.md", "data_format.md"]),
        "llamafactory": ("railmind_training_repro_20260815/qwen3vl/code", ["src", "requirements"],
            ["setup.py", "setup.cfg", "pyproject.toml", "MANIFEST.in", "requirements.txt", "LICENSE", "README.md"]),
        "ms-swift": ("RL_swift/ms-swift-main", ["swift", "requirements"],
            ["setup.py", "setup.cfg", "pyproject.toml", "MANIFEST.in", "requirements.txt", "LICENSE", "README.md"]),
    }
    allowed = {".py", ".json", ".txt", ".md", ".toml", ".cfg", ".in", ".cpp", ".h", ".cu", ".c", ".yaml", ".yml", ".jinja"}
    records = {}
    for name, (origin, directories, files) in snapshots.items():
        base = source / origin
        target = release / "third_party" / name
        selected = [base / f for f in files if (base / f).is_file()]
        for directory in directories:
            if (base / directory).exists():
                selected.extend(p for p in (base / directory).rglob("*") if p.is_file() and not p.is_symlink()
                    and "__pycache__" not in p.parts and p.suffix in allowed)
        if not (base / "LICENSE").is_file():
            raise ValueError(f"Missing third-party license for {name}")
        mapping = {}
        for path in selected:
            if path.stat().st_size > 5_000_000:
                raise ValueError(f"Unexpected large source file: {path.name}")
            rel = path.relative_to(base)
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            mapping[str(rel)] = sha(dest)
        records[name] = {"source": origin, "files": mapping, "snapshot_type": "所选源码副本，不代表已确认上游 Git 提交"}
    config = source / "railmind_training_repro_20260815/groundingdino/code/config/cfg_railway.py"
    dest = release / "configs/dino/cfg_railway.py"
    shutil.copy2(config, dest)
    prompt = source / "agent/RailwayCARS/pipeline/code/prompt/config/AnomalyDetection.json"
    payload = json.loads(prompt.read_text(encoding="utf-8"))
    payload["category_mapping"].pop("negative_prompt", None)
    (release / "prompts/augmentation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    records["nano_prompts"] = {"source": str(prompt.relative_to(source)), "source_sha256": sha(prompt),
        "change": "Removed negative_prompt incorrectly nested as a category; preserved 13 generation prompts."}
    coco_path = source / "agent/RailwayCARS/dataset/val/810_test_modified/annotations_coco.json"
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    from railmind.schema import SCENE_ALIASES
    counts = Counter(SCENE_ALIASES.get(im["file_name"].split('/')[0], im["file_name"].split('/')[0]) for im in coco['images'])
    missing = sum(not (coco_path.parent / im["file_name"]).is_file() for im in coco["images"])
    original = [im for im in coco["images"] if int(im["id"]) <= 810]
    extra = [im for im in coco["images"] if int(im["id"]) > 810]
    audit = {"benchmark": {"source": str(coco_path.relative_to(source)), "source_sha256": sha(coco_path),
        "coco_records": len(coco["images"]), "annotations": len(coco["annotations"]),
        "disk_images": sum(p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} for p in coco_path.parent.rglob('*') if p.is_file()),
        "missing_images": missing, "scenes": dict(counts),
        "extended_image_count": len(extra),
        "original_810_derivation": "按对应来源文件中的图像 ID 显式派生 810 比较子集；历史原始标注文件未独立找回",
        "original_810_count": len(original), "extra_scene_counts": dict(Counter(SCENE_ALIASES.get(im["file_name"].split('/')[0], '') for im in extra))},
        "training": {}, "historical_railway_rl_launch_recovered": False,
        "paper_identity": "RailMind: Multimodal Reasoning for Railway Infrastructure Security (用户提供的 13 页论文草稿)",
        "audit_date": "2026-10-07", "publication_status": "本地候选发布目录"}
    for label, rel in [("server_sft", "agent/qwen_vl/data/qwen_sft_mixed_24265_noenc.json"),
                       ("portable_sft", "railmind_training_repro_20260815/qwen3vl/data/qwen_sft_mixed_24265_portable.json")]:
        path = source / rel
        value = json.loads(path.read_text(encoding="utf-8"))
        audit["training"][label] = {"source": rel, "records": len(value), "sha256": sha(path)}
    for label, rel in [("dino_train2", "agent/RailwayCARS/dataset/train/3334_train2_with_nest_noclip/annotations_converted.jsonl")]:
        path = source / rel
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        audit["training"][label] = {"source": rel, "images": len(values),
            "instances": sum(len(v.get('detection', {}).get('instances', [])) for v in values), "sha256": sha(path)}
    (release / "docs/source_inventory.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    # 数据样本 ID 清单保持私有，不复制到发布目录。
    (release / "third_party/source_lock.json").write_text(json.dumps(records, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    # 不要复制私有论文 PDF、原始图像、用户记录、含凭据的 API 脚本或模型权重。
    for folder in ["configs", "prompts", "training"]:
        shutil.copytree(release / folder, release / "railmind/assets" / folder, dirs_exist_ok=True)
    secrets = []
    config = ast.parse((source / "agent/network_app0.11/config.py").read_text(encoding="utf-8"))
    for node in ast.walk(config):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "api_key" and isinstance(value, ast.Constant):
                    secrets.append(value.value)
    violations = []
    for path in release.rglob('*'):
        if not path.is_file() or path.suffix in {".png", ".pyc"}: continue
        text = path.read_text(encoding='utf-8', errors='replace')
        if any(secret and secret in text for secret in secrets):
            violations.append(str(path.relative_to(release)))
    if violations:
        raise ValueError("Source credentials found in release files: " + ', '.join(violations))
    print(json.dumps({"snapshot_files": sum(len(r.get('files',{})) for r in records.values()),
        "benchmark": audit['benchmark'], "training": audit['training'], "source_credentials_found": False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
