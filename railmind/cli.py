import argparse
import json
import sys
from pathlib import Path

from .schema import load, resource_root

ROOT = resource_root()


def parser():
    p = argparse.ArgumentParser(prog="railmind", description="RailMind data-to-evaluation research workflows")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("demo", help="Run a CPU-only synthetic workflow, no keys or models")
    q.add_argument("--output", default="outputs/demo")
    q = sub.add_parser("manifest", help="Build an image-hashed manifest from COCO annotations")
    q.add_argument("--coco", required=True)
    q.add_argument("--image-root", required=True)
    q.add_argument("--split", required=True, choices=["train", "val", "test"])
    q.add_argument("--output", required=True)
    q = sub.add_parser("verify", help="Verify image hashes and dataset schema")
    q.add_argument("--manifest", required=True)
    q.add_argument("--image-root", required=True)
    q = sub.add_parser("leakage", help="Check image and augmentation lineage across splits")
    q.add_argument("manifests", nargs="+")
    q = sub.add_parser("subset", help="Create an explicit image-ID subset, never infer count from folder name")
    q.add_argument("--manifest", required=True)
    q.add_argument("--ids", required=True, help="JSON list of string IDs")
    q.add_argument("--output", required=True)
    for name, prompt in [("annotate", "teacher.txt"), ("semantic", "semantic.txt")]:
        q = sub.add_parser(name, help="OpenAI-compatible multimodal API; dry-run by default")
        q.add_argument("--manifest", required=True)
        q.add_argument("--image-root", required=True)
        q.add_argument("--prompt", default=str(ROOT / "prompts" / prompt))
        q.add_argument("--output", required=True)
        q.add_argument("--execute", action="store_true")
    q = sub.add_parser("review", help="Apply explicit visual-review decisions to records")
    q.add_argument("--records", required=True)
    q.add_argument("--decisions", required=True)
    q.add_argument("--image-root", required=True)
    q.add_argument("--output", required=True)
    for name in ("export-sft", "export-rl"):
        q = sub.add_parser(name)
        q.add_argument("--records", required=True)
        q.add_argument("--image-root", required=True)
        q.add_argument("--output", required=True)
    q = sub.add_parser("export-dino", help="Export reviewed pixel boxes to ODVG and COCO dataset config")
    for arg in ("records", "validation", "train-root", "val-root", "output-dir"):
        q.add_argument("--" + arg, required=True)
    q = sub.add_parser("plan-augment", help="Plan seeded scene-aligned train-only generation pairs")
    q.add_argument("--backgrounds", required=True)
    q.add_argument("--references", required=True)
    q.add_argument("--targets", required=True)
    q.add_argument("--seed", type=int, default=42)
    q.add_argument("--output", required=True)
    q = sub.add_parser("augment", help="Async NanoBanana gateway adapter; dry-run by default")
    q.add_argument("--tasks", required=True)
    q.add_argument("--background-root", required=True)
    q.add_argument("--reference-root", required=True)
    q.add_argument("--prompts", default=str(ROOT / "prompts/augmentation.json"))
    q.add_argument("--output-dir", required=True)
    q.add_argument("--execute", action="store_true")
    q = sub.add_parser("collect-augment", help="Create pending-review manifest from completed image states")
    q.add_argument("--output-dir", required=True)
    q.add_argument("--output", required=True)
    q = sub.add_parser("ground", help="Use semantic labels as prompts for the source-matched detector")
    for arg in ("manifest", "semantic", "image-root", "checkpoint", "output"):
        q.add_argument("--" + arg, required=True)
    q.add_argument("--code-root", default=str(ROOT / "third_party/groundingdino"))
    q.add_argument("--model-config", default=str(ROOT / "configs/dino/cfg_railway.py"))
    q.add_argument("--text-encoder")
    q.add_argument("--box-threshold", type=float, default=0.35)
    q.add_argument("--text-threshold", type=float, default=0.35)
    q.add_argument("--device", default="cuda")
    q.add_argument("--execute", action="store_true")
    q = sub.add_parser("evaluate", help="Classification and fixed-IoU detection, with explicit macro/micro averages")
    q.add_argument("--manifest", required=True)
    q.add_argument("--predictions", required=True)
    q.add_argument("--iou", type=float, default=0.5)
    q.add_argument("--allow-missing", action="store_true")
    q.add_argument("--output", required=True)
    q = sub.add_parser("train", help="Resolve a versioned framework recipe; dry-run by default")
    q.add_argument("kind", choices=["sft", "rl", "dino"])
    for arg in ("recipe", "model", "dataset", "output"):
        q.add_argument("--" + arg, required=arg in {"recipe", "output"})
    for arg in ("framework-root", "checkpoint", "text-encoder", "dataset-config"):
        q.add_argument("--" + arg)
    q.add_argument("--execute", action="store_true")
    return p


def run(a):
    from . import api, augment, data, evaluation, grounding, training
    if a.command == "demo":
        from .demo import demo
        return demo(a.output)
    if a.command == "manifest":
        return data.coco_manifest(a.coco, a.image_root, a.split, a.output)
    if a.command == "verify":
        return data.verify_manifest(load(a.manifest), a.image_root)
    if a.command == "leakage":
        return data.check_leakage(a.manifests)
    if a.command == "subset":
        from .schema import subset_manifest
        return subset_manifest(load(a.manifest), load(a.ids), a.output)
    if a.command in {"annotate", "semantic"}:
        return api.annotate(a.manifest, a.image_root, a.prompt, a.output, a.command == "annotate", a.execute)
    if a.command == "review":
        return data.apply_review(load(a.records), load(a.decisions), a.image_root, a.output)
    if a.command in {"export-sft", "export-rl"}:
        return data.export_training(load(a.records), a.image_root, a.output, a.command.removeprefix("export-"))
    if a.command == "export-dino":
        return data.export_dino(load(a.records), load(a.validation), a.train_root, a.val_root, a.output_dir)
    if a.command == "plan-augment":
        return augment.plan(load(a.backgrounds), load(a.references), load(a.targets), a.seed, a.output)
    if a.command == "augment":
        return augment.run(load(a.tasks), a.background_root, a.reference_root, a.prompts, a.output_dir, a.execute)
    if a.command == "collect-augment":
        return augment.collect(a.output_dir, a.output)
    if a.command == "ground":
        return grounding.ground(a.manifest, a.semantic, a.image_root, a.model_config, a.checkpoint, a.output,
            a.code_root, a.box_threshold, a.text_threshold, a.device, a.execute, a.text_encoder)
    if a.command == "evaluate":
        return evaluation.evaluate(load(a.manifest), load(a.predictions), a.output, a.iou, a.allow_missing)
    if a.command == "train":
        if a.kind in {"sft", "rl"} and not all((a.model, a.dataset)):
            raise ValueError("SFT/RL require --model and --dataset")
        return training.launch(a.kind, a.recipe, a.model, a.dataset, a.output, a.execute,
            a.framework_root, a.checkpoint, a.text_encoder, a.dataset_config)


def main():
    try:
        result = run(parser().parse_args())
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(f"railmind: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
