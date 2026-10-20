import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from .schema import load, save, resource_root


def launch(kind, recipe_path, model, dataset, output, execute=False, framework_root=None,
           checkpoint=None, text_encoder=None, dataset_config=None):
    recipe = load(recipe_path)
    world = int(os.environ.get("NPROC_PER_NODE", "4"))
    if world < 1:
        raise ValueError("NPROC_PER_NODE must be positive")
    root = resource_root()
    output = Path(output).resolve()
    if execute and output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; select a new run directory")
    if kind in {"sft", "rl"}:
        records = load(dataset)
        expected = recipe.get("expected_records")
        if not records or (expected is not None and len(records) != expected):
            raise ValueError(f"Recipe expects {expected} records; got {len(records)}. Select a matching profile, do not silently change counts.")
        for record in records:
            if not isinstance(record.get("messages"), list) or not record.get("images"):
                raise ValueError("Training records need messages and images")
            for image in record["images"]:
                if not Path(image).is_absolute() or not Path(image).is_file():
                    raise ValueError("Training export images must exist as resolved absolute paths")
    if kind == "sft":
        registry = Path(dataset).resolve().parent / "dataset_info.json"
        if not registry.is_file() or load(registry).get("railmind_sft", {}).get("file_name") != Path(dataset).name:
            raise ValueError("Use export-sft to create a matching dataset registry")
        config = dict(recipe["config"])
        config.update(model_name_or_path=str(Path(model).resolve()), dataset="railmind_sft",
                      dataset_dir=str(Path(dataset).resolve().parent), output_dir=str(output))
        if "deepspeed" in config and config["deepspeed"].startswith("configs/"):
            config["deepspeed"] = str(root / config["deepspeed"])
        if world * config["per_device_train_batch_size"] * config["gradient_accumulation_steps"] != recipe["effective_batch_size"]:
            raise ValueError("GPU count changes effective batch size; adjust a copied recipe explicitly")
        generated = output / "resolved_sft.yaml"
        command = ["llamafactory-cli", "train", str(generated)]
    elif kind == "rl":
        if any("solution" not in x for x in records):
            raise ValueError("RL dataset needs solution fields")
        from .schema import labels
        for row in records:
            labels(json.loads(row["solution"])["anomaly_classes"])
        config = recipe["config"]
        if world * config["per_device_train_batch_size"] * config["gradient_accumulation_steps"] != recipe["effective_batch_size"]:
            raise ValueError("GPU count changes nominal effective batch size")
        command = ["swift", "rlhf", "--rlhf_type", "grpo", "--model", str(Path(model).resolve()),
                   "--dataset", str(Path(dataset).resolve()), "--output_dir", str(output),
                   "--external_plugins", str(root / "training/railway_reward.py"), "--reward_funcs", "railway_category"]
        for key, value in config.items():
            command.extend(["--" + key, str(value).lower() if isinstance(value, bool) else str(value)])
    else:
        if not all((framework_root, checkpoint, text_encoder, dataset_config)):
            raise ValueError("DINO requires --framework-root, --checkpoint, --text-encoder, --dataset-config")
        command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(world),
                   "main.py", "-c", str(root / "configs/dino/cfg_railway.py"),
                   "--datasets", str(Path(dataset_config).resolve()), "--output_dir", str(output),
                   "--pretrain_model_path", str(Path(checkpoint).resolve()),
                   "--options", "text_encoder_type=" + str(Path(text_encoder).resolve()), "--save_log"]
    result = {"dry_run": not execute, "kind": kind, "command": shlex.join(command), "recipe": recipe,
              "note": "SFT resolved YAML is generated on execution" if kind == "sft" else "GPU run requires its dedicated framework environment"}
    if not execute:
        return result
    if kind in {"sft", "rl"} and not Path(model).is_dir():
        raise ValueError("Provide a local base or merged SFT model directory")
    if kind == "dino":
        for path in (Path(framework_root)/"main.py", Path(checkpoint), Path(text_encoder), Path(dataset_config)):
            if not path.exists():
                raise ValueError(f"Missing DINO input: {path}")
        sets = load(dataset_config)
        if not sets.get("train") or not sets.get("val"):
            raise ValueError("DINO dataset config requires train and val")
        count = 0
        instances = 0
        for entry in sets["train"]:
            if entry.get("dataset_mode") != "odvg":
                raise ValueError("RailMind detector recipe expects ODVG training data")
            for row in load(entry["anno"]):
                count += 1
                instances += len(row.get("detection", {}).get("instances", []))
        for key, actual in [("expected_training_images", count), ("expected_instance_annotations", instances)]:
            if recipe.get(key) is not None and recipe[key] != actual:
                raise ValueError(f"DINO recipe {key} expects {recipe[key]}, got {actual}")
    if not shutil.which(command[0]):
        raise ValueError(f"Missing framework executable: {command[0]}")
    output.mkdir(parents=True, exist_ok=True)
    if kind == "sft":
        save(generated, config)
    save(output / "launch.json", result)
    env = os.environ.copy()
    env["NPROC_PER_NODE"] = str(world)
    if kind == "sft":
        env["FORCE_TORCHRUN"] = "1"
    subprocess.run(command, check=True, env=env, cwd=framework_root if kind == "dino" else None)
    return {"finished": True, "output": str(output)}
