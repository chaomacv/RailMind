import hashlib
import json
import math
from pathlib import Path

CLASSES = (
    "nest", "nonut", "rustyfence", "rustypaint", "peeling", "noclip",
    "snappedclip", "debris", "plastic", "rustypole", "agedcement",
    "rustyplate", "loosenut",
)
SCENES = (
    "steel_bridge", "rail_track", "environment", "catenary_pole",
    "noise_barrier", "power_tower",
)
SCENE_ALIASES = dict(zip(
    ("钢架桥", "轨道", "环境", "接触网杆", "声屏障", "铁塔"), SCENES,
))
DESCRIPTIONS = {
    "nest": "bird nest on a catenary pole or power tower",
    "nonut": "missing nut or bolt on a steel bridge",
    "rustyfence": "corroded bridge guardrail",
    "rustypaint": "corroded steel bridge coating",
    "peeling": "peeling steel bridge coating",
    "noclip": "missing rail fastener clip",
    "snappedclip": "fractured rail fastener clip",
    "debris": "accumulated debris near railway infrastructure",
    "plastic": "plastic film or sheeting near the railway",
    "rustypole": "corroded noise barrier column",
    "agedcement": "aged noise barrier mortar",
    "rustyplate": "corroded noise barrier panel",
    "loosenut": "loosened antenna clamp nut on a power tower",
}
SCENE_CLASSES = {
    "steel_bridge": {"nonut", "rustyfence", "rustypaint", "peeling"},
    "rail_track": {"noclip", "snappedclip"},
    "environment": {"debris", "plastic"},
    "catenary_pole": {"nest"},
    "noise_barrier": {"rustypole", "agedcement", "rustyplate"},
    "power_tower": {"nest", "loosenut"},
}


def resource_root():
    checkout = Path(__file__).resolve().parent.parent
    return checkout if (checkout / "prompts").is_dir() else Path(__file__).resolve().parent / "assets"


def load(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".jsonl":
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in value)
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def subset_manifest(rows, ids, output):
    unique(rows)
    if len(ids) != len(set(ids)) or not all(isinstance(x, str) for x in ids):
        raise ValueError("Subset IDs must be unique strings")
    known = {row["id"] for row in rows}
    if not ids or set(ids) - known:
        raise ValueError("Subset contains empty or unknown IDs")
    selected = [row for row in rows if row["id"] in set(ids)]
    save(output, selected)
    return {"images": len(selected), "output": str(output), "selected_ids": ids}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def labels(value):
    if not isinstance(value, list) or any(not isinstance(x, str) or x not in CLASSES for x in value):
        raise ValueError("Labels must be a list of canonical railway categories")
    if len(value) != len(set(value)):
        raise ValueError("Duplicate categories")
    return sorted(value)


def scene(value):
    value = SCENE_ALIASES.get(value, value)
    if value not in SCENES:
        raise ValueError(f"Unknown scene: {value}")
    return value


def checked_path(root, relative):
    root = Path(root).resolve()
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected relative path: {relative}")
    path = (root / path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or out-of-root image: {relative}")
    return path


def bbox(value, width, height):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("Expected pixel xyxy box")
    x1, y1, x2, y2 = map(float, value)
    if not all(math.isfinite(x) for x in (x1, y1, x2, y2)):
        raise ValueError("Non-finite box")
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Invalid pixel xyxy box: {value}")
    return [x1, y1, x2, y2]


def parse_answer(text):
    """只解析明确的最终 JSON，不解析分析文字中的类别。"""
    text = text.strip()
    if "<answer>" in text:
        text = text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if not lines[-1].strip() == "```":
            raise ValueError("Unclosed JSON fence")
        text = "\n".join(lines[1:-1])
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Expected JSON object")
    result["anomaly_classes"] = labels(result.get("anomaly_classes"))
    if "scene" in result:
        result["scene"] = scene(result["scene"])
    return result


def unique(rows):
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image IDs")


def train_only(rows):
    if not rows:
        raise ValueError("Empty training dataset")
    unique(rows)
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("Training exports accept train split only")
