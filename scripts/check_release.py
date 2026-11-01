#!/usr/bin/env python3
"""检查发布文件、文档链接、源码哈希及常见凭据模式。"""
import ast
import hashlib
import json
import re
from pathlib import Path

# Public documentation illustrations approved for this source release.
# Match both the exact repository path and SHA256; do not exempt entire folders.
PUBLIC_DOC_ASSETS = {
    "docs/assets/railmind-teaser.jpg": "bcf2deabbe9ace6deaf241eed2f1f517655dd6852975be60297fc2a270b51a10",
    "docs/assets/scenes/catenary-pole.png": "697c8c536986bf9200f77d57e62bd8db0f5743130aa8262b76880da01ffa79aa",
    "docs/assets/scenes/environment.png": "ae022e55a8d2e9f8025e83a52f350183a069ef9bb2cd4c89d45d5883027659be",
    "docs/assets/scenes/noise-barrier.png": "61b63348318f2615d728077539c9d0553ae474e76711f6e6401bf36c49101f40",
    "docs/assets/scenes/power-tower.png": "3e3da02ad9d6671703ca797c54a6297f56aebcd990816b49307c0acac3fd19d3",
    "docs/assets/scenes/rail-track.png": "c07ebdf13ccfc9077bd321b9aa0b9f7cd0adb1ab49ec8a0a7a31322f8bbf8c63",
    "docs/assets/scenes/steel-bridge.png": "13bbe2c37393422485d3c23ab0c56d66dc2000121ba9e7360f724cb844266546"
}

root = Path(__file__).resolve().parents[1]
errors = []
count = 0
for path in root.rglob("*"):
    rel = path.relative_to(root)
    if rel.parts[0] in {"local", "build", "dist", "outputs", "data", "models", "weights", "logs", "runs", "wandb"}:
        continue
    if any(p in {".git", "__pycache__"} or p.startswith(".venv") or p.endswith(".egg-info") for p in rel.parts):
        continue
    if path.is_symlink():
        errors.append(f"Symlink: {rel}")
    if not path.is_file():
        continue
    count += 1
    public_asset = rel.as_posix() in PUBLIC_DOC_ASSETS
    if public_asset and hashlib.sha256(path.read_bytes()).hexdigest() != PUBLIC_DOC_ASSETS[rel.as_posix()]:
        errors.append(f"Public documentation asset hash mismatch: {rel}")
    if (path.suffix.lower() in {".jsonl", ".csv", ".tsv", ".jpg", ".jpeg", ".png", ".webp"} and not public_asset) or "benchmark_810_ids" in path.name:
        errors.append(f"Private data or sample list in release: {rel}")
    if path.stat().st_size > 5_000_000 or path.suffix in {".pth", ".pt", ".bin", ".safetensors", ".enc", ".pdf"}:
        errors.append(f"Non-source payload: {rel}")
    if path.suffix == ".py" and "third_party" not in rel.parts:
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            errors.append(f"Python syntax: {rel}")
    if path.suffix == ".json":
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            errors.append(f"JSON syntax: {rel}")
    if path.suffix not in {".py", ".md", ".json", ".toml", ".txt", ".yml", ".yaml", ".example"}:
        continue
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = [r"sk-[A-Za-z0-9_-]{20,}", r"AIza[A-Za-z0-9_-]{30,}", r"gh[pousr]_[A-Za-z0-9]{30,}", r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
                r"--token\s+eyJ[A-Za-z0-9_-]{30,}"]
    if any(re.search(pattern, text) for pattern in patterns):
        errors.append(f"Potential credential: {rel}")
    if "third_party" not in rel.parts and path.suffix == ".md":
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            if target.startswith(("http:", "https:", "#", "mailto:")):
                continue
            if not (path.parent / target.split('#')[0]).exists():
                errors.append(f"Broken local doc link in {rel}: {target}")
lock = root / "third_party/source_lock.json"
if not lock.is_file():
    errors.append("Source snapshots have not been assembled")
else:
    for name, record in json.loads(lock.read_text()).items():
        for relative, expected in record.get("files", {}).items():
            path = root / "third_party" / name / relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append(f"Source snapshot hash mismatch: {name}/{relative}")
result = {"status": "FAIL" if errors else "PASS", "source_files_checked": count, "errors": errors}
print(json.dumps(result, indent=2))
raise SystemExit(bool(errors))
