#!/usr/bin/env python3
"""检查发布文件、文档链接、源码哈希及常见凭据模式。"""
import ast
import hashlib
import json
import re
from pathlib import Path

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
    if path.suffix.lower() in {".jsonl", ".csv", ".tsv", ".jpg", ".jpeg", ".png", ".webp"} or "benchmark_810_ids" in path.name:
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
