#!/usr/bin/env python3
"""Check release files, documentation links, source hashes and common credential patterns."""
import ast
import hashlib
import json
import re
from pathlib import Path

# Public documentation illustrations approved for this source release.
# Match both the exact repository path and SHA256; do not exempt entire folders.
PUBLIC_DOC_ASSETS = {
    "docs/assets/railmind-teaser.jpg": "bcf2deabbe9ace6deaf241eed2f1f517655dd6852975be60297fc2a270b51a10",
    "docs/assets/scenes/catenary-pole.png": "5b3b6905c9cb7a165b47512c235d6847a0421aa7a4f74088433b3bdd13783901",
    "docs/assets/scenes/environment.png": "cdcbe202ba58e48916943e96eb8564a7cd16bfdf4fad176febfbf537726a3fc4",
    "docs/assets/scenes/noise-barrier.png": "6caa17dfe5e34809d17fcd44ca1bbd18f8d01dbb6f81fc8d26920ca962f6a666",
    "docs/assets/scenes/power-tower.png": "5d7da8652392cc9c056bfbcc97eb126bba4b2a80bd61752f86caff5cab6fb0b2",
    "docs/assets/scenes/rail-track.png": "8f636012dac1cad8b47a597a54fe37721ed8ef80f5540c08a578829b36e79972",
    "docs/assets/scenes/steel-bridge.png": "27cd19220cee921cdef5dd894c8ed9bbb1a97e8505d169651cb2d0c26b99a7ea"
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
