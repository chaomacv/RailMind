import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .schema import checked_path, digest, load, parse_answer, save
from .data import verify_manifest


def image_uri(path):
    mime = mimetypes.guess_type(str(path))[0]
    if mime not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise ValueError("Use a decoded JPEG, PNG, WebP or GIF image")
    content = Path(path).read_bytes()
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("Image exceeds 20 MiB; resize explicitly and record provenance first")
    return f"data:{mime};base64," + base64.b64encode(content).decode("ascii")


def request_json(url, payload, key, timeout=120):
    if not key:
        raise ValueError("Missing API key environment variable")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("Expected an HTTP(S) service URL without embedded credentials")
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    # Retry only requests rejected before processing; never retry unknown generation outcomes.
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if "json" not in response.headers.get("Content-Type", "").lower():
                    raise ValueError("Gateway returned non-JSON content; check model routing")
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"API HTTP {error.code}; response body omitted to protect credentials") from None
    raise RuntimeError("API retry limit reached")


def annotate(manifest, image_root, prompt_path, output, teacher=True, execute=False):
    rows = load(manifest)
    verify_manifest(rows, image_root)
    prefix = "RAILMIND_TEACHER" if teacher else "RAILMIND"
    url = os.environ.get(prefix + "_BASE_URL", "").rstrip("/")
    model = os.environ.get(prefix + "_MODEL", "")
    if not execute:
        return {"dry_run": True, "images": len(rows), "model": model or "unset", "role": prefix,
                "output": str(output), "note": "Pass --execute to make API calls"}
    if not url or not model:
        raise ValueError(f"Set {prefix}_BASE_URL and {prefix}_MODEL")
    prompt = Path(prompt_path).read_text(encoding="utf-8")
    done = load(output) if Path(output).exists() else []
    from .schema import unique
    unique(done)
    by_id = {r["id"]: r for r in done}
    if set(by_id) - {r["id"] for r in rows}:
        raise ValueError("Existing output belongs to another manifest")
    prompt_sha = digest(prompt_path)
    for row in rows:
        previous = by_id.get(row["id"])
        if previous:
            if (previous["sha256"] != row["sha256"] or previous.get("model") != model
                    or previous.get("prompt_sha256") != prompt_sha or previous.get("endpoint") != url):
                raise ValueError("Resume provenance mismatch; choose a new output file")
            continue
        # Ground truth, desired generation labels and scene metadata are never sent to the model.
        content = [{"type": "text", "text": prompt},
                   {"type": "image_url", "image_url": {"url": image_uri(checked_path(image_root, row["image"]))}}]
        body = request_json(url + "/chat/completions", {"model": model,
            "messages": [{"role": "user", "content": content}], "temperature": 0.0,
            "max_tokens": 2048, "stream": False}, os.environ.get(prefix + "_API_KEY", ""))
        text = body["choices"][0]["message"]["content"]
        answer = parse_answer(text)
        result = dict(row)
        result.update({"model": model, "endpoint": url, "prompt_sha256": prompt_sha,
                       "raw_response": text})
        result["teacher" if teacher else "prediction"] = answer
        # A new answer always needs its own review, even if its image was reviewed before.
        if teacher:
            result.pop("review", None)
        done.append(result)
        save(output, done)
    return {"completed": len(done), "output": str(output)}
