# Teacher-model annotation

[English](../README.md) · [简体中文](zh-CN/03_data_engine.md)


Teacher models convert railway images into scene, class, evidence and risk supervision. They do not replace manual detector-box annotations.

## Configure a compatible teacher service

```bash
cp .env.example .env
# Fill in .env locally, then load the environment variables
source .env
python -m railmind annotate --manifest data/train/manifest.jsonl \
  --image-root data/train/images --output outputs/teacher.jsonl
```

Commands preview and validate requests by default. Set `RAILMIND_TEACHER_BASE_URL`, `RAILMIND_TEACHER_API_KEY` and `RAILMIND_TEACHER_MODEL`; add `--execute` for real calls. The base URL ends in `/v1`; the adapter appends `/chat/completions`. Gemini is accessed through an **API-compatible gateway**, not Google's native protocol. Verify provider model names and multimodal support before calling.

Requests contain only image bytes and prompts, excluding ground-truth classes, filenames, generation targets and scene metadata. `teacher.txt` supplies class definitions and component context and requests cautious descriptions of visible evidence. Generated coordinates are not ground truth.

Responses must contain a JSON object with standard `anomaly_classes`. Invalid answers raise errors rather than silently becoming negatives. Save progress after each image; resume checks image, model, endpoint and prompt hashes. Use a new output file when changing experiment settings. HTTP 429 receives bounded retries; uncertain generations are not blindly resubmitted.

## Review before training

Use the manifest SHA256 in review decisions. Format example:

```json
[{"id":"example-01", "sha256":"EXACT_IMAGE_SHA256", "approved":true,
  "verified_labels":["noclip"], "reviewer":"reviewer-name"}]
```

```bash
python -m railmind review --records outputs/teacher.jsonl \
  --decisions data/reviews.json --image-root data/train/images \
  --output outputs/reviewed_teacher.jsonl
python -m railmind export-sft --records outputs/reviewed_teacher.jsonl \
  --image-root data/train/images --output data/sft/train.json
```

Only approved records are exported. Class corrections require a `corrected_teacher` object with corrected final classes and consistent evidence; changing only labels leaves misleading descriptions. Review hallucinations, risk statements, class coverage and negatives.

Historical JSON in `fullAnalysis.raw_output` is not automatically reviewed: convert and review it. Private paths, raw responses and supervision files are excluded from the source release.
