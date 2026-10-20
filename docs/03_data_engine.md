# Large-Model Teacher Annotation

The teacher converts existing railway images into scene/category/evidence/risk supervision. It does not replace manual detector box annotation.

## Configure A Compatible Teacher

```bash
cp .env.example .env
# Edit .env locally, then:
source .env
python -m railmind annotate --manifest data/train/manifest.jsonl \
  --image-root data/train/images --output outputs/teacher.jsonl
```

The command above is a dry-run. Set `RAILMIND_TEACHER_BASE_URL`, `RAILMIND_TEACHER_API_KEY`, `RAILMIND_TEACHER_MODEL`, then add `--execute` to call the service. The base URL ends at `/v1`; the adapter appends `/chat/completions`. Gemini is accessed through an **OpenAI-compatible gateway**, not the native Google API protocol. Confirm the selected provider's model name and multimodal support.

The request contains only image bytes and the prompt. Ground-truth categories, filenames, generation targets and scene metadata are not sent. `teacher.txt` supplies definitions and component-function context, with cautious visible-evidence summaries. No model-generated coordinates are treated as ground truth.

Responses must contain a JSON object with canonical `anomaly_classes`; invalid responses fail rather than silently becoming negatives. Successful records are checkpointed after each image. Resume checks image, model, endpoint and prompt hashes. Use a new output file when changing the experiment. HTTP 429 is retried with a short bound; unknown generation outcomes are not blindly resubmitted.

## Review Before Training

Create a review file, using SHA256 values from the manifest:

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

Only approved records are exported. If labels change, add a `corrected_teacher` object containing the corrected final classes and consistent evidence; changing category labels alone would preserve misleading teacher descriptions. Human reviewers should also check hallucinations, risk claims, category coverage and negative examples.

Historical JSON stored under `fullAnalysis.raw_output` is not automatically treated as approved. Convert it to this schema and review it. Never publish private image paths or full original responses without checking release rights.
