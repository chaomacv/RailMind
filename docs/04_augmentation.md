# NanoBanana Scene-Aligned Data Augmentation

The server implementation uses an asynchronous GRSAI-style NanoBanana gateway: submit `/v1/draw/nano-banana`, poll `/v1/draw/result`, then download the generated image. This is a **provider-specific adapter**, not a universal Google SDK. Endpoint and model availability must be checked with the selected provider before making paid calls.

## Prepare Background And Reference Pools

Build train-only manifests for both pools, then apply visual-review decisions with `review`. Planning requires approved, image-hash-matched reviews, including explicit empty verified labels for normal backgrounds. Backgrounds should lack the intended defect; references should show it clearly. Keep the scene aligned. For example, synthesize a fractured clip on an existing rail clip, not on a noise-barrier panel. Multi-defect requests require a reference whose verified categories include the requested combination.

Set generation quotas before the run. Use the real training category histogram to prioritize rare categories, for example weights proportional to `1 / max(class_count, 1)`, then allocate and review integer quotas. The planner uses explicit target counts rather than hiding this decision in an implicit sampler.

```bash
python -m railmind plan-augment --backgrounds data/backgrounds/reviewed.jsonl \
  --references data/references/reviewed.jsonl \
  --targets configs/augmentation_targets.example.json --seed 42 \
  --output outputs/augmentation_tasks.jsonl
python -m railmind augment --tasks outputs/augmentation_tasks.jsonl \
  --background-root data/backgrounds/images --reference-root data/references/images \
  --output-dir outputs/generated
```

The planner rejects held-out splits, incompatible scenes, identical background/reference images and absent reference categories. Seeded task choices are reproducible. Background/reference annotation quality must be verified by the dataset curator.

## Generate And Preserve Provenance

Set `NANO_BANANA_BASE_URL`, `NANO_BANANA_API_KEY`, `NANO_BANANA_MODEL`. Install image support with `pip install -e '.[images]'`. Add `--execute` to the second command.

The 13 source-derived prompts require physically plausible, limited defect changes; preserved perspective, lighting and component geometry; and removal of reference annotation boxes/text. Multi-class prompts combine selected constraints. Inspect results for conflicts; they are not guaranteed to obey every instruction.

Each task stores the source hashes, prompt, model, provider and accepted provider task ID. Polling resumes from the saved ID after interruption; downloaded images are checked on resume. A crash after provider acceptance but before saving the ID is still ambiguous: check the provider dashboard before rerunning that task.

```bash
python -m railmind collect-augment --output-dir outputs/generated \
  --output data/synthetic/manifest.jsonl
```

The generated images remain under `outputs/generated`, which is the `--image-root` for subsequent annotation and review. The manifest stores relative names, actual image dimensions and source lineage. It contains `target_labels`, not assumed `gt_labels`.

## Visual Quality Gate

Inspect every retained image for actual defect presence, scale, geometry, realism, unwanted additional defects, reference transfer artifacts and burned-in annotations. Write verified category sets using the `review` schema. SFT additionally requires reviewed teacher evidence. RL requires a reviewed synthetic image with verified labels. Detector training requires reviewed **actual** pixel boxes; do not reuse original boxes after a generative edit.

Record reviewer, image hash and acceptance/rejection. Retain failed samples and reasons locally for audit, but keep rejected samples out of training. Run lineage leakage checks against every validation/test manifest before training.
