# NanoBanana same-scene augmentation

[English](../README.md) · [简体中文](zh-CN/04_augmentation.md)


The server implementation submits to `/v1/draw/nano-banana`, polls `/v1/draw/result` and downloads images through a GRSAI-style asynchronous gateway. This is a **provider-specific adapter**, not a general Google SDK. Verify current endpoints and model availability before paid calls.

## Prepare background and reference pools

Use training-only sources for both pools. Build manifests and apply visual decisions with `review`. Planning requires approval and matching image hashes; normal backgrounds also need an explicit empty verified-label list.

Backgrounds should lack the target defect, references should clearly show it, and scenes must agree. A broken clip belongs on a rail fastener, not a noise-barrier panel. Multi-defect references must cover the requested combination.

Set generation quotas explicitly. One possible training-only weighting is `1 / max(class_count, 1)`, followed by reviewed integer quotas. The planner consumes target counts without hiding sampling decisions.

```bash
python -m railmind plan-augment --backgrounds data/backgrounds/reviewed.jsonl \
  --references data/references/reviewed.jsonl \
  --targets configs/augmentation_targets.example.json --seed 42 \
  --output outputs/augmentation_tasks.jsonl
python -m railmind augment --tasks outputs/augmentation_tasks.jsonl \
  --background-root data/backgrounds/images --reference-root data/references/images \
  --output-dir outputs/generated
```

Planning rejects non-training sources, incompatible scenes, identical background/reference images and references missing requested classes. Seeds reproduce selection; they do not validate annotation quality.

## Generation and provenance

Set `NANO_BANANA_BASE_URL`, `NANO_BANANA_API_KEY` and `NANO_BANANA_MODEL`; install image support with `pip install -e '.[images]'`. Add `--execute` to the generation command for actual calls.

The 13-class prompts require localized, physically plausible edits, preserved perspective/lighting/geometry, and removal of reference annotation boxes and text. Multi-class tasks combine constraints, but conflicts and compliance still require review.

Each task stores source hashes, prompt, model, provider and accepted task ID. Resume polling from the saved ID and verify downloaded images. A crash after provider acceptance but before ID persistence leaves an uncertain task; check the provider console before resubmitting.

```bash
python -m railmind collect-augment --output-dir outputs/generated \
  --output data/synthetic/manifest.jsonl
```

Images remain in `outputs/generated`, used as the subsequent `--image-root`. Manifests retain relative paths, actual dimensions and provenance. `target_labels` are not automatically promoted to `gt_labels`.

## Visual quality review

Check actual defects, scale, geometry, realism, unintended defects, transfer artifacts and residual annotations. Record verified classes using `review`. SFT additionally requires reviewed teacher evidence; RL requires reviewed synthetic images and verified classes; detector training requires reviewed **actual pixel boxes** after editing.

Record reviewer, image hash and approval/rejection. Keep failures and reasons locally; rejected samples do not enter training. Check lineage against all validation/test manifests before training. Generated images and supervision are training data and are not uploaded.
