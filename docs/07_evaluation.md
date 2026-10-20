# Inference, Evaluation And Ablations

## Semantic Prediction

Configure `RAILMIND_BASE_URL`, `RAILMIND_API_KEY`, `RAILMIND_MODEL`. Start a compatible model service separately; this repository is not an inference server.

```bash
python -m railmind semantic --manifest data/test/manifest.jsonl \
  --image-root data/test/images --output outputs/semantic.jsonl
```

Add `--execute` to call the service. No benchmark labels, filenames or source metadata are sent in the request. The base URL ends at `/v1`, not `/chat/completions`. Keep HTTPS verification enabled and verify the endpoint returns model JSON, not a storage gateway response.

## Grounding

Run in the source-matched detector environment after building the CUDA operator:

```bash
python -m railmind ground --manifest data/test/manifest.jsonl \
  --semantic outputs/semantic.jsonl --image-root data/test/images \
  --checkpoint weights/dino_finetuned.pth --text-encoder weights/bert-base-uncased \
  --output outputs/detections.jsonl
```

Add `--execute` to infer. The source default uses caption categories separated by periods, image resize to 800 with max 1333, normalized model boxes converted to original **xyxy pixels**, and fixed box/text thresholds of 0.35. Record threshold selection on validation data, not the test set.

Predicted categories, not GT categories, supply the caption. Empty predicted categories produce empty detections. Decoded detector phrases must map to exactly one requested canonical category; ambiguous phrases are counted in `unmapped_phrases`, not assigned arbitrarily. This phrase mapping can lower recall for partial token responses; inspect that counter and validate mapping before making research claims. Model config/checkpoint key mismatches fail rather than silently using a partially loaded detector.

The new adapter directly calls the source-matched `models/GroundingDINO` implementation. It does not use the stray `groundingdino/util/inference.py` helper that imports a different official package layout.

## Metrics

```bash
python -m railmind evaluate --manifest data/test/manifest.jsonl \
  --predictions outputs/detections.jsonl --iou 0.5 --output outputs/metrics.json
```

Classification can be evaluated from `semantic.jsonl` alone. Category predictions are compared with image-level GT sets. Detection uses same-category, score-descending, one-to-one greedy box matching at the chosen IoU. Duplicate predictions become false positives. Missing predictions fail by default; `--allow-missing` explicitly treats them as empty. Unknown image IDs or categories fail.

Outputs distinguish:

- Classification macro PR/RC/F1 over 13 categories; macro-F1 is the arithmetic mean of category F1 values.
- Detection macro PR/RC/F1 over 13 categories.
- Detection per-scene PR/RC/F1, mean of present scenes, and six-scene mean only when all six are present.
- Detection micro PR/RC/F1 from pooled instance counts.

All values are fractions in `[0,1]`. Zero-support categories score zero. The harmonic mean of aggregate precision/recall is **not** substituted for the arithmetic macro-F1. Fixed-IoU precision is **not** COCO AP/mAP. If COCO AP is needed, use a separately documented pycocotools evaluation and retain the prediction score/GT conversion protocol.

The release matching/pooling protocol is named `railmind-release-v1`; confirm alignment with the exact historical result-generation script before comparing published numbers. Table I uses scene means, while other manuscript tables describe category macro or harmonic aggregation inconsistently. Do not mix these output fields in one table.

## Experiments To Report

1. Same-scale Qwen baseline versus descriptive SFT versus SFT+GRPO versus SFT+GSPO, under identical prompts and image-level category evaluation.
2. Direct MLLM box output versus reasoning-guided grounding. Direct-box ablations require their own explicit parsing/coordinate adapter; they are not silently treated as semantic predictions.
3. Original training data versus scene-aligned augmentation, with the same detector and test manifest.
4. Predicted-category captions versus all-class captions. GT-caption evaluation is an oracle upper bound, never the main result.
5. IoU sweep 0.1 to 0.5 with fixed model outputs; threshold sweeps selected on validation data only.
6. Per-category/scene and tail-class results, missing-output rate, unmapped detector phrases, latency, memory and failures on small/occluded defects.

The catenary-pole scene is small (11 images in the 820 version); report its size and uncertainty. Sparse annotations can count visually valid extra detections as false positives. Audit cases rather than changing annotations after seeing model results.
