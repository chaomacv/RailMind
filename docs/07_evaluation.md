# Inference, evaluation and ablations

[English](../README.md) · [简体中文](zh-CN/07_evaluation.md)


## Semantic predictions

Set `RAILMIND_BASE_URL`, `RAILMIND_API_KEY` and `RAILMIND_MODEL`, and deploy a compatible model service separately. This repository supplies workflow tools, not an inference server.

```bash
python -m railmind semantic --manifest data/test/manifest.jsonl \
  --image-root data/test/images --output outputs/semantic.jsonl
```

Add `--execute` to call the service. Requests exclude benchmark labels, filenames and provenance metadata. The base URL ends in `/v1`, not `/chat/completions`. Keep HTTPS verification enabled and confirm that the endpoint returns model JSON rather than a storage-gateway response.

## Defect localization

Compile CUDA operators in the source-matched detector environment first:

```bash
python -m railmind ground --manifest data/test/manifest.jsonl \
  --semantic outputs/semantic.jsonl --image-root data/test/images \
  --checkpoint weights/dino_finetuned.pth --text-encoder weights/bert-base-uncased \
  --output outputs/detections.jsonl
```

Add `--execute` for inference. Default class prompts are period-separated. Resize the short side to 800 with a maximum long side of 1333. Convert normalized boxes to **original-image xyxy pixel coordinates**. Box/text thresholds are both 0.35; select and record thresholds on validation data, never test data.

Prompts use predicted classes, not ground truth. Empty class predictions produce empty detections. Decoded phrases must uniquely map to a requested canonical class; ambiguous phrases increment `unmapped_phrases` rather than receiving arbitrary labels. Partial-token phrases can reduce recall through mapping failure: inspect counts and validate mapping before drawing conclusions. Configuration/checkpoint key mismatches raise errors rather than silently using partially loaded models.

The adapter calls the matching `models/GroundingDINO` implementation directly, not `groundingdino/util/inference.py` from another package layout.

## Metrics

```bash
python -m railmind evaluate --manifest data/test/manifest.jsonl \
  --predictions outputs/detections.jsonl --iou 0.5 --output outputs/metrics.json
```

For classification, evaluate `semantic.jsonl` by comparing predicted and ground-truth image-level class sets. For detection, use descending-score, same-class, one-to-one greedy matching at the selected IoU; duplicate predictions are false positives. Missing predictions raise errors unless `--allow-missing` explicitly treats them as empty. Unknown IDs or classes raise errors.

Outputs separately report:

- Classification macro PR/RC/F1 across 13 classes; macro-F1 is the arithmetic mean of class F1 values.
- Detection macro PR/RC/F1 across 13 classes.
- Per-scene detection PR/RC/F1 and the mean over represented scenes; the six-scene mean requires all six.
- Detection micro PR/RC/F1 from aggregated instance counts.

Values lie in `[0,1]`; unsupported ground-truth classes receive zero. The harmonic mean of aggregated precision and recall **does not replace** arithmetic class-macro F1. Fixed-IoU precision **is not** COCO AP/mAP. For COCO AP, separately document pycocotools settings and score/ground-truth conversions.

The released matching/aggregation protocol is `railmind-release-v1`. Check historical result scripts before comparing paper figures. Paper Table I uses scene averages; other tables have inconsistent class-macro versus harmonic-aggregation wording. Do not mix these fields in a single comparison.

## Recommended experiments

1. Compare base Qwen, description SFT, SFT+GRPO and SFT+GSPO under identical model scale, prompts and image-level class protocols.
2. Compare direct VLM box output with understanding-guided localization. Specify parsing and coordinate adaptation; box output is not a semantic prediction.
3. Compare original data with same-scene augmentation using the same detector and test manifest.
4. Compare predicted-class and all-class prompts. Ground-truth prompts are an oracle upper bound, not a main result.
5. Hold model outputs fixed while scanning IoU 0.1–0.5; choose confidence thresholds only on validation data.
6. Report class/scene and long-tail results, missing-output rates, unmapped phrases, latency, GPU memory, and small-object/occlusion failures.

The 820-image version has only 11 catenary-pole images: report sample size and uncertainty. Sparse annotations may count visually plausible extra detections as false positives. Audit cases without changing test annotations after inspecting outputs.
