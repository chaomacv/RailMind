# SFT And GroundingDINO Training

Use separate Python/CUDA environments for Qwen, RL and the legacy detector implementation. The CPU package works without PyTorch. GPU dependency compatibility is not established by the offline tests, and installing every training framework into one environment is not a supported shortcut.

## Qwen SFT

Prepare reviewed data with `export-sft`; it creates `train.json` and the matching `dataset_info.json`. Install the source-matched framework in a compatible GPU environment:

```bash
python -m pip install -e .
python -m pip install -e third_party/llamafactory
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
python -m railmind train sft --recipe configs/sft/custom_lora.json \
  --model models/Qwen3-VL-8B-Instruct --dataset data/sft/train.json \
  --output outputs/sft_custom
```

This prints the resolved command without training. Add `--execute` once the model, data and GPU stack are ready. The launcher uses `FORCE_TORCHRUN=1`, generates an absolute-path config, and rejects nonempty run directories. The generated `.yaml` is JSON syntax, which is valid YAML and accepted by the framework's YAML parser.

Recipe choices:

- `custom_lora.json`: your own reviewed dataset; does not assert manuscript results.
- `paper_20931.json`: manuscript counts, 3 epochs, nominal effective batch 32, LR 5e-6. Exact historical settings are incomplete; full tuning is an explicitly documented assumption, not proof of the original method.
- `server_lora_24265.json`: offline bundle profile, rank 8, 3 epochs, LR 1e-5, effective batch 32, cutoff 8192. This includes additional direct box-output samples and is not the manuscript descriptive-only profile.

The batch check is `GPU count * per-device batch * gradient accumulation`. Copy and adjust a recipe explicitly when changing GPU count. The framework template must support `qwen3_vl`; the older top-level server LLaMA-Factory tree is not the selected snapshot. Obtain the base model from its official model distribution and retain its license.

Save the resolved config, dataset/prompt hashes, actual package versions, random seed, hardware, logs and output checkpoint. Merge LoRA adapters using the selected framework before providing a full SFT model to the RL template, unless a separately documented adapter training recipe is used.

## Human Box Annotations

Synthetic images need new visual inspection and new boxes. Review decisions can include:

```json
[{"id":"sample-id", "sha256":"EXACT_IMAGE_SHA256", "approved":true, "reviewer":"reviewer",
  "verified_labels":["noclip"], "instances":[{"label":"noclip","bbox":[100,200,160,240]}]}]
```

`bbox` is **xyxy pixels**. `review` checks bounds and agreement with verified labels. For existing human COCO annotations, apply review to the imported manifest; teacher annotations are not required for detector training.

```bash
python -m railmind export-dino --records data/train/reviewed_boxes.jsonl \
  --validation data/val/manifest.jsonl --train-root data/train/images \
  --val-root data/val/images --output-dir data/dino
```

This writes ODVG training JSONL, COCO validation JSON, a label map and a local absolute-path `datasets.json`. It checks training/validation hashes and augmentation lineage. Regenerate the config on a new machine.

## Source-Matched Detector

The vendored detector is **Open-GroundingDino**, using `models/GroundingDINO`, not the differently structured official inference-only package. In a compatible environment, install its requirements and build its ops:

```bash
python -m pip install -r third_party/groundingdino/requirements.txt
cd third_party/groundingdino/models/GroundingDINO/ops
python setup.py build install
python test.py
cd ../../../../..
```

Install PyTorch/torchvision for your CUDA stack before building; check the upstream snapshot README for its legacy baseline. Neither this release nor the prior bundle captures a working container image or a complete historic environment lock. Do not use the server's Python 3.14 base as an assumed compatible GPU training environment.

The copied `cfg_railway.py` preserves Swin-T, BERT freezing, 30 epochs and the actual optimizer/scheduler settings. Download or supply trusted pretrained Swin-T and BERT weights separately.

```bash
python -m railmind train dino --recipe configs/dino/recipe.json \
  --framework-root third_party/groundingdino \
  --checkpoint weights/groundingdino_swint_ogc.pth \
  --text-encoder weights/bert-base-uncased \
  --dataset-config data/dino/datasets.json --output outputs/dino_train2
```

Add `--execute` for training. The launcher avoids obsolete `zcy` Conda paths and uses the active interpreter. The manuscript's “batch size 4” and source config's per-process batch 4 should not be silently interpreted as the same global batch. Preserve the resolved world size and report it.
