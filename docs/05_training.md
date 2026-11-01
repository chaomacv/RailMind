# SFT and GroundingDINO training

[English](../README.md) · [简体中文](zh-CN/05_training.md)


Use separate Python/CUDA environments for Qwen, RL and the legacy detector. The CPU toolkit does not require PyTorch. Offline tests do not establish GPU compatibility; avoid installing all training frameworks into one environment.

## Qwen SFT

Export reviewed data with `export-sft`, producing `train.json` and `dataset_info.json`. Install the matching framework in a compatible GPU environment:

```bash
python -m pip install -e .
python -m pip install -e third_party/llamafactory
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
python -m railmind train sft --recipe configs/sft/custom_lora.json \
  --model models/Qwen3-VL-8B-Instruct --dataset data/sft/train.json \
  --output outputs/sft_custom
```

The command previews the resolved launch by default. Prepare models, data and GPU dependencies before adding `--execute`. The launcher sets `FORCE_TORCHRUN=1`, generates absolute-path configuration and rejects nonempty output directories. Generated `.yaml` uses JSON syntax, which is valid YAML.

Choose a recipe:

- `custom_lora.json`: reviewed custom data, without a paper-reproduction claim.
- `paper_20931.json`: paper-scale data, 3 epochs, nominal effective batch 32 and learning rate 5e-6. Historical configuration is incomplete; full fine-tuning is an explicit assumption rather than recovered evidence.
- `server_lora_24265.json`: later offline-package recipe, rank 8, 3 epochs, learning rate 1e-5, effective batch 32 and cutoff length 8192. It includes extra direct-box-output samples and must be reported separately from paper description supervision.

Batch checking uses `GPU count × per-device batch × gradient accumulation`. Copy and adjust recipes explicitly when changing GPU count. The framework must support the `qwen3_vl` template; the older server-level LLaMA-Factory has different templates. Obtain base models through official distribution channels and retain their licenses.

Save resolved configuration, data/prompt hashes, seeds, actual framework version, base model, hardware, logs and checkpoints. Merge LoRA as required before RL on a full SFT model; document adapter-based RL separately.

## Manual box review

Reinspect and annotate synthetic images. Example review format:

```json
[{"id":"sample-id", "sha256":"EXACT_IMAGE_SHA256", "approved":true, "reviewer":"reviewer",
  "verified_labels":["noclip"], "instances":[{"label":"noclip","bbox":[100,200,160,240]}]}]
```

Use **xyxy pixel coordinates** for `bbox`. Review validates bounds and consistency with verified classes. Existing manual COCO annotations can be imported and reviewed without first generating teacher text.

```bash
python -m railmind export-dino --records data/train/reviewed_boxes.jsonl \
  --validation data/val/manifest.jsonl --train-root data/train/images \
  --val-root data/val/images --output-dir data/dino
```

Exports include ODVG training JSONL, COCO validation JSON, class mappings and `datasets.json` with local absolute paths. They check train/validation hashes and augmentation lineage. Regenerate after moving machines; do not upload these data artifacts.

## Source-matched detector

The snapshot is **Open-GroundingDino**, using `models/GroundingDINO`, not the official inference-only package layout. Install dependencies and compile operators in a compatible environment:

```bash
python -m pip install -r third_party/groundingdino/requirements.txt
cd third_party/groundingdino/models/GroundingDINO/ops
python setup.py build install
python test.py
cd ../../../../..
```

Install CUDA-matched PyTorch/torchvision first; consult the upstream README's historical environment baseline. Neither this release nor the private reproduction package is a fully validated container or historical environment lock. Do not assume the server's Python 3.14 supports GPU training.

The retained `cfg_railway.py` specifies Swin-T, frozen BERT, 30 epochs and the actual optimizer/schedule configuration. Obtain Swin-T and BERT pretrained weights separately.

```bash
python -m railmind train dino --recipe configs/dino/recipe.json \
  --framework-root third_party/groundingdino \
  --checkpoint weights/groundingdino_swint_ogc.pth \
  --text-encoder weights/bert-base-uncased \
  --dataset-config data/dino/datasets.json --output outputs/dino_train2
```

Add `--execute` to train. The launcher uses the current interpreter rather than the obsolete `zcy` Conda path. The paper's “batch size 4” and the source's “batch 4 per process” are not necessarily the same global batch; record the process count.
