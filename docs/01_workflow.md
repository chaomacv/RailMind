# Workflow and module boundaries

[English](../README.md) · [简体中文](zh-CN/01_workflow.md)


The research question is whether domain-adapted semantic understanding combined with a specialized localizer is more reliable than asking a multimodal model to output coordinates directly.

1. Split real images by acquisition batch and provenance before teacher annotation or augmentation. Store manual classes, boxes and model responses separately.
2. Generate structured supervision from visible component conditions, scenes, class definitions and cautious risk descriptions; review it before training.
3. Select same-scene backgrounds and defect references from training data. Edit only the target defect, preserve unrelated geometry, verify actual labels and do not inherit pre-edit boxes.
4. Train Qwen on reviewed descriptions and the detector on verified boxes. Optimize them separately without a unified end-to-end loss.
5. Optionally apply class-set-reward RL to the SFT model using synthetic images with reviewed classes. Generation intent is not ground truth.
6. During inference, Qwen returns explicit final classes; the detector uses predicted classes as text prompts without reading evaluation labels.
7. Evaluate classification independently. Match same-class boxes by IoU, and report scene averages separately from class-macro metrics.

## Paper-to-code mapping

| Paper module | Released implementation | Provenance |
|---|---|---|
| Teacher data engine | `railmind/api.py`, `prompts/teacher.txt`, `review` | Annotation scripts, domain prompts and portable adapters |
| NanoBanana defect transfer | `railmind/augment.py`, `prompts/augmentation.json` | Same-scene workflow and 13-class prompt mapping |
| Qwen SFT | `export-sft`, SFT recipes | Offline LLaMA-Factory source and training YAML |
| Detector training | `export-dino`, matching configuration and snapshot | Open-GroundingDino train2 configuration |
| GRPO/GSPO | `railmind/reward.py`, reward plugin and RL recipes | Paper reward and server ms-swift examples; original railway launch missing |
| Understanding-guided localization | `semantic`, `ground` | Qwen-driven detection/evaluation scripts |
| Classification and detection metrics | `railmind/evaluation.py` | Paper protocol with explicit aggregation and matching rules |

The new teacher prompt requests concise visible evidence and structured output. It does not claim to reproduce historical prompts exactly or recover hidden reasoning. Prompt hashes track subsequent experiments.

Unrelated banking/RAG examples, old website user storage, duplicate frameworks, weights, incomplete archives and bulk experiment outputs are excluded. Training/evaluation datasets and sample-ID lists are not released.
