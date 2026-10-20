# SFT/RL Skill Map And Community Notes

This note is a practical learning map for the post-training parts of RailMind. It is written for readers who understand Python/PyTorch and want to adapt the project to their own domain dataset without treating SFT and RL as black-box commands.

RailMind uses SFT for domain knowledge transfer and output-format control, then optionally uses category-reward RL to improve final anomaly-category decisions on reviewed synthetic images. The grounding model is trained separately because localization quality depends on instance boxes, not only on language supervision.

## Why This Belongs In RailMind

The project is not only a railway benchmark. It is also a compact example of a modern VLM post-training workflow:

- Build domain supervision from existing images and expert prompts.
- Use a stronger teacher model to create first-pass captions, risk descriptions and category evidence.
- Keep a human review step between generated supervision and training.
- Use SFT to teach the model the domain language and answer schema.
- Use RL only where the reward can be defined cleanly and checked automatically.
- Evaluate classification, grounding and failure modes separately.

This separation is intentional. SFT is good at imitation and domain adaptation. RL is good at optimizing a narrow, measurable behavior after the model already understands the task. It is usually a mistake to ask RL to fix missing data quality, weak prompts or unclear labels.

## Skill Matrix

| Area | What To Understand | What RailMind Provides |
|---|---|---|
| Data engineering | JSONL records, image hashes, split hygiene, leakage checks | `manifest`, `verify`, `leakage`, review schemas |
| Teacher annotation | Prompt design, structured JSON, evidence fields, manual review | `annotate`, `review`, `prompts/teacher.txt` |
| SFT | Chat template, multimodal messages, LoRA, effective batch, cutoff length | `export-sft`, `configs/sft/*.json` |
| RL | Reward function, final-answer parsing, rollout count, malformed outputs | `export-rl`, `training/railway_reward.py`, `configs/rl/*.json` |
| Detection | ODVG/COCO export, class mapping, validation protocol | `export-dino`, `configs/dino/*` |
| Evaluation | Macro-F1, scene averages, IoU matching, duplicate detections | `evaluate`, `docs/07_evaluation.md` |
| Engineering | Dry-run launchers, environment isolation, source locks, tests | `train`, `scripts/check_release.py`, `tests/` |

## SFT: What To Learn First

SFT should answer three questions before any GPU run starts:

1. What exact behavior should the model imitate?
2. Which fields are trusted labels, and which fields are generated hints?
3. How will we detect that the model learned the wrong shortcut?

For RailMind, SFT teaches the model to inspect railway images, name visible anomaly categories and explain evidence in a stable schema. The 20,931-sample manuscript profile focuses on descriptive supervision. The later 24,265-sample server profile also includes 3,334 direct box-output samples, so it must be reported as a different experiment.

Important SFT skills:

- Know the chat template used by the base model. A Qwen3-VL dataset is not automatically compatible with an older Qwen2-VL template.
- Keep image paths portable and keep image hashes in the manifest.
- Separate train, validation and test images before calling a teacher model.
- Review synthetic or teacher-generated labels before they become training targets.
- Track effective batch size as `num_gpus * per_device_batch * gradient_accumulation`.
- Save the resolved training config, dataset hash, seed, framework version and base model identity.
- Treat LoRA rank, learning rate, cutoff length and image-pixel budget as experimental variables, not harmless defaults.

Common SFT failure modes:

- The model learns folder names or file naming conventions instead of visual evidence.
- Teacher text contains a category that is not visible in the image.
- Synthetic prompt targets are treated as ground truth without human review.
- Box-output samples are mixed into descriptive SFT, then compared against a text-only baseline.
- Long responses improve qualitative impressions but reduce final-answer precision.

## RL: What To Learn First

RL should be introduced only after an SFT model can already follow the task format. In RailMind, the reward is deliberately narrow: compare the final predicted anomaly-category set with the reviewed category set.

The release uses:

- `1.0` for exact category-set match.
- `0.5` for strict superset predictions.
- `0.0` for missed labels, unknown labels, invalid JSON or malformed final answers.

This reward is useful because it is cheap, deterministic and aligned with one visible behavior: final category selection. It does not measure localization, risk explanation quality or whether the chain-of-thought is sensible. Those must be evaluated separately.

Important RL skills:

- Parse only the final answer. A category mentioned in reasoning text is not a prediction.
- Log malformed-output rate, exact-match rate and superset rate, not just mean reward.
- Keep SFT-only, SFT+GRPO and SFT+GSPO checkpoints separate.
- Use the same evaluation manifest for all variants.
- Watch for overprediction: the `0.5` superset reward can improve recall while hurting precision.
- Avoid writing a custom RL trainer unless the framework behavior is first reproduced with a tiny fixture.

GRPO and GSPO are framework-level choices, not magic labels. In the included ms-swift snapshot, GRPO and GSPO differ in importance-sampling behavior. The manuscript text and the recovered code are not perfectly identical, so this repository documents the discrepancy rather than inventing a new unverified optimizer.

## Suggested Learning Route

Start from the small CPU path:

```bash
python -m railmind demo --output outputs/demo
python -m unittest discover -s tests -v
```

Then inspect the generated files:

- `outputs/demo/manifest.jsonl`
- `outputs/demo/reviewed.jsonl`
- `outputs/demo/sft/train.json`
- `outputs/demo/rl/train.jsonl`
- `outputs/demo/eval/metrics.json`

After that, move to a real dataset in stages:

1. Build a manifest and verify splits.
2. Run teacher annotation as a dry-run, then with a small paid/API batch.
3. Review outputs and export SFT.
4. Train a small LoRA run and evaluate formatting plus category behavior.
5. Add reviewed synthetic images.
6. Export RL prompts and run a short RL sanity check.
7. Compare SFT-only, GRPO and GSPO on the same benchmark.

The goal is to catch data and reward mistakes before expensive training. In post-training, a one-hour data audit often saves several days of GPU ambiguity.

## Community And Tooling Map

These are the communities and project families that match RailMind's workflow. Use them as entry points when replacing components or extending the project.

| Topic | Useful Project Families | What To Look For |
|---|---|---|
| Qwen VLM SFT | Qwen model docs, LLaMA-Factory, ms-swift | Supported chat template, image processor, LoRA merge, multi-GPU launch |
| RLHF/RLAIF for LLM/VLM | ms-swift, TRL-style recipes, OpenRLHF-style systems | GRPO/GSPO/PPO support, reward plugin API, rollout batching, logging |
| Synthetic data | Image-editing APIs, NanoBanana-like workflows, review tools | Prompt reproducibility, task IDs, resume behavior, human approval records |
| Visual grounding | GroundingDINO/Open-GroundingDino, COCO/ODVG tools | Label mapping, phrase extraction, IoU evaluation, checkpoint compatibility |
| Evaluation | COCO-style evaluation, custom macro-F1 scripts, error analysis notebooks | Class macro vs scene mean, duplicate detections, empty predictions |
| Data governance | DVC-like manifests, checksums, dataset cards | Hash locks, split records, private-data boundaries, license notes |

When reading community examples, check four details before copying a command:

- Does the example target the same base model family and template?
- Does it use the same multimodal message format?
- Does its batch size mean per-device batch, gradient batch or rollout batch?
- Does its reward parse final answers, full responses or hidden reasoning?

Those small differences often explain why two similar-looking training commands produce different behavior.

## RailMind-Specific Advice

For this repository, the safest public narrative is:

- SFT transfers railway inspection vocabulary, defect categories and answer structure into Qwen3-VL.
- NanoBanana-style augmentation expands long-tail visual cases, but every synthetic image must be reviewed.
- RL is an optional category-decision refinement stage, not the source of railway knowledge.
- GroundingDINO remains the localization component and should be evaluated with IoU-based matching.
- The 20,931 SFT profile, 24,265 mixed server profile and 3,134 RL prompt profile must be reported separately.

If the project is later expanded, good next additions would be:

- A tiny public toy dataset with permissive images, so new users can run a real teacher-free export path.
- A result-card template that records dataset hashes, model IDs, exact prompts, reward version and benchmark split.
- A notebook for per-class confusion and failure-case browsing.
- A minimal container recipe for one verified GPU stack.

## Reading Checklist

Before running full training, read these files in order:

1. `docs/02_data.md`
2. `docs/03_data_engine.md`
3. `docs/04_augmentation.md`
4. `docs/05_training.md`
5. `docs/06_rl.md`
6. `docs/07_evaluation.md`
7. `docs/08_reproducibility.md`

This page is the bridge between those operational docs and the broader post-training skill set needed to adapt RailMind to a new inspection domain.
