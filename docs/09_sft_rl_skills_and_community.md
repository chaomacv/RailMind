# SFT/RL skills and community tools

[English](../README.md) · [简体中文](zh-CN/09_sft_rl_skills_and_community.md)


This guide is for readers familiar with Python/PyTorch who want to adapt RailMind to their own domain. It explains the mechanisms behind the post-training commands.

RailMind uses SFT for domain knowledge and output structure, then optionally uses class-reward RL to improve final decisions on reviewed synthetic images. Localization trains separately because it requires instance boxes, not language supervision alone.

## Why this guide exists

The VLM workflow covers domain supervision from images and expert prompts; stronger teachers generate initial descriptions, risk explanations and class evidence; human review precedes training; SFT teaches domain expression and answer format; RL is introduced only when reward is clear and automatically checkable; classification, localization and failures are evaluated separately.

SFT supports imitation and domain adaptation. RL optimizes narrow, measurable behavior after the model understands the task. RL cannot compensate for poor data, weak prompts or unclear class definitions.

## Skills matrix

| Area | Concepts | RailMind entry points |
|---|---|---|
| Data engineering | JSONL, hashes, splits, leakage | `manifest`, `verify`, `leakage`, review schema |
| Teacher annotation | Prompts, structured JSON, evidence, review | `annotate`, `review`, `prompts/teacher.txt` |
| SFT | Chat templates, multimodal messages, LoRA, batch, cutoff | `export-sft`, `configs/sft/*.json` |
| RL | Reward, final-answer parsing, rollouts, invalid formats | `export-rl`, `training/railway_reward.py`, `configs/rl/*.json` |
| Detection | ODVG/COCO export, classes, validation | `export-dino`, `configs/dino/*` |
| Evaluation | Macro-F1, scene averages, IoU, duplicates | `evaluate`, `docs/07_evaluation.md` |
| Engineering | Previews, isolation, source checks, tests | `train`, `scripts/check_release.py`, `tests/` |

## Understanding SFT

Before GPU training, ask what behavior the model should imitate, which fields are trusted labels rather than generated suggestions, and how you will detect shortcuts.

The objective is to inspect railway images, identify visible defects and explain evidence in a stable format. The 20,931-record paper recipe emphasizes descriptions; the later 24,265-record mixture includes 3,334 direct-box-output samples. Report these separately.

Essential practices:

- Use the base model's chat template; Qwen3-VL data cannot automatically use an older Qwen2-VL template.
- Keep paths portable and record image hashes.
- Split training, validation and test data before teacher calls.
- Review generated labels before treating them as training targets.
- Record effective batch as `GPU count × per-device batch × gradient accumulation`.
- Save resolved configuration, data hashes, seeds, framework versions and base-model identifiers.
- Treat LoRA rank, learning rate, cutoff length and image-pixel budget as experiment variables.

Typical failures include learning directory/filename shortcuts, invisible classes in teacher text, unreviewed generation targets treated as truth, box-output data mixed into description-only comparisons, and longer attractive answers with worse final-class precision.

## Understanding RL

Start after SFT reliably follows the output format. The reward compares final predicted and reviewed class sets: exact match earns `1.0`, a strict superset earns `0.5`, and omissions, unknown classes, invalid JSON or final-format errors earn `0.0`.

This cheap deterministic reward targets class selection. It does not measure localization, risk explanations or reasoning quality; evaluate those separately.

- Parse final answers only, not class mentions in reasoning.
- Record invalid-format rate, exact match, superset frequency and mean reward together.
- Save separate SFT, SFT+GRPO and SFT+GSPO checkpoints.
- Evaluate all variants on the same manifest.
- Monitor overprediction: partial superset reward may raise recall while harming precision.
- Validate framework behavior on small examples before considering a custom RL trainer.

Retained ms-swift GRPO/GSPO examples differ in importance sampling. The manuscript and recovered code are not fully aligned; the repository records the difference rather than inventing an unverified optimizer.

## Suggested learning sequence

Start with the CPU example:

```bash
python -m railmind demo --output outputs/demo
python -m unittest discover -s tests -v
```

Inspect:

- `outputs/demo/manifest.jsonl`
- `outputs/demo/reviewed.jsonl`
- `outputs/demo/sft/train.json`
- `outputs/demo/rl/train.jsonl`
- `outputs/demo/eval/metrics.json`

Then migrate incrementally to authorized real data:

1. Build manifests and check splits.
2. Preview teacher requests, then try a small number of live calls.
3. Review and export SFT data.
4. Run a small LoRA experiment and evaluate format/class behavior.
5. Add reviewed synthetic images.
6. Export RL prompts and perform a short integrity check.
7. Compare SFT, GRPO and GSPO on the same benchmark.

Find data/reward problems before expensive training. Auditing post-training data can be more effective than extending GPU runs without a diagnosis.

## Community and toolchain

These project families are possible replacement components or extensions; verify version capabilities against your environment.

| Topic | Projects/community | What to check |
|---|---|---|
| Qwen VLM SFT | Qwen documentation, LLaMA-Factory, ms-swift | Templates, processors, LoRA merging, multi-GPU launch |
| LLM/VLM RLHF/RLAIF | ms-swift, TRL, OpenRLHF | Algorithms, reward interfaces, rollout batches, logging |
| Synthetic data | Editing APIs, NanoBanana-style pipelines, review tools | Prompt provenance, task IDs, resume, manual review |
| Grounding | GroundingDINO/Open-GroundingDino, COCO/ODVG tools | Classes, phrase extraction, IoU, checkpoint compatibility |
| Evaluation | COCO tools, macro-F1 scripts, analysis notebooks | Class/scene averages, duplicates, empty predictions |
| Data management | DVC-style manifests, checksums, dataset cards | Hashes, splits, private-data boundaries, licensing |

Before copying a command, check whether the model family/template and multimodal format match, whether “batch” means per-device/gradient/rollout batch, and whether reward parses the final answer, full response or reasoning. These distinctions often explain different outcomes from superficially similar commands.

## RailMind priorities

- SFT transfers inspection terminology, defect classes and answer structure to Qwen3-VL.
- NanoBanana expands rare visual cases; all retained synthetic images require review.
- RL is optional class-decision optimization, not a replacement for domain learning.
- GroundingDINO handles localization, evaluated by IoU matching.
- Report 20,931-record SFT, 24,265-record mixed SFT and the paper's 3,134-prompt RL setting separately.

Potential additions include runtime-generated fictional teaching examples, experiment cards with data/prompt hashes and reward versions, class-confusion/failure browsers and a validated GPU container. Research datasets and sample-ID lists remain private.

## Reading checklist

Before full training, read `docs/02_data.md` through `docs/08_reproducibility.md` in order. This guide connects operational documentation to post-training concepts for transfer to new inspection domains.
