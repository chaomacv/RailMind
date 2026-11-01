# Post-training papers, curve interpretation and tuning

[English](../README.md) · [简体中文](zh-CN/10_post_training_paper_roadmap.md)


This roadmap moves from running SFT/RL to interpreting curves and choosing hyperparameters. It organizes existing papers rather than claiming exhaustive coverage of current research. Original titles and links are retained below.

For a domain VLM such as RailMind, ask whether the model learns railway visual evidence or just answer style; whether SFT improves the task rather than loss on biased data; whether RL improves decisions or creates overconfident shortcuts; and whether learning rate, KL, reward and output length support checkpoint selection.

## Three levels of post-training

1. **SFT learns the response distribution:** domain style, format, evidence expression and common task decomposition.
2. **Preference optimization learns relative preferences:** DPO/IPO/KTO/ORPO/SimPO compare better and worse answers, generally without online environment interaction.
3. **Reward-based RL optimizes scores:** PPO/GRPO/DAPO/GSPO sample, score and update responses while controlling policy drift.

Use SFT for missing or unstable target behavior, preference optimization when preferred/rejected pairs are available, and online RL when rewards are reliable, cost is manageable and SFT alone is insufficient.

## Development timeline

| Period | Representative work | Main idea | Contribution |
|---|---|---|---|
| 2017–2019 | PPO; Fine-Tuning Language Models from Human Preferences | Constrained policy updates, rewards from human judgments | RLHF foundations |
| 2021–2022 | FLAN, LoRA, InstructGPT, Self-Instruct, Constitutional AI | Instructions, low-rank adaptation, RLHF, synthetic tasks, AI feedback | Instruction following |
| 2023 | LLaVA, InstructBLIP, QLoRA, LIMA, DPO, RRHF | Visual instructions, quantized adaptation, data quality, preference losses | Lower adaptation/reproduction cost |
| 2024 | KTO, ORPO, SimPO, DeepSeekMath/GRPO | Simplified/reference-free preferences, group-relative RL | Reduced overhead and reasoning applications |
| 2025–2026 | Qwen2.5-VL/Qwen3-VL reports, DeepSeek-R1, DAPO, GSPO, GRPO theory | Large-scale VLMs, verifiable rewards, sequence/group policy optimization | Reasoning and multimodal domain tasks |

## Reading the main papers

### SFT and instruction tuning

1. **FLAN:** understand multi-task instruction mixtures and zero-shot transfer; view teacher-generated railway tasks through this lens.
2. **Self-Instruct:** study synthetic instructions. Railway use needs stronger supervision and stricter review because labels and localization have operational consequences.
3. **LIMA:** examine curated data quality rather than maximizing generation volume.
4. **LoRA / QLoRA:** distinguish trainable parameter scope from quantization-related memory savings. Neither fixes bad data.
5. **LLaVA / InstructBLIP:** understand why visual instruction tuning needs carefully designed image-text tasks rather than text-only SFT.
6. **Qwen VLM reports:** inspect resolution, visual tokens, multilingual/multimodal capabilities and post-training assumptions of the base family.

### RLHF, preferences and verifiable rewards

1. **PPO:** clipping, policy ratios and stable updates; understand failure modes without necessarily reimplementing the optimizer.
2. **Human Preferences:** learn the reward-model-then-policy sequence.
3. **InstructGPT:** connect SFT, preferences, reward models, PPO, KL control and evaluation.
4. **Constitutional AI:** study rule-guided AI feedback, which reduces some human-review cost without removing verification.
5. **DPO:** train directly from preference pairs without an explicit reward model or online PPO loop.
6. **RRHF / IPO / KTO / ORPO / SimPO:** explore simplified preference objectives when PPO cost or stability is unsuitable.
7. **DeepSeekMath / GRPO:** compare multiple responses to one prompt within a group and reduce dependence on a separate value model.
8. **DeepSeek-R1:** focus on verifiable rewards and staged reasoning training; reward/task selection determines signal reliability.
9. **DAPO / GSPO:** dynamic sampling, decoupled clipping and sequence-level optimization at scale.
10. **GRPO theory:** after implementation experience, study estimators, bias, noise and length sensitivity.

## Interpreting SFT curves

Lower training loss means better prediction of supervised tokens, not necessarily better reasoning, localization or classification.

| Signal | Healthier pattern | Warning | Investigation |
|---|---|---|---|
| Training loss | Smooth decline then plateau | Rapid decline without validation gains | Lower LR/epochs; improve diversity |
| Validation loss | Declines or stays close to training | Rises after early training | Early stopping, fewer epochs, split audit |
| Format validity | Improves promptly | Oscillates or degrades | More format examples, lower LR |
| Class macro-F1 | Improves on validation | Frequent classes alone improve | Balance data or class sampling |
| Output length | Stable near the target | Longer without metric gains | Audit constraints, supervision and reward |
| Hallucinated classes | Decrease | Generation targets replace visible evidence | Stricter review; remove unverified synthetic data |

Select checkpoints using validation macro-F1, format validity, per-class precision/recall and inspected failures, not SFT loss alone.

## Tuning SFT

Domain VLMs are often constrained by data quality before optimization. These are exploratory starting points, not universal settings.

| Parameter | Conservative starting point | Too high | Too low |
|---|---|---|---|
| Learning rate | Explore LoRA 5e-6–1e-5; full tuning may need less | Format drift, forgetting, unstable loss | Slow learning, domain underfit |
| Epochs | Try 1–3 for curated repeated data | Memorization, validation decline, repetition | Insufficient format learning |
| LoRA rank | Try 8–32 | Memory use, small-data overfit | Insufficient adaptation capacity |
| Warmup fraction | Try 0.03–0.10 | Wasted update budget | Early spikes |
| Weight decay | Small value or framework default | Can hinder adapter learning | Weak regularization |
| Cutoff/max length | Fit image prompts and final format | Cost and verbosity | Truncated evidence or final JSON |
| Image-pixel budget | Match base-model settings first | High memory use | Lost small-defect detail |

Change one main variable at a time: find a stable LR on small runs, compare data recipes at fixed LR, then rank/epochs, then scale GPUs/batch. Verify configuration sanity in the first experiment.

## Interpreting RL curves

Reward can improve while real quality declines. Track signals jointly:

| Signal | Healthier pattern | Warning | Possible cause |
|---|---|---|---|
| Mean reward | Rises then stabilizes | Immediately very high | Trivial reward or leakage |
| Exact match | Rises | Reward rises but matches do not | Superset/format loopholes |
| Superset frequency | Controlled | Continually rises | Overprediction |
| KL to reference | Nonzero and bounded | Near zero or exploding | No learning or policy drift |
| Entropy | Gradual decline | Early collapse | Mode collapse |
| Output length | Stable and task-appropriate | Continually grows for score | Unconstrained verbosity |
| Invalid JSON rate | Falls | Remains high | SFT not ready |
| Per-class recall | Improves without precision collapse | Rare classes disappear | Reward/data imbalance |

Plot exact match, supersets, invalid outputs and macro-F1 on the same validation set. A model that always predicts extra defects may earn partial reward while being less suitable for inspection. These are diagnostic clues, not causal proof from a single curve.

## Tuning RL

Use cautious updates. A single run causing abrupt format or behavior changes warrants an update-strength review.

| Parameter | Meaning | Too high | Too low |
|---|---|---|---|
| RL learning rate | Policy-update strength | KL spikes, drift, reward gaming | Little improvement |
| KL coefficient | Constraint toward reference | Blocks learning | Drift/collapse |
| Clip range | Policy-ratio update limit | Instability | Slow learning |
| Group size / rollouts | Within-group comparison quality | Cost; possible preference for lucky long samples | Noisy advantage estimates |
| Max response length | Search budget | Verbosity/cost | Truncation |
| Temperature | Exploration | Low-quality noise | Insufficient exploration |
| Reward weights | Allocation across behaviors | Exploits easy terms | Weak signal |

Confirm SFT format validity first. Run a tiny RL experiment and inspect raw generations. Plot reward, KL, length and invalidity every few steps. Stop early if length or overprediction explains reward gains. Validate every checkpoint with actual task metrics.

## A staged RailMind experiment plan

- Poor class understanding: improve SFT data.
- Good understanding with over/underprediction: explore class RL.
- Poor localization: inspect GroundingDINO and box annotations; class RL does not directly fix coordinates.
- Better validation scores but worse inspected answers: audit metrics, reward and labels.

Suggested progression:

1. `SFT-small`: 200–500 reviewed samples, one GPU, short run; check format and visible evidence.
2. `SFT-domain`: full reviewed descriptions, conservative LR, 1–3 epochs.
3. `SFT-ablation`: real-only versus real plus reviewed synthetic data.
4. `RL-smoke`: 100–300 reviewed synthetic prompts, low LR, short integrity run.
5. `RL-full`: fixed SFT checkpoint and full reviewed RL data; plot exact match, supersets, KL and length.
6. `Joint-eval`: the same private 820-image benchmark for SFT, GRPO, GSPO and detector variants.

Save exact data/prompt hashes, configuration, seeds, framework versions, logs and sampled generations so curves refer to traceable experiments. Keep samples and data manifests private.

## Papers and original links

The following source links are retained from the original reading list:

- PPO: Proximal Policy Optimization Algorithms - https://arxiv.org/abs/1707.06347
- Fine-Tuning Language Models from Human Preferences - https://arxiv.org/abs/1909.08593
- FLAN: Finetuned Language Models Are Zero-Shot Learners - https://arxiv.org/abs/2109.01652
- LoRA: Low-Rank Adaptation of Large Language Models - https://arxiv.org/abs/2106.09685
- InstructGPT: Training Language Models to Follow Instructions with Human Feedback - https://arxiv.org/abs/2203.02155
- Self-Instruct: Aligning Language Models with Self-Generated Instructions - https://arxiv.org/abs/2212.10560
- LLaVA: Visual Instruction Tuning - https://arxiv.org/abs/2304.08485
- InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning - https://arxiv.org/abs/2305.06500
- QLoRA: Efficient Finetuning of Quantized LLMs - https://arxiv.org/abs/2305.14314
- LIMA: Less Is More for Alignment - https://arxiv.org/abs/2305.11206
- Qwen2.5-VL Technical Report - https://arxiv.org/abs/2502.13923
- Qwen3-VL Technical Report - https://arxiv.org/abs/2511.21631
- Constitutional AI: Harmlessness from AI Feedback - https://arxiv.org/abs/2212.08073
- RRHF: Rank Responses to Align Language Models with Human Feedback without tears - https://arxiv.org/abs/2304.05302
- DPO: Direct Preference Optimization - https://arxiv.org/abs/2305.18290
- IPO / general preference learning theory - https://arxiv.org/abs/2310.12036
- KTO: Model Alignment as Prospect Theoretic Optimization - https://arxiv.org/abs/2402.01306
- ORPO: Monolithic Preference Optimization without Reference Model - https://arxiv.org/abs/2403.07691
- SimPO: Simple Preference Optimization with a Reference-Free Reward - https://arxiv.org/abs/2405.14734
- DeepSeekMath and GRPO - https://arxiv.org/abs/2402.03300
- DeepSeek-R1 - https://arxiv.org/abs/2501.12948
- DAPO: An Open-Source LLM Reinforcement Learning System at Scale - https://arxiv.org/abs/2503.14476
- GSPO: Group Sequence Policy Optimization - https://arxiv.org/abs/2507.18071
- Demystifying GRPO: Its Policy Gradient is a U-Statistic - https://arxiv.org/abs/2603.01162

## One-week reading order

1. FLAN, LoRA and QLoRA
2. InstructGPT
3. LLaVA or InstructBLIP
4. DPO
5. DeepSeekMath/GRPO
6. DeepSeek-R1
7. DAPO or GSPO

After each paper, write: “This paper would make me change ____ in RailMind because ____.” Turn reading into engineering judgment rather than memorizing algorithm names.
