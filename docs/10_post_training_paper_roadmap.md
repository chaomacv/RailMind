# Post-Training Paper Roadmap And Curve Reading

This note is for the point where "I can run SFT/RL" is no longer enough. The goal is to understand what the training curves are telling you, how to choose hyperparameters, and which papers explain the evolution of modern post-training.

RailMind is a domain VLM project, so the practical question is not "what is the newest algorithm?" but:

- Is the model learning railway evidence, or only memorizing answer style?
- Is SFT improving task behavior, or just lowering loss on a biased dataset?
- Is RL improving final category decisions, or creating overconfident shortcuts?
- Are learning-rate, KL, reward and output-length curves stable enough to trust the checkpoint?

## Mental Model

Post-training has three layers:

1. **SFT teaches the response distribution.** It makes the model copy domain answer style, schema, evidence language and common task decomposition.
2. **Preference optimization teaches relative preference.** DPO/IPO/KTO/ORPO/SimPO-style methods compare good and bad answers, usually without online environment interaction.
3. **RL with verifiable or learned reward optimizes a score.** PPO/GRPO/DAPO/GSPO-style methods sample outputs, score them and update the policy while trying not to drift too far.

Use SFT when the target behavior is missing or inconsistent. Use preference optimization when you can collect preferred/rejected pairs. Use online RL only when the reward is reliable, cheap enough and hard to imitate with SFT alone.

## Development Timeline

| Period | Representative Papers | Core Idea | Why It Matters |
|---|---|---|---|
| 2017-2019 | PPO; Fine-Tuning Language Models from Human Preferences | Constrain policy updates; learn reward from human judgments | Foundation of RLHF-style optimization |
| 2021-2022 | FLAN; LoRA; InstructGPT; Self-Instruct; Constitutional AI | Instruction tuning, low-rank adapters, RLHF pipeline, synthetic instructions, AI feedback | Turns base LMs into useful instruction followers |
| 2023 | LLaVA; InstructBLIP; QLoRA; LIMA; DPO; RRHF | VLM instruction tuning, quantized PEFT, quality-over-quantity SFT, direct preference losses | Makes alignment cheaper and easier to reproduce |
| 2024 | KTO; ORPO; SimPO; DeepSeekMath/GRPO | Reference-free or simplified preference optimization; group-relative RL | Reduces memory/engineering overhead and pushes reasoning tasks |
| 2025-2026 | Qwen2.5-VL/Qwen3-VL reports; DeepSeek-R1; DAPO; GSPO; GRPO theory papers | large-scale VLM training recipes, RLVR, sequence-level/group policy optimization, better theoretical analysis | Moves post-training from chat alignment toward reasoning and multimodal domain agents |

## Influential Papers To Read

Read these in order if you want a fast but serious foundation.

### SFT And Instruction Tuning

1. **FLAN: Finetuned Language Models Are Zero-Shot Learners**  
   Learn this for the basic idea that mixing many instruction-formatted tasks improves zero-shot behavior. RailMind's teacher-generated railway tasks are a domain-specific version of this idea.

2. **Self-Instruct: Aligning Language Models with Self-Generated Instructions**  
   Learn this for synthetic instruction generation. RailMind uses a stronger teacher model and stricter review because railway labels have safety and grounding consequences.

3. **LIMA: Less Is More for Alignment**  
   Learn this for the lesson that high-quality curated examples can dominate raw dataset size. This is a direct warning against blindly adding every generated railway sample.

4. **LoRA and QLoRA**  
   Learn these for adapter training and memory control. LoRA changes what parameters are trainable; QLoRA changes memory economics. Neither makes bad data good.

5. **LLaVA and InstructBLIP**  
   Learn these for visual instruction tuning. They explain why VLMs need careful image-text task construction, not only text SFT.

6. **Qwen2.5-VL and Qwen3-VL technical reports**  
   Read these to understand the base-model family behind the RailMind recipes: image resolution handling, visual tokenization, multilingual/multimodal capability and post-training assumptions.

### RLHF, Preference Optimization And RLVR

1. **PPO: Proximal Policy Optimization Algorithms**  
   Read this first for clipping and stable policy updates. You do not need to reimplement PPO, but you should recognize ratio clipping and why updates can collapse.

2. **Fine-Tuning Language Models from Human Preferences**  
   This is an early modern bridge from reward learning to language model fine-tuning. It makes the "reward model then optimize policy" pattern concrete.

3. **InstructGPT: Training Language Models to Follow Instructions with Human Feedback**  
   This is the canonical industrial RLHF pipeline: SFT, preference data, reward model, PPO, KL control and evaluation.

4. **Constitutional AI**  
   Read this for RLAIF: using AI feedback guided by written principles. It is relevant when human review is expensive, but it does not remove the need for validation.

5. **DPO: Direct Preference Optimization**  
   Read this for the major simplification: direct policy training from preference pairs without an explicit reward model and online PPO loop.

6. **RRHF, IPO, KTO, ORPO and SimPO**  
   These papers form the "make preference optimization simpler" branch. They are useful when PPO-style RL is too expensive or unstable for a project.

7. **DeepSeekMath and GRPO**  
   Read this for group-relative baselines: sample multiple outputs for the same prompt, compare them within the group and avoid a separate value model.

8. **DeepSeek-R1**  
   Read this for large-scale RL with verifiable rewards and staged reasoning training. The important lesson is not "RL solves everything"; it is that reward design and task selection decide whether RL has a clean signal.

9. **DAPO and GSPO**  
   Read these for newer large-scale RL engineering: dynamic sampling, decoupled clipping and sequence-level policy optimization. These are closer to modern reasoning-model training practice than classic single-response PPO examples.

10. **Recent GRPO theory papers**  
   Read these after implementation experience. They help explain what GRPO is estimating and when group-relative objectives can be biased, noisy or length-sensitive.

## How To Read SFT Curves

SFT curves are deceptively friendly. A falling training loss only says the model is becoming better at predicting the supervised tokens. It does not prove better reasoning, grounding or classification.

Watch these signals:

| Signal | Healthy Pattern | Warning Pattern | Typical Fix |
|---|---|---|---|
| Train loss | Smooth decline then flatten | Drops fast while validation stagnates | Lower LR, fewer epochs, more diverse data |
| Validation loss | Declines or stays close to train | Rises after early steps | Early stop, reduce epochs, improve split hygiene |
| Exact schema rate | Increases quickly | Fluctuates or degrades | Add format-focused examples, lower LR |
| Category macro-F1 | Improves on validation | Only frequent classes improve | rebalance data, class-aware sampling |
| Output length | Stable near target | Grows without metric gain | length penalty, shorter targets, RL reward audit |
| Hallucinated labels | Decreases | Model predicts target prompt labels not visible image labels | stricter review, remove unverified synthetic data |

For RailMind, do not choose checkpoints by SFT loss alone. Choose by validation category macro-F1, schema validity, per-class recall/precision and a small manually inspected failure set.

## How To Control SFT Hyperparameters

Start conservative. Domain VLM SFT is usually data-quality limited before it is optimizer-limited.

| Parameter | Conservative Starting Point | What Happens If Too High | What Happens If Too Low |
|---|---|---|---|
| Learning rate | LoRA: `5e-6` to `1e-5`; full fine-tune often lower | Format drift, forgetting, unstable loss | Slow learning, underfit domain terms |
| Epochs | 1-3 for high-quality repeated data | Memorization, worse validation, repeated phrasing | weak schema following |
| LoRA rank | 8-32 | More memory, overfit small data | Not enough capacity for domain adaptation |
| Warmup ratio | 0.03-0.10 | Wastes steps if too long | early spike if too short |
| Weight decay | small or framework default | can hurt adapter learning if aggressive | less regularization |
| Max length/cutoff | enough for image prompt + final schema | expensive, learns verbose answers | truncates evidence or final JSON |
| Image pixels | match base-model recipe first | memory blow-up | missed small defects |

When tuning, change one axis at a time:

1. Find a stable LR on a small run.
2. Fix LR, compare data recipes.
3. Fix data, compare LoRA rank and epochs.
4. Only then scale GPUs and batch size.

The first experiment should answer "is this setup sane?", not "what is the leaderboard number?"

## How To Read RL Curves

RL curves are easier to misread than SFT curves because reward can improve while real quality worsens.

Watch these signals together:

| Signal | Healthy Pattern | Warning Pattern | Meaning |
|---|---|---|---|
| Mean reward | rises then stabilizes | jumps to high reward immediately | reward too easy or leakage |
| Exact-match rate | rises | reward rises but exact-match flat | superset/format loophole |
| Superset rate | controlled | keeps rising | model is overpredicting classes |
| KL to reference | nonzero but bounded | collapses to 0 or explodes | no learning or policy drift |
| Entropy | slowly decreases | collapses early | mode collapse |
| Response length | stable or task-appropriate | grows just to game reward | reward misses verbosity |
| Invalid JSON rate | decreases | remains high | SFT not ready for RL |
| Per-class recall | improves without precision collapse | rare classes disappear | reward/data imbalance |

For RailMind category RL, the key plot is not only reward. Plot exact match, superset rate, invalid-output rate and macro-F1 on the same validation set. A model that always predicts extra railway defects can receive partial reward but still be worse for inspection.

## How To Control RL Hyperparameters

RL should move the model gently. If one run changes the model personality or output format dramatically, the update is probably too strong.

| Parameter | Practical Meaning | If Too High | If Too Low |
|---|---|---|---|
| RL learning rate | policy update strength | KL spike, format drift, reward hacking | no measurable improvement |
| KL coefficient | keeps model near reference | blocks learning | model drifts or collapses |
| clip range | limits policy ratio update | unstable updates | slow learning |
| group size / rollouts | reward comparison quality | expensive, may reward lucky long samples | noisy advantage estimates |
| max response length | search budget | verbosity and cost | truncated answers |
| temperature | exploration | noisy bad samples | no exploration |
| reward weights | what behavior is optimized | model exploits easy terms | signal too weak |

Good RL workflow:

1. Confirm SFT model has high schema-validity before RL.
2. Run a tiny RL job and inspect raw generations.
3. Plot reward, KL, length and invalid rate every few steps.
4. Stop early if reward rises by exploiting length or extra labels.
5. Evaluate the same checkpoint set with task metrics, not reward alone.

## RailMind-Specific Recommendations

For this project, use this decision rule:

- If the model does not understand railway categories, improve SFT data.
- If the model understands categories but chooses too many or too few, try category RL.
- If the model localizes poorly, train/evaluate GroundingDINO and box data. Category RL will not fix detector boxes.
- If validation metrics improve but manual inspection worsens, trust the inspection and audit the reward/labels.

Suggested experiment ladder:

1. `SFT-small`: 200-500 reviewed examples, one GPU, short run. Check format and visible evidence.
2. `SFT-domain`: full reviewed text dataset, conservative LR, 1-3 epochs.
3. `SFT-ablation`: remove synthetic data, compare real-only vs synthetic-reviewed.
4. `RL-smoke`: 100-300 reviewed synthetic prompts, low LR, short schedule.
5. `RL-full`: fixed SFT checkpoint, full reviewed RL data, plot exact/superset/KL/length.
6. `Joint-eval`: same 820 benchmark, compare SFT-only, SFT+GRPO, SFT+GSPO and detector variants.

The most important habit is to save the "boring" evidence: exact dataset hash, prompt hash, config, seed, framework version, logs and a sampled generation table. Without those, curves become stories you cannot replay.

## Paper List With Links

Foundations:

- PPO: Proximal Policy Optimization Algorithms - https://arxiv.org/abs/1707.06347
- Fine-Tuning Language Models from Human Preferences - https://arxiv.org/abs/1909.08593
- FLAN: Finetuned Language Models Are Zero-Shot Learners - https://arxiv.org/abs/2109.01652
- LoRA: Low-Rank Adaptation of Large Language Models - https://arxiv.org/abs/2106.09685

Instruction tuning and VLM SFT:

- InstructGPT: Training Language Models to Follow Instructions with Human Feedback - https://arxiv.org/abs/2203.02155
- Self-Instruct: Aligning Language Models with Self-Generated Instructions - https://arxiv.org/abs/2212.10560
- LLaVA: Visual Instruction Tuning - https://arxiv.org/abs/2304.08485
- InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning - https://arxiv.org/abs/2305.06500
- QLoRA: Efficient Finetuning of Quantized LLMs - https://arxiv.org/abs/2305.14314
- LIMA: Less Is More for Alignment - https://arxiv.org/abs/2305.11206
- Qwen2.5-VL Technical Report - https://arxiv.org/abs/2502.13923
- Qwen3-VL Technical Report - https://arxiv.org/abs/2511.21631

Preference optimization and RL:

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

## One-Page Reading Order

If you only have one week:

1. FLAN, LoRA, QLoRA
2. InstructGPT
3. LLaVA or InstructBLIP
4. DPO
5. DeepSeekMath GRPO
6. DeepSeek-R1
7. DAPO or GSPO

After each paper, write one sentence in this format:

`This paper changes my RailMind practice by _____.`

That habit turns paper reading into engineering judgment instead of name collection.
