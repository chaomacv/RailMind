# Workflow And Module Boundaries

The research question is whether domain-adapted semantic reasoning plus specialized grounding is more reliable than requiring an MLLM to generate coordinates directly.

1. Partition real images by acquisition/source group before any teacher annotation or augmentation. Preserve manual category/box annotations separately from model responses.
2. Generate structured teacher supervision using visible component state, scene context, category definitions and cautious risk descriptions. Review outputs before training.
3. For augmentation, choose a training background and a training reference with the same scene. Apply the intended defect without changing unrelated geometry or transferring annotations. Verify the image and its actual labels.
4. Train Qwen on reviewed descriptions, and train the detector on human-reviewed boxes. These are independently optimized components, not one joint end-to-end loss.
5. Optionally apply category-set RL to the SFT model using verified synthetic labels. Generation intent is insufficient ground truth.
6. At inference, Qwen returns explicit final categories. The detector receives those categories, not evaluation GT, as its caption.
7. Evaluate category recognition without box matching, and localization with same-category IoU matching. Keep scene-mean and class-macro averages separate.

## Paper-To-Code Map

| Manuscript module | Release implementation | Source evidence |
|---|---|---|
| Teacher data engine | `railmind/api.py`, `prompts/teacher.txt`, `review` | Annotation scripts and domain prompts; new portable adapter |
| NanoBanana defect transfer | `railmind/augment.py`, `prompts/augmentation.json` | Existing scene-aligned pipeline and 13 prompt mappings |
| Qwen SFT | `export-sft`, SFT recipe profiles | Offline LLaMA-Factory bundle and training YAML |
| Detector training | `export-dino`, source-matched config and snapshot | Open-GroundingDino train2 setup |
| GRPO/GSPO | `railmind/reward.py`, reward plugin and recipe profiles | Manuscript reward and server ms-swift examples; original railway launch missing |
| Reasoning-guided grounding | `semantic`, `ground` | Qwen-result-driven detector evaluation scripts |
| Classification/detection metrics | `railmind/evaluation.py` | Manuscript protocols; clarified pooling and matching |

The new teacher prompt uses concise observable evidence and structured outputs. It does not pretend to be the exact original prompt or recover hidden reasoning. Prompt hashes make future runs traceable.

The source release deliberately excludes unrelated banking and RAG demos, old website user storage, duplicate model frameworks, weights, failed compression fragments and bulk experiment outputs.
