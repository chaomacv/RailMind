# Reproducibility audit and release status

[English](../README.md) · [简体中文](zh-CN/08_reproducibility.md)


Training and evaluation datasets are not released. CPU data/evaluation tests validate workflow behavior, not successful GPU training or the accuracy of paper results.

## Recovered materials

| Item | Evidence and status |
|---|---|
| Current benchmark | 820 COCO records, 820 images, 2,083 annotations, no missing references |
| Scene counts | Match the paper's 275/88/128/11/227/91 |
| 810 subset | Explicitly derivable comparison subset; historical original file not independently recovered |
| Later mixed Qwen SFT | 24,265 records with portable source/configuration |
| Paper SFT | Described as 20,931 records; exact historical launch not recovered |
| GroundingDINO | train2 source, private annotations/configuration and necessary training code recovered |
| NanoBanana | Same-scene generation, asynchronous protocol and 13-class prompts recovered |
| RL | ms-swift engine/examples recovered; original railway launch and exact 3,134-prompt dataset unconfirmed |
| Weights and data | Remain private; the offline reproduction package is separate |

Counts and annotation hashes appear in the [inventory](source_inventory.json). Framework file hashes are in `third_party/source_lock.json`. Hashes identify retained content; they do not establish GPU compatibility.

## Important differences from the manuscript draft

- A legacy directory says 810, while its contents match the paper's 820-image benchmark.
- Later mixed SFT adds 3,334 box-output records to 20,931 descriptions. It cannot establish the paper's description-supervision result.
- The paper learning rate is 5e-6, the portable LoRA recipe uses 1e-5, and another historical recipe uses 5e-5. Preserve their identities.
- The paper describes GSPO using an absolute-value baseline; recovered engine examples use sequence-level importance sampling. Follow actual framework semantics and reconcile the manuscript separately.
- Align scene-average, class-macro and harmonic aggregate P/R definitions before rebuilding result tables.
- Legacy scripts contained invalid `zcy` Conda paths and hardcoded credentials; released code uses the current interpreter and environment variables.
- The older website's Qwen2.5 endpoint is a separate deployment, not the paper's Qwen3-VL training setup. Its MinIO routing failure does not prove a missing research checkpoint.

## Release scope

Included: unified CLI, data schemas, generic prompts, experiment recipes, reward plugin, evaluation, regression tests, documentation and necessary third-party source/licenses.

Excluded: training/evaluation images, labels, teacher supervision, SFT/RL samples, manifests, split/subset IDs, encrypted files, weights, credentials, website accounts/conversations, private manuscript PDFs, banking/RAG projects, bulk experiment results and incomplete archives. Public README illustrations are separately approved documentation assets, checked by exact path and hash.

Do not upload the parent server `RailMind/` directory. The approximately 62 GiB offline package contains additional private training material; it is not a GitHub commit and does not guarantee complete RL or paper ablations.

## Remaining release decisions

The new-code license requires author selection. Later examples, datasets or weights need separate authorization checks. Official model acquisition, final citation and GPU dependency environments also need clarification. These are pending, not validated. Research datasets remain private in the current release.

## Validation levels

1. **CPU workflow:** demo, parsing/export, rewards, leakage checks and metrics.
2. **Mock API protocols:** request structure, credential-free previews and resume checks; no paid calls.
3. **Framework launch:** executable/configuration parsing and command previews; no CUDA compilation or training established by source assembly.
4. **Paper reproduction:** authorized private data, checkpoints, environment versions, historical launch details and consistent metrics are required. The first three levels do not establish this one.
