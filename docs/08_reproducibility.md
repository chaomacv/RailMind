# Reproducibility Audit And Release Status

This source release withholds all training and evaluation datasets. CPU data/evaluation tests establish workflow correctness, not GPU training success or manuscript accuracy.

## What Was Recovered

| Item | Evidence/status |
|---|---|
| Current benchmark | 820 COCO records and 820 images, 2,083 annotations, no missing files |
| Scene counts | Match manuscript 275/88/128/11/227/91 |
| 810 subset | Can explicitly derive image IDs 1-810; old historical file not independently recovered |
| Qwen later mixed SFT | 24,265 records; portable source/config available |
| Manuscript SFT | 20,931 records described; exact original launch/config not recovered |
| GroundingDINO | train2 source, local labels/config and necessary training code recovered |
| NanoBanana | Scene-aligned generation implementation, async protocol and 13 prompts recovered |
| RL | ms-swift engine/examples recovered; exact railway-specific launch and 3,134-prompt dataset not identified |
| Model weights/data | Remain private/off-repo; available offline bundle is a different artifact |

Detailed source counts and annotation hashes are in [source_inventory.json](source_inventory.json). Per-file framework hashes are in `third_party/source_lock.json`. Those hashes identify copied source contents; they do not certify a GPU environment.

## Material Differences From The Draft

- The old folder name says 810, but its current content matches the 820-image manuscript benchmark.
- The later mixed SFT experiment adds 3,334 detector box-output examples to the 20,931 description records. Its config cannot be used to claim descriptive-only manuscript results.
- Manuscript SFT LR is 5e-6; portable LoRA LR is 1e-5; another old server config uses 5e-5. Profiles remain separate.
- The manuscript describes GSPO with an absolute baseline; the recovered engine example uses sequence-level importance sampling. Use source-backed framework semantics and correct the manuscript explanation separately.
- Scene-mean, class-macro and harmonic-mean F1 definitions require reconciliation before recreating tables.
- Some old scripts use nonexistent `zcy` Conda paths and hardcoded secrets. The release uses active interpreters and environment variables.
- The current website's Qwen2.5 endpoint is a separate legacy deployment, not the Qwen3-VL paper training artifact. Its earlier MinIO routing failure does not imply paper training checkpoints are missing.

## Release Payload

Included: newly organized CLI, schemas, prompts, config profiles, reward plugin, evaluation, regression tests, documentation, selected third-party code and licenses.

Excluded: actual railway image/label corpora, `.enc` files, model checkpoints, API keys, Cloudflare tokens, campus-network credentials, website users/chat histories, private manuscript PDF, banking/RAG subprojects, bulk result directories and incomplete compression files.

Do not upload the parent `RailMind/` directory. The approximately 62 GiB offline archive is not a GitHub commit. It contains a fuller private training snapshot, not a guarantee of complete RL or every manuscript ablation.

## Before Public Release

Select the author-approved license for new code; confirm release rights for any future examples/data/weights; specify official dataset/model download links and conditions; confirm the final paper citation; and validate the selected GPU dependency environments. These items are documented open release tasks, not silently claimed complete.

The author has authorized source-only publication. Training/evaluation images, annotations, supervision, manifests, and split or subset ID lists are excluded.

## Verification Levels

1. **CPU workflow:** demo, parser/data exports, reward, split leakage and evaluation tests.
2. **Mock API contract:** request structure, credential-free dry-runs and resume checks, without live paid calls.
3. **Framework launch:** executable/config resolution and dry-run commands. Source snapshots preserved, CUDA compilation and training not executed in this preparation.
4. **Paper reproduction:** requires private approved datasets, actual checkpoint/environment versions, original launch details and agreed evaluation aggregation. Not established by levels 1-3.
