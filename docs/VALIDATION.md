# Release validation records

[English](../README.md) · [简体中文](zh-CN/VALIDATION.md)


RailMind requires Python >=3.10. Historical server checks, GitHub CI and unverified GPU/API work are distinguished below.

## Historical server checks

On 2026-10-07, the source server used Python 3.14.6 to validate the dependency-free CPU workflow:

- Synthetic CPU demo passed without API calls or GPU training.
- 27 regression tests passed, covering export, rewards, box matching, invalid formats, split/lineage leakage, review hashes, resume handling and provider task-ID persistence.
- Editable installation in a separate virtual environment passed.
- Installed CLI help and offline demo passed.
- Wheel building, isolated installation and demo execution outside the source directory passed.
- Release scanning checked Python/JSON syntax, documentation links, credential patterns and upstream hashes.
- Framework snapshots contained 738 source files with upstream licenses and per-file SHA256.
- Private benchmark audit: 820 records, 820 images and 2,083 annotations, with no missing references.
- Private detector train2 audit: 3,334 images and 10,613 annotations.

## 2026-10-20 source-release validation

The server candidate again passed 27 tests, release scanning and the offline demo. Sample/subset ID lists, including packaged-resource copies, were excluded before release; only provenance statistics were retained.

The initial release passed GitHub CI on Python 3.10, 3.11 and 3.12: installation, 27 tests, offline demo and scanning. See the [historical run](https://github.com/chaomacv/RailMind/actions/runs/39425682658).

## Public documentation asset checks

Seven additional regression tests validate the exact-path/content-hash exception for public README illustrations while ensuring that replacements, unlisted images, training records, model payloads and credentials remain blocked. The corresponding [CI run](https://github.com/chaomacv/RailMind/actions/runs/41305630035) passed on Python 3.10, 3.11 and 3.12.

Training/evaluation datasets and manifests remain excluded. These checks do not validate CUDA operators, all GPU dependency combinations, live NanoBanana/teacher services or historical paper training.

## Local checks

Run from the source root:

```bash
python -m unittest discover -s tests -v
python -m railmind demo --output outputs/new-validation-run
python scripts/check_release.py
```

Release source, configuration and documentation, plus the explicitly approved public illustrations. Exclude private local material, outputs, caches, virtual environments, generated datasets and weights.
