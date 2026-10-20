# Local Release Validation

Validated on the source server on 2026-10-07 using Python 3.14.6 for the dependency-free CPU workflow. The project targets Python >=3.10; the GitHub CI matrix for 3.10/3.11/3.12 is defined but has not run remotely because the repository is not uploaded.

- CPU synthetic demo: PASS, no API requests or GPU training.
- Regression tests: 27 tests, covering exports, reward, box matching, malformed outputs, split/lineage leakage, review hashes, resume and provider-task persistence.
- Editable package installation in a workspace-local virtual environment: PASS.
- Installed CLI help and offline demo: PASS.
- Wheel build, separate wheel installation, and offline CLI demo from outside the source checkout: PASS.
- Release scan: Python/JSON syntax, documentation links, common credential patterns and third-party source hashes checked.
- Framework snapshot assembly: 738 source files, with original third-party licenses and per-file SHA256.
- Source benchmark audit: 820 image records, 820 disk images, 2,083 annotations, no missing references.
- Source detector train2 audit: 3,334 images, 10,613 annotations.

No claim is made that CUDA operators, all GPU dependency combinations, real NanoBanana/teacher services or historical paper training runs were executed. Those checks are outside the current release-preparation scope.

Reproduce the local checks from the source checkout:

```bash
python -m unittest discover -s tests -v
python -m railmind demo --output outputs/new-validation-run
python scripts/check_release.py
```

Only source/config/doc files belong in the publication payload. `local/`, `outputs/`, caches, virtual environments, generated datasets and model weights are excluded.

## Source-only release review (2026-10-20)

The server candidate passed 27 tests, release scanning and the offline demo. Public staging additionally removes dataset subset lists, including packaged asset copies, and retains only aggregate source statistics. All training/evaluation datasets are withheld. GPU training and real API calls remain unverified.
