# Source snapshots

The release assembly selects necessary source files from the existing server workspace, preserves upstream licenses, and records SHA256 for each selected file in `source_lock.json`.

- `groundingdino/`: the actual Open-GroundingDino training implementation from the offline bundle. It uses `models/GroundingDINO`, **not** the official `groundingdino.models` package. Build its own CUDA ops.
- `llamafactory/`: the Qwen3-VL-capable code from the offline bundle. The older top-level server `LLaMA-Factory/` has a different template set.
- `ms-swift/`: the server source tree, version `4.0.0.dev0`. Its GSPO example uses sequence-level importance sampling. No exact historical railway RL launch was recovered.

These are source snapshots with per-file hashes, not claims of specific upstream Git commits. Training framework dependencies and GPU runtime are installed separately; the CPU workflow does not import them. Upstream attribution remains in each snapshot's README and LICENSE.
