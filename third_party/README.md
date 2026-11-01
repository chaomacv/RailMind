# Third-party source snapshots

These snapshots are selected copies of necessary framework sources from the existing server workspace, recording the implementations used in the research workflow. Upstream licenses are retained, and per-file SHA256 hashes are recorded in `source_lock.json`.

- `groundingdino/`: the offline package's Open-GroundingDino training implementation, using `models/GroundingDINO` rather than the official `groundingdino.models` package layout. Its CUDA operators must be compiled separately.
- `llamafactory/`: the offline package's Qwen3-VL-capable code. The server's older top-level LLaMA-Factory installation has a different set of templates.
- `ms-swift/`: the server's `4.0.0.dev0` source. Its GSPO example uses sequence-level importance sampling; the original railway-specific RL launch script has not been recovered.

Hashes identify the retained file contents, not a verified upstream Git commit. Install framework dependencies and GPU environments separately; the CPU workflow does not import these frameworks. README files, licenses and comments inside each upstream snapshot are preserved in their original language to retain provenance and checksums.
