# Development tooling (outside the official harness, val only)

These scripts were used to choose MixTTA settings on the validation stream. They feed cached val images to a method in batches of 64, as the harness does. They read nothing from the test split. The test stream is read only by the official harness through `starter/run.py`, in a single online pass per submission.

Cached tensors go in `tools/dev/cache/` (git-ignored), or in `$HW2_DEV_CACHE` if it is set. Run the scripts from this directory.

| Script | Purpose |
|---|---|
| `cache.py` | Decode the val PNGs once into `cache/val.pt` (uint8 tensors and labels) |
| `segment.py val 3` | Split the val stream into its 3 domains by dynamic programming on early-layer statistics of the source model. Writes `val_bounds.npy` (boundaries at images 776 and 1500) |
| `fe.py` | Fast evaluator: `val7` (the 7 val stream orders, per-domain accuracy) and `run` |
| `threshold_val.py` | Chooses the domain-shift threshold `shift_thr` from val only (rule in its docstring) |
| `repick.py method mixtta_local` | Scores a registered method across all 56 validation runs and applies the 231-batch collapse check. Run `python cache.py` first to build the validation cache. |
| `gpu_slot.sh` | Runs a command once one of N GPU slots is free |
