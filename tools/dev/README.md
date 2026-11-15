# Development tooling (outside the official harness)

These scripts were used to choose MixTTA settings. They feed cached images to a method in batches of 64, as the harness does. They do not produce submissions; use `starter/run.py` for that.

Cached tensors go in `tools/dev/cache/` (git-ignored), or in `$HW2_DEV_CACHE` if it is set. Run the scripts from this directory.

| Script | Purpose |
|---|---|
| `cache.py` | Decode val and test PNGs once into `cache/{val,test}.pt` (uint8 tensors; labels for val) |
| `segment.py <split> <K>` | Split a stream into K domains by dynamic programming on early-layer statistics of the source model. Writes `<split>_bounds.npy` |
| `val_bounds.npy`, `test_bounds.npy` | Domain boundaries found by `segment.py` (val: 776 / 1500; test: 11 boundaries, domains of 676–1252 images) |
| `fe.py` | Fast evaluator: `val7` (7 val stream orders, per-domain accuracy), `proxy` (synthetic test-like streams), `longproxy` (the 4 proxy streams back to back), `run` |
| `proxy.py <mix>` | Labelled proxy for test: val images with synthetic versions of the 12 test corruptions, blended with the original image at strength `mix` (0.5 used) |
| `mx.py`, `exp.py` | Experimental MixTTA-MS variant with switchable options (extra branches, shift detection, priors, train-before-predict), and the experiment runner. Results are in `exp_log.txt` |
| `pv.py`, `lp.py` | Proxy + val7, and long proxy, for methods registered in `starter/methods.py` |
| `rawshare.py`, `collapse.py` | Label-free collapse checks on the test stream (class shares of the raw ensemble argmax per domain, detected shifts) |
| `shiftsim.py` | Checks offline that the domain-shift detector fires at the true boundaries |
| `balance.py`, `record.py` | Offline per-domain class-balance analysis on recorded probabilities |
| `seeds.sh <method>` | Test runs with seeds 1–3 (stability check) |

## Calibration against Kaggle public scores

The 0.5 proxy ranks the 5 scored MixTTA methods in the same order as Kaggle. It is harder in absolute terms: `mixtta_ms224` scores 80.1 on the proxy and 0.916 on Kaggle.

| Method | Proxy 0.5 | Kaggle |
|---|---:|---:|
| `bn` | 62.70 | 0.720 |
| `mixtta` | 73.54 | 0.862 |
| `mixtta_st_notrain` | 77.59 | 0.890 |
| `mixtta_st2` | 79.11 | 0.910 |
| `mixtta_ms224` | 80.13 | 0.916 |

Neither proxy shows long-stream drift reliably:

- The short proxy streams last 33 batches, while test lasts 192.
- The long proxy reuses each val image 4 times.

Always check new self-training variants on the test stream itself, which needs no labels: look at the class shares per domain (`rawshare.py`).
