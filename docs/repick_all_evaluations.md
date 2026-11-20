# Val-only re-pick: all evaluations

All 35 evaluations of `tools/dev/repick.py`, in the order the stages ran. Each stage starts from the settings kept in the stages before it, and its first row repeats that setting.
- **Score:** the mean over the 7 val stream orders × 4 boundary positions × 2 seeds.
- **Long stream:** the 7 orders back to back. A setting is vetoed if one class takes more than 10 % of the predictions at its end (5 % is balanced), or if accuracy drops by more than 5 pt. No candidate was vetoed.
- **Stages 2–3:** each tracking setting gets its own shift threshold from the val rule in `tools/dev/threshold_val.py`.
- **Results:** in `tools/dev/repick_results.jsonl`, with the decision log in `tools/dev/repick_log.txt`.

| Stage | Candidate | Score (%) | Seed 0 / 1 | Max class share, end of long stream (%) | Kept |
|---|---|---:|---|---:|:---:|
| 1. Steps × LAME T | 2 steps, T = 1 | 92.70 | 92.62 / 92.79 | 5.5 |  |
|  | 2 steps, T = 2 | 92.94 | 92.83 / 93.05 | 5.5 |  |
|  | 2 steps, T = 3 | 92.23 | 92.20 / 92.27 | 5.8 |  |
|  | 3 steps, T = 1 | 92.81 | 92.85 / 92.77 | 5.4 |  |
|  | 3 steps, T = 2 | 93.12 | 93.17 / 93.08 | 5.4 | ✓ |
|  | 3 steps, T = 3 | 92.39 | 92.45 / 92.34 | 5.8 |  |
|  | 4 steps, T = 1 | 92.83 | 92.77 / 92.89 | 5.2 |  |
|  | 4 steps, T = 2 | 93.33 | 93.22 / 93.45 | 5.3 |  |
|  | 4 steps, T = 3 | 92.56 | 92.50 / 92.62 | 6.0 |  |
| 2. Tracking | no reset (fixed rate 0.4) | 93.12 | 93.17 / 93.08 | 5.4 |  |
|  | reset, rate floor 0.4, threshold 6.8 | 93.02 | 93.11 / 92.93 | 5.2 |  |
|  | reset, rate floor 0.2, threshold 4.8 | 93.52 | 93.54 / 93.49 | 5.2 | ✓ |
|  | reset, rate floor 0.1, threshold 5 | 93.57 | 93.59 / 93.54 | 5.3 |  |
|  | reset, rate floor 0.05, threshold 5 | 93.54 | 93.55 / 93.52 | 5.3 |  |
| 3. In-batch split | off, threshold 4.8 | 93.52 | 93.54 / 93.49 | 5.2 |  |
|  | on, threshold 9 | 93.91 | 93.95 / 93.87 | 5.1 | ✓ |
| 4. Second pass | off | 93.91 | 93.95 / 93.87 | 5.1 | ✓ |
|  | first 1 batch | 93.98 | 94.01 / 93.95 | 5.1 |  |
|  | first 2 batches | 94.05 | 94.07 / 94.02 | 5.2 |  |
|  | first 3 batches | 94.09 | 94.09 / 94.09 | 5.2 |  |
| 5. Class prior | off | 93.91 | 93.95 / 93.87 | 5.1 | ✓ |
|  | λ = 1 | 93.96 | 93.98 / 93.93 | 5.2 |  |
|  | λ = 2 | 93.92 | 93.96 / 93.87 | 5.1 |  |
|  | λ = 3 | 93.80 | 93.84 / 93.75 | 5.1 |  |
| 6. Per-stage learning rates (stem, layer1–4) | none (×1 everywhere) | 93.91 | 93.95 / 93.87 | 5.1 |  |
|  | ×(2, 2, 2, 1, 0.25) | 94.16 | 94.23 / 94.10 | 5.2 | ✓ |
|  | ×(2, 2, 2, 1, 0.1) | 94.19 | 94.29 / 94.09 | 5.3 |  |
|  | ×(2, 2, 2, 2, 0.25) | 94.32 | 94.42 / 94.22 | 5.2 |  |
|  | ×(2, 2, 2, 2, 0.1) | 94.29 | 94.39 / 94.19 | 5.3 |  |
|  | ×(3, 3, 3, 1, 0.25) | 94.31 | 94.40 / 94.22 | 5.2 |  |
|  | ×(3, 3, 3, 1, 0.1) | 94.30 | 94.35 / 94.25 | 5.2 |  |
|  | ×(3, 3, 3, 2, 0.25) | 94.36 | 94.45 / 94.28 | 5.2 |  |
|  | ×(3, 3, 3, 2, 0.1) | 94.33 | 94.46 / 94.20 | 5.2 |  |
|  | ×(4, 4, 4, 1, 0.25) | 94.35 | 94.46 / 94.25 | 5.2 |  |
|  | ×(4, 4, 4, 1, 0.1) | 94.41 | 94.48 / 94.34 | 5.3 |  |
|  | ×(4, 4, 4, 2, 0.25) | 94.33 | 94.41 / 94.24 | 5.3 |  |
|  | ×(4, 4, 4, 2, 0.1) | 94.37 | 94.47 / 94.28 | 5.3 |  |
| 7. Adaptive LAME temperature | off (fixed T = 2) | 94.16 | 94.23 / 94.10 | 5.2 | ✓ |
|  | mean max-probability 0.4 | 94.30 | 94.36 / 94.24 | 5.2 |  |
|  | mean max-probability 0.3 | 94.31 | 94.30 / 94.32 | 5.2 |  |
|  | mean max-probability 0.2 | 94.18 | 94.28 / 94.07 | 5.3 |  |
