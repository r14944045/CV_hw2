# Draft: report changes for the overnight results

**Superseded (2026-11-20):** applied to report.md for `mixtta_lfsq4`; report.md is the current text, and this draft's prior formula is incomplete.

Written for the morning. It assumes the final file is `mixtta_lfsq` or `mixtta_lfsq4` (see OVERNIGHT.md for the
choice). Numbers are from the run logs and tools/dev/round*_log.txt.

## Q2 Method: what changes
- **Final model:** `mixtta_lfsq` (Kaggle public 0.94042, val 95.27 %), or `mixtta_lfsq4` (0.94319, val 95.93 %).
- **Tracking:** fixed, with no shift detection or split. The statistics follow the stream with rate 0.4 and are mixed
  50/50 with the source statistics. The local class prior collapsed on the public board when combined with
  shift-reset tracking (0.69 / 0.83), and only worked with fixed tracking.
- **Local class prior** (developed in a parallel session, chosen on val). The self-training target and the
  prediction combine three terms:
  - the image's own probability (2/3);
  - its LAME output (1/3);
  - the log of a class prior from nearby predictions: the LAME outputs of the current and previous batch, weighted by
    exp(-distance / 12) in stream position, plus a pseudo-count of 0.2.
- **Adaptive LAME temperature:** per batch, the temperature (between 1 and 3) that brings the mean max-probability
  to 0.3, instead of a fixed T = 2.
- **Per-stage learning rates:** ×(2, 2, 2, 1, 0.25) for stem, layer1–layer4 (`mixtta_lfsq`), or ×(4, 4, 4, 1, 0.1)
  (`mixtta_lfsq4`).

## Q3 Strategy: rows to add to the strategy table
| Strategy (`--method`) | Val (%) | Kaggle public |
|---|---:|---:|
| + local class prior on fixed tracking (`mixtta_local_fixed`) | 95.12 | 0.92789 |
| + per-stage learning rates (`mixtta_lfs`) | 95.31 | 0.93815 |
| + adaptive LAME temperature (`mixtta_lfsq`) | 95.27 | 0.94042 |
| + stronger per-stage rates ×(4, 4, 4, 1, 0.1) (`mixtta_lfsq4`) | 95.93 | 0.94319 |
| local class prior on shift-reset tracking (`mixtta_r8` / `mixtta_local`), rejected | 96.69 / 96.07 | 0.82682 / 0.68554 |

## Q3 text to add (short)
- **Robustness to unseen corruptions:** the private board uses other corruption types. Settings were therefore also
  judged on the ImageNet-C corruptions (Hendrycks & Dietterich, 2019) added to val, as the FAQ allows. The published
  list was split mechanically: one half for choosing, the other half for confirmation.
- **Published components tested on this suite and not kept:**
  - EATA filtering;
  - ROID certainty/diversity weights, weight ensembling and prior correction;
  - DELTA's reweighting (below the margin);
  - SAR model recovery (never triggered);
  - NOTE instance-aware BN (much worse);
  - TRIBE balanced BN and UnMix-TNS (both failed a screen);
  - ImageNet-C augmentation of the self-training copy (worse).
- **A lesson:** val-based metrics reuse val's 33-batch label order. They could not see the slow, self-reinforcing
  drift that made the local prior collapse with shift-reset tracking; only the public board showed it.

## Page budget
The report is currently 5 pages (pandoc, 2 cm margins, 10 pt), so these rows and about 10 lines fit within 6 pages.
The Q4 figure would need regenerating from the final method's val run
(`python tools/plot_val_examples.py --method <final>`).
