# Local class prior validation experiment

On validation, 24.8% of adjacent image pairs share a label. At a separation of 64 images the rate is 5.5%. This motivated estimating which classes are common near each image. The method uses predictions from the current batch and the immediately preceding batch. It keeps one batch of class probabilities and adds no backbone passes.

For each image, sum the other images’ LAME probabilities, weighted by `exp(-distance / 12)` in stream positions. Exclude the image’s own row and add 0.2 per class. Combine the log of these counts (weight 1/2), the image’s unsoftened log probabilities (2/3) and its LAME log probabilities (1/3). Double the combined logits and apply softmax to get the self-training target. Its argmax is the prediction.

We screened the changes on validation using the original order (seed 0), reverse order (offset 32, seed 1) and domain order 2–0–1 (offset 16, seed 0). This was a follow-up to stage 5, separate from the seven-stage selection above. In an official harness rerun, stage 5 reproduced its archived predictions exactly. The new method corrected 35 of its mistakes and introduced 7, reducing errors from 111 to 83. Errors fell from 45 to 33 for scan lines, 32 to 25 for wave warp and 34 to 25 for fisheye. The runs used a shared RTX 5090. All measurements for this experiment use validation data.

Across the 7 stream orders × 4 offsets × 2 seeds, the mean rose from **93.91% to 94.84%** (+0.93 points). The two seed means were 94.90% and 94.78%. On the 231-batch stream the new method scored 95.98% on the first 2,112 images and 98.20% on the last 2,112. The largest predicted class accounted for 5.26% of the final window, passing the collapse check. These runs reuse the same validation images. Reproduce them with `python tools/dev/repick.py method mixtta_local` after building the validation cache. [Per-order results](../out/mixtta_local/val_protocol.json)

We also shuffled the images independently within each validation domain to break up runs of similar classes while preserving the domain shifts. This control favored keeping the original source BN weight of 0.5. The control uses `numpy.random.default_rng(17)` to permute each domain in order and `fe.run` with adaptation seed 0.

| Source BN weight | Local correction | Official val (%) | Shuffled within domain (%) |
|---:|---|---:|---:|
| 0.5 | Off (stage 5) | 94.74 | 80.73 |
| 0.5 | On (`mixtta_local`) | **96.07** | **85.98** |
| 0.7 | Off | 95.45 | 56.39 |
| 0.7 | On | 96.02 | 62.83 |

After selecting the preset, we repeated the original-order validation run on RGB images re-encoded with Pillow at JPEG quality 35 (`fe.run`, seed 0). Stage 5 classified 1,856/2,112 correctly (87.88%). The local correction classified 1,896/2,112 correctly (89.77%).

**Design history.** The slide 21 FAQ rules out using observations from testing to design a method. The stage-5 settings were reselected on validation, but the experiment log says earlier offline test analysis influenced the candidate ideas. Both the stage-5 baseline and its 96.07% extension inherit that history. [Assignment slides](../CVPDL%202026%20hw2%20slides.pptx.pdf) · [Experiment log](../tools/dev/repick.py)

To measure the local correction independently of the shift detector and in-batch split, we enabled it on the earlier `mixtta_ms224` baseline with its fixed BN tracking rate of 0.4. This raised the official validation count from 1,985 to 2,009 correct out of 2,112 (+1.14 percentage points). The 56-run score above belongs to the stage-5 version. The fixed-tracking run used 4,148 MiB and took 15.1 s on a shared RTX 5090. The slides require the full stream to finish within 90 s on an RTX 3090 with at most 10 GB of VRAM. That target-hardware timing is unverified. [Fixed-tracking validation log](../out/mixtta_local_fixed/val_run_log.json)
