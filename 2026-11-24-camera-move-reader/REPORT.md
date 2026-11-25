# Study 2 report: Can a small model read a camera move well enough to re-shoot it?

Spheron Research, run 24 Nov 2026 on one H100 PCIe on Spheron. Plan: `PLAN.md` (frozen 17:37 UTC, hashes in `notes/plan.sha256`). Every number below is in `raw/results/`; every event and change is in `notes/ledger.md`.

## Answer in five lines

1. **On held-out scenes with moves of the kinds it trained on, a small head on frozen Depth Anything 3 reads moves much better than anything else tested**: the full move card was exactly right on 89.0% of 1,000 clips (95% CI 87.0 to 91.0), against 62.0% (58.8 to 65.0) for Depth Anything 3 plus naming rules, the best baseline. It calls a pure zoom a dolly on 5% to 15% of zoom clips (two seeds) where Depth Anything 3 plus rules does so on 75%.
2. **That gain does not carry to move pairs it never saw.** On 500 clips built around five held-out pairs, exact cards were 53.6% (49.2 to 57.8) against 51.0% (46.8 to 55.4): no measurable difference. On clips with an orbit (the orbit plus tilt pair) it is worse than its own input, 22% against 55%.
3. **On camera moves keyed by a human artist (Big Buck Bunny's shot cameras replayed in our scenes), there is no measurable gain either**: 71% and 65% (two seeds) against 77% exact on the 17 windows that move. Intervals are wide.
4. **Re-shooting with the predicted camera, there is no clear winner** between the head and Depth Anything 3 alone. The head is slightly better on LPIPS and slightly worse on PSNR and SSIM, all by small amounts.
5. **The off-the-shelf alternatives are weak at this job.** Qwen3-VL-8B scores below a "nothing moved" answer on per-axis direction. COLMAP fails outright on 39% to 53% of 4-second clips.

So the honest result is the one the plan named in advance: on the evidence here, **Depth Anything 3 plus a simple naming rule is the tool to ship today**. Our trained head learns the move grammar we rendered and does not yet generalise past it. We should not publish its weights as a "camera move reader for filmmakers" yet.

## Main chart

![Exact move card with 95% CI for every arm on three test sets. Unseen scenes: our head 89.0% and 89.4% (two seeds), Depth Anything 3 plus rules 62.0%, COLMAP 20.0% and 20.4%, Qwen3-VL-8B 6.5%, static floor 5.9%. Unseen move pairs: head 53.6% and 50.8%, DA3 plus rules 51.0%, COLMAP 21.4% and 32.0%, Qwen3-VL 0%. Big Buck Bunny moving windows: head 70.6% and 64.7%, DA3 plus rules 76.5%, COLMAP 0% and 11.8%, Qwen3-VL 0%.](charts/exact-card-by-arm.svg)

Supporting chart, how often a pure zoom is reported as a dolly: `charts/zoom-read-as-dolly.svg` (head 15% and 5% on unseen scenes against 75% for DA3 plus rules).

## What was tested

**Data.** Every clip is a Blender 4.5.17 render (Cycles, 24 samples, GPU denoise, 392 x 224, 16 frames from a 4-second move) of a procedural scene: 10 to 28 CC0 Poly Haven props on a textured ground under an HDRI sky. Camera moves come from a grammar of nine primitives (pan, tilt, roll, dolly, truck, pedestal, zoom, orbit, dolly zoom), one to three per clip, with easing and handheld shake on 30%. A rule labeller turns any camera path into a move card: per axis a direction and a size, plus orbit, dolly zoom, handheld and easing.

| Split | Clips | Held out |
|---|---|---|
| train | 6,000 | nothing |
| val | 500 | early stopping, and the per-arm handheld threshold |
| test_scenes | 1,000 | 15% of models, skies and textures never seen in training |
| test_combos | 500 | the same unseen assets, and five move pairs never seen in training: tilt+truck, pan+pedestal, zoom+pedestal, roll+dolly, orbit+tilt |
| human_moves | 62 | Big Buck Bunny's own shot cameras (4-second windows) replayed in test scenes |
| real_tos | 13 | live-action Tears of Steel shots, no ground truth |

The leak check passed. Train and val share none of their 279 and 263 assets with the 41 test assets, no train or val clip contains a held-out pair, every test_combos clip contains one, and no first frame repeats across splits (`raw/results/leak_check.txt`).

**Arms.**

| Arm | What it is |
|---|---|
| REF | Reference floor: the camera never moves, the card says "static" |
| A1 | COLMAP 4.2 (pycolmap), SIFT, exhaustive matching, one camera per frame so focal can change, 3 seeds per clip; fails unless all 16 frames register |
| A2 | Same, one shared camera (fixed focal), the usual default |
| B | Depth Anything 3, DA3-Base (Apache-2.0), feed-forward poses and focal, named by the rule labeller |
| C | Qwen3-VL-8B-Instruct (Apache-2.0), zero-shot, greedy, one fixed prompt defining every term |
| D | Ours: a 6-layer transformer (width 256, 20.7 MB of weights) on frozen DA3-Base. Input: DA3's per-frame camera and focal plus DA3 features. Output: a correction to the camera path and the move card. Two seeds. |

Rule-named arms (A1, A2, B, and D "via rules") each get one handheld threshold, the value that maximises handheld accuracy on val for that arm. Without it, every noisy predicted path reads as handheld.

## Results: unseen scenes (test_scenes, 1,000 clips)

| Arm | Fail | Rotation error (deg) | ATE (depth units) | Zoom error | Direction acc. | Zoom F1 | Exact card | Handheld acc. |
|---|---|---|---|---|---|---|---|---|
| REF static | 0% | 7.09 | 0.111 | 0.058 | 72.2% | 0.27 | 5.9% | 70.8% |
| A1 COLMAP per-frame focal | 39.0% | 6.94 | 0.037 | 0.930 | 72.1% | 0.49 | 20.0% | 69.6% |
| A2 COLMAP shared focal | 53.1% | 4.63 | 0.016 | 0.052 | 75.9% | 0.27 | 20.4% | 80.9% |
| B DA3-Base + rules | 0% | 1.46 (1.37 to 1.56) | 0.012 | 0.038 | 90.0% (88.9 to 90.9) | 0.84 | 62.0% (58.8 to 65.0) | **90.4%** |
| C Qwen3-VL-8B | 0% | n/a | n/a | n/a | 63.5% | 0.37 | 6.5% | 70.7% |
| D ours, seed 0 | 0% | **1.14** (1.08 to 1.20) | 0.013 | **0.033** | **98.1%** (97.8 to 98.5) | **0.97** | **89.0%** (87.0 to 91.0) | 81.3% |
| D ours, seed 1 | 0% | 1.15 (1.09 to 1.23) | 0.013 | 0.031 | 98.2% | 0.96 | 89.4% (87.5 to 91.2) | 82.1% |
| D seed 0, named by rules | 0% | 1.14 | 0.013 | 0.033 | 91.8% | 0.83 | 61.4% | 89.9% |

Brackets are 95% bootstrap intervals over clips (1,000 resamples).

How to read it:

- **Direction accuracy** is per axis over 7 axes, so "no motion" is the most common answer. That is why the do-nothing floor scores 72%. Read every row against it.
- **Rotation error and ATE** are on the clips an arm did not fail. COLMAP's path numbers therefore describe only the easier clips it survived. ATE is in units of median scene depth after a similarity alignment, and only on clips where the true camera travels more than 0.03 of that depth.
- **Zoom error** is the mean absolute error of the per-frame log focal ratio to frame 0. The per-frame-focal COLMAP variant often returns absurd focal lengths, and 3.7% of its clips returned a focal of zero or below. Those count as failures.
- **Failures and the move card.** A failed clip counts as "no move" for per-axis accuracy and as wrong for the exact card. A tool that returns nothing has not named the move.

**Zoom vs dolly, the problem this study set out to fix.** On clips with a zoom, the true zoom direction was found 94.2% of the time by D (both seeds), 83.3% by B, 42.6% by A1, 27.1% by C and never by A2, which cannot represent a zoom. On zoom clips with no dolly, a dolly was wrongly reported on **15.2% and 5.3%** of clips by the two D seeds, against **75.0%** for B, 80.3% for C and 32.6% for A1.

The head reaches that through its direct card output. Run through the rule labeller instead, its own corrected path still shows a false dolly on 56.8% of those clips. The correction lives in the classifier, not in the path.

**By move type (exact card, D seed 0 vs B):** dolly 86% vs 62%, dolly zoom 86% vs 53%, orbit 88% vs 60%, pan 86% vs 55%, pedestal 87% vs 62%, roll 91% vs 61%, tilt 81% vs 49%, truck 86% vs 63%, zoom 78% vs 22%, static 100% vs 100%. Handheld clips: 87% vs 58%.

**Where the head is worse: handheld.** Its direct handheld output is right 81% to 82% of the time, against 90% for Depth Anything 3 with a calibrated threshold. Naming handheld from the path with that threshold (the "named by rules" row) is the better choice today.

## Results: unseen move pairs (test_combos, 500 clips)

| Arm | Fail | Rotation error (deg) | Direction acc. | Zoom F1 | Exact card | Handheld acc. |
|---|---|---|---|---|---|---|
| REF static | 0% | 9.27 | 64.4% | 0.29 | 0.0% | 70.8% |
| A1 COLMAP per-frame focal | 25.6% | 6.49 | 67.4% | 0.43 | 21.4% | 70.6% |
| A2 COLMAP shared focal | 43.2% | 3.48 | 75.5% | 0.29 | 32.0% | 83.8% |
| B DA3-Base + rules | 0% | 2.09 (1.84 to 2.32) | 85.7% (84.1 to 87.2) | 0.75 | 51.0% (46.8 to 55.4) | **84.6%** |
| C Qwen3-VL-8B | 0% | n/a | 54.0% | 0.33 | 0.0% | 71.0% |
| D ours, seed 0 | 0% | 2.08 (1.93 to 2.25) | **92.1%** (91.1 to 92.9) | **0.93** | 53.6% (49.2 to 57.8) | 71.8% |
| D ours, seed 1 | 0% | 2.06 (1.90 to 2.23) | 91.5% | 0.90 | 50.8% (46.4 to 55.2) | 69.8% |

What happens on pairs it never saw:

- Per-axis direction and zoom still improve over B.
- The full card does not: 53.6% and 50.8% against 51.0%, inside each other's intervals.
- Path rotation error is the same as B's.
- Handheld detection drops to the do-nothing floor.

Breaking it down by the primitive a clip contains (each held-out pair contributes most of these clips), exact card for D seed 0 vs B:

| Clips containing | n | D seed 0 | B |
|---|---|---|---|
| zoom (mostly zoom+pedestal) | 106 | **48%** | 12% |
| pan (mostly pan+pedestal) | 108 | **67%** | 49% |
| roll (mostly roll+dolly) | 120 | 74% | 82% |
| truck (mostly tilt+truck) | 100 | 49% | 52% |
| orbit (orbit+tilt) | 93 | 22% | **55%** |

On orbit clips the head's camera path is also worse than the DA3 path it starts from: rotation error 3.9 against 3.0 degrees.

The head keeps its zoom advantage on new pairs and loses ground on orbits. It has learned the combinations in the training grammar rather than the motion underneath them.

## Results: re-shoot test (200 test_scenes clips)

Each arm's predicted move was replayed in the original scene and rendered:

- Anchored at the true first camera.
- Translation rescaled to the true scene depth.
- Focal length following the predicted zoom profile.
- A failed clip re-shot locked off.

Frames 1 to 15 were compared with the original.

| Re-shoot | PSNR (dB) | SSIM | LPIPS (lower is better) |
|---|---|---|---|
| True path, re-rendered (noise floor) | 65.07 | 1.000 | 0.000 |
| Locked off (do nothing) | 21.63 | 0.517 | 0.305 |
| A1 COLMAP per-frame focal (75 of 200 failed) | 21.80 | 0.531 | 0.314 |
| A2 COLMAP shared focal (107 of 200 failed) | 22.52 | 0.542 | 0.274 |
| B DA3-Base | **22.76** | **0.572** | 0.159 |
| D ours, seed 0 | 22.45 | 0.557 | **0.147** |

Paired differences, D minus B, over the same 200 clips (95% bootstrap):

| Metric | D minus B | 95% interval |
|---|---|---|
| PSNR | −0.31 dB | −0.56 to −0.09 |
| SSIM | −0.016 | −0.025 to −0.007 |
| LPIPS | −0.012 | −0.018 to −0.005 |

All three are measurable and small, and they point different ways: lower LPIPS is better, lower PSNR and SSIM are worse. No clear winner. B against the locked-off camera: LPIPS −0.147 (−0.163 to −0.129).

**PSNR is a poor judge here.** A camera that is right but off by a fraction of a degree misaligns every textured pixel, so even good re-shoots land near the do-nothing floor. LPIPS separates working methods from failing ones. `raw/results/reshoot_zoom_examples.png` shows three zoom clips: original, DA3 and ours at frames 0, 7 and 15. Both methods drift; in the third clip both overshoot the move.

## Results: human-keyed camera moves (Big Buck Bunny)

This test was **added after the main test results were known**, so treat it as exploratory. Its window rule was fixed before any of its clips were rendered.

**Source.** The Big Buck Bunny scene files (download.blender.org/demo/bbb/blender.zip, CC BY 3.0, Blender Foundation | www.blender.org) still carry the film's camera animation. Matching scene frames to film frames would need the edit list, so the motion was replayed in held-out test scenes instead. This tests the moves a human layout artist made. It does not test the look of real footage.

**Selection.** 142 scene files gave 62 four-second windows: one perspective camera, and at least 34 of 336 depth rays hitting geometry, the same rule as the main renders. 530 candidate windows were dropped for too few depth hits, because many linked set files are missing. The labeller calls 45 of the 62 windows static: over four seconds, most of the film's camera work stays under its 3 degree and 0.03 depth thresholds.

| Arm | Exact card, all 62 | Exact card, 17 moving windows | Rotation error, moving (deg) |
|---|---|---|---|
| REF static | 72.6% | 0% | n/a |
| A1 COLMAP per-frame focal | 0.0% (71% failed) | 0% (12% failed) | 3.17 |
| A2 COLMAP shared focal | 3.2% (81% failed) | 11.8% (29% failed) | 1.72 |
| B DA3-Base + rules | 90.3% | **76.5%** (58.8 to 94.1) | 0.47 |
| C Qwen3-VL-8B | 67.7% | 0% | n/a |
| D ours, seed 0 | **91.9%** | 70.6% (47.1 to 88.2) | 0.46 |
| D ours, seed 1 | 88.7% | 64.7% (41.2 to 88.2) | 0.44 |

What it shows:

- On static windows COLMAP fails almost every time, because there is no parallax to start a reconstruction.
- On the moving windows, mostly dollies, the head is no better than Depth Anything 3 alone.
- With 17 clips the intervals are wide.

## Results: real footage (Tears of Steel, qualitative)

13 live-action shots from Tears of Steel (720p, CC BY 3.0, (CC) Blender Foundation | mango.blender.org) were cut with ffmpeg scene detection and run through B, C and D (`raw/results/real_tos_cards.json`). There is no ground truth. The three arms agree on one shot of 13 (shot 1, a near-static two-shot, all "static"). On the rest they disagree on direction or on the kind of move.

That fits the synthetic-to-real gap the plan listed as a risk. The study does not measure accuracy on real footage. The actors in Tears of Steel keep their image rights, so these frames are not for promotional use.

## Exports

`harness/export_camera.py` writes a predicted path in four formats, each checked against ground truth on a test clip (largest absolute matrix error):

| Format | What reads it | Round-trip error |
|---|---|---|
| RealEstate10K-style `.txt` | Camera-control video models (CameraCtrl, Wan Fun Camera) | 1.7e-8 |
| `.usda` | USD tools | 5.0e-8 |
| Blender `.py` | Blender (keyframed camera) | 2.3e-7 |
| Nuke `.chan` | Nuke, rotation order ZXY | 5.9e-16 (Euler decomposition, 1,000 random rotations) |

These work today with any arm's path, including Depth Anything 3's.

## Compute

| Step | Wall time (UTC) |
|---|---|
| Setup, assets, pilot | 17:10 to 17:35 |
| Render 8,000 clips (four Blender processes on one GPU, about 0.85 s per clip alone) | 17:20 to 20:00 |
| Baselines (COLMAP on CPU in parallel, DA3, Qwen3-VL) | 17:41 to 20:01 |
| Train two seeds in parallel (40 epochs, 6,000 clips, about 14 min each) | 20:01 to 20:15 |
| Predict, score, leak check | 20:15 to 20:17 |
| Re-shoot renders and scoring | 20:17 to 20:37 |
| Human-move test, copy off, terminate | 20:20 to 20:40 |

One H100 PCIe 80 GB for the whole run. The machine was terminated at about 20:40 UTC and confirmed terminated.

## Changes from the plan

Listed in full in `notes/ledger.md`. The ones that touch results:

**Bug fix after rendering started.** `moves.py` could not sample a held-out pair when it drew an empty or one-primitive move, so test_combos first rendered 6 of 500 clips. Fixed so a held-out clip always gets two or more primitives. Every other split is unchanged, and test_combos was re-rendered.

**Scorer rules added on val before any test scoring:**
- A camera no lens can have (focal of zero or below, or non-finite values) counts as a failure.
- A "static" reference row was added.
- A per-move-type breakdown was added. The plan already listed it.

**Checks dropped or changed:**
- The ground-truth real-camera check, as planned, was not possible: no free production files are on the Big Buck Bunny, Sintel or Tears of Steel download pages.
- It was replaced by two weaker checks, the human-move replay and the qualitative Tears of Steel shots. The first was designed after the main results were seen.

**Training detail.** Validation loss bottomed near epoch 20 for both seeds and rose after (seed 0: 2.95 at epoch 20, 3.21 at 39). The checkpoint with the best validation loss was kept, as planned.

## Limits

- **Static scenes.** Props never move, so nothing teaches the model to ignore a moving subject. Real shots are full of them.
- **One renderer, one look.** Every test clip is a Cycles render of the same kind of procedural scene. The human-move test changes the motion, not the image.
- **Labels come from thresholds.** "Static", sizes and handheld depend on the labeller's fixed thresholds (3 degrees of rotation, 0.03 of scene depth, 0.05 log focal). A different threshold changes every arm's exact-card score.
- **Small human-move sample.** 17 moving windows give intervals about 40 points wide.
- **Scale.** Translation is in units of scene depth. The re-shoot borrows the true depth for every arm, so it does not test scale recovery.

## What this means for publishing

- **Ready now: the benchmark and harness.** Scene generator, move grammar, labeller, leak check, the re-shoot test with its noise floor and do-nothing floor, the exporters, and these results, including the negative ones. Nothing comparable was found in the prior-art check (`notes/prior_art.md`). The re-shoot test in particular is new.
- **Ready now: DA3 plus rules as a working tool.** It is Apache-2.0 end to end. It never failed a clip here, beats COLMAP and the VLM by a wide margin, and is the most reliable namer on human-keyed moves and on handheld.
- **Not ready: our head as a model release.** Its gains are real on the training grammar but vanish on new move pairs and human moves.

The next run would have to fix that, and it is a bigger job than this one:

- A grammar that samples any combination of moves.
- Human-keyed camera paths in training as well as test.
- Moving subjects.
- A real-footage test set with ground truth. Blender Studio's subscriber Vault holds shot files for several films under CC BY 4.0, behind a paid subscription.

## Files

- `DATA.md`: column definitions for every CSV.
- `results/*.csv`: arms, per move type, training, re-shoot and paired re-shoot differences, built by `harness/analyze.py` from `raw/`.
- `charts/`: `exact-card-by-arm` (main) and `zoom-read-as-dolly`, SVG and PNG.
- `PLAN.md`: frozen plan. `notes/plan.sha256`: hashes at the freeze.
- `notes/ledger.md`: run log, every event and change with times.
- `notes/prior_art.md`: prior-art check.
- `harness/`: all code. Rendering, labeller, baselines, training, scoring, re-shoot, exporters, human-move and real-shot scripts.
- `raw/` is not stored in this repository (about 2.3 GB, mostly logs). Download: <raw data link>. It holds:
- `raw/results/`:
  - `results_*.json`: every metric with intervals and by-type breakdowns.
  - `reshoot*.json`: re-shoot scores, per clip and paired.
  - `leak_check.txt`.
  - `real_tos_cards.json`.
  - `reshoot_zoom_examples.png`.
- `raw/runs/seed0`, `raw/runs/seed1`: model weights (`model.pt`; SHA-256 `cef5604f...` and `e8097494...`), training logs, predictions on every split. `raw/runs/vlm`: the VLM's raw answers.
- `raw/data/*/*/meta.json`: scene spec, move spec and exact cameras for every clip (8,093 files). The 31 GB of rendered frames were not kept; `harness/render_clips.py` rebuilds any clip from its seed and spec, and a re-render of the true path matched the original at 65 dB.
- `raw/logs/`: render, baseline, training and pipeline logs.

## How to cite

```
Spheron Research (2026). Can a small model read a camera move well enough to re-shoot it? Measured 24 Nov 2026, published <publish date>, version 1.
https://www.spheron.network/research/<slug>/
Raw data: https://github.com/spheronai/gpu-benchmarks/tree/main/research/2026-11-24-camera-move-reader
```

```bibtex
@misc{spheron2026cameramovereader1,
  author       = {{Spheron Research}},
  title        = {Can a small model read a camera move well enough to re-shoot it?},
  year         = {2026},
  note         = {Measured 24 Nov 2026, published <publish date>, version 1},
  howpublished = {\url{https://www.spheron.network/research/<slug>/}},
  url          = {https://github.com/spheronai/gpu-benchmarks/tree/main/research/2026-11-24-camera-move-reader}
}
```

`<slug>` and `<publish date>` are filled in when the landing-site page is built.
