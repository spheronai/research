# Study 2 report, run two: does a camera move reader trained on more than one kind of motion beat Depth Anything 3 plus rules?

Spheron Research, run 24 to 25 Nov 2026 on one H100 PCIe on Spheron. Plan: `PLAN.md` (frozen before any rendering, hashes in `notes/plan.sha256`). Follows run one (`../2026-11-24-camera-move-reader/REPORT.md`). Every number below is in `results/*.csv` and `raw/results/`; every event and change is in `notes/ledger.md`.

## Answer in five lines

1. **No. None of the four training mixes met the pre-registered bar.** The mix picked on val (D-GFH: 2,000 grammar + 2,000 free-form + 2,000 human-path clips) read **fewer** held-out move pairs exactly than Depth Anything 3 (DA3-Base) plus naming rules: 36.8% and 36.6% (two seeds) against 50.4% on 500 clips, a paired difference of -13.6 points (95% CI -19.2 to -7.8).
2. **On 978 moving windows of real human camera paths (RealEstate10K test videos), it was also worse**: 55.2% and 56.9% exact against 69.5% for DA3 plus rules, paired -14.3 points (CI -18.4 to -10.4). Training on 6,000 human paths alone (D-H) stopped improving after its second epoch and scored 28.7% and 33.3%.
3. **The head still wins on motion like its training data.** On run one's held-out scenes it reached 85.9% and 86.6% against 61.8%, and on 500 clips with moving props 76.2% and 78.0% against 53.8%. It also fixes DA3's habit of reading a zoom as a dolly: 7.6% and 10.6% of zoom clips against 74.2%.
4. **Its corrected camera path is not better than DA3's.** On human paths its rotation error is 0.65° against 0.49° for DA3 alone, and the same naming rules applied to its path score within 4 points of DA3's path on every test set (66.9% against 69.5% on human paths). Re-shooting 200 clips with each path gives no arm a clear edge: D-GFH is 0.27 dB lower PSNR (CI -0.50 to -0.06) with no LPIPS difference.
5. **So the result is the one the plan named in advance.** Across runs one and two, **Depth Anything 3 plus a simple naming rule is the tool**; a learned namer on top of it fits the motion it trained on and does not generalise past it. We stop this line and do not publish the head's weights as a camera move reader.

## Main chart

![Exact move card, head minus DA3 plus rules, paired 95% CI, two seeds per training mix, across four test sets. On test_scenes and test_moving every mix except D-H is 18 to 26 points above zero. On test_combos every mix is below zero (D-GFH -13.6 and -13.8, D-G -4.6 and -3.8 with CIs crossing zero, D-F -15.8 and -20.0, D-H -49.0 and -48.4). On test_human moving windows every mix is 13 to 41 points below zero.](charts/exact-card-vs-B.svg)

Supporting chart, absolute exact card per split: `charts/exact-card-by-split.svg`.

## Pre-registered criteria

The mix with the best exact card on val (mean of two seeds) is judged. It counts as a success only if all three hold.

| Mix | val exact | S1: test_combos, head minus B (paired CI must be above 0) | S2: test_human moving, head minus B (CI above 0) | S3: test_scenes exact (at least 86.0%) | Pass |
|---|---|---|---|---|---|
| **D-GFH (picked)** | **74.3%** | -13.6 (-19.2 to -7.8); -13.8 (-19.4 to -7.8) | -14.3 (-18.4 to -10.4); -12.7 (-16.6 to -8.8) | 85.9%; 86.6% | **No** |
| D-G | 69.5% | -4.6 (-10.4 to +1.0); -3.8 (-9.2 to +1.8) | -35.0 (-39.0 to -31.1); -31.3 (-35.1 to -27.3) | 83.9%; 87.4% | No |
| D-F | 68.5% | -15.8 (-21.2 to -10.6); -20.0 (-25.4 to -14.4) | -31.0 (-34.8 to -27.1); -29.1 (-33.1 to -25.2) | 79.7%; 79.6% | No |
| D-H | 31.9% | -49.0 (-53.2 to -44.6); -48.4 (-53.0 to -43.8) | -40.8 (-44.8 to -37.1); -36.2 (-39.9 to -32.1) | 27.7%; 27.2% | No |

Points of exact card, seed 0; seed 1. B (DA3 plus rules) scored 62.8% on val. Source: `results/criteria.csv`.

## Results table

Exact move card (all seven axis directions plus the orbit flag right), %, with bootstrap 95% CI for single rows. D rows show seed 0; seed 1.

| Arm | test_scenes (1,000) | test_combos (500) | test_human moving (978) | test_moving (500) |
|---|---|---|---|---|
| Static floor | 5.9 | 0.0 | 0.0 | 9.8 |
| A1: COLMAP, per-frame focal | not run | 21.8 (18.2 to 25.4), 26% failed | 10.3 (8.4 to 12.2), 12% failed | not run |
| A2: COLMAP, shared focal | not run | 33.8 (29.6 to 37.8), 43% failed | 15.4 (13.3 to 17.7), 25% failed | not run |
| **B: DA3 plus rules** | **61.8 (58.6 to 64.8)** | **50.4 (46.4 to 55.0)** | **69.5 (66.4 to 72.3)** | **53.8 (49.4 to 57.8)** |
| C: Qwen3-VL-8B | not run | not run | 6.0 (4.6 to 7.6) | 7.0 (4.8 to 9.4) |
| D-G | 83.9; 87.4 | 45.8; 46.6 | 34.6; 38.2 | 72.2; 76.8 |
| D-F | 79.7; 79.6 | 34.6; 30.4 | 38.5; 40.4 | 74.4; 72.6 |
| D-H | 27.7; 27.2 | 1.4; 2.0 | 28.7; 33.3 | 24.0; 24.0 |
| **D-GFH** | **85.9; 86.6** | **36.8; 36.6** | **55.2; 56.9** | **76.2; 78.0** |

Other metrics for B and the picked mix (D-GFH seed 0; seed 1):

| Metric | Split | B | D-GFH |
|---|---|---|---|
| Rotation error, degrees | test_human moving | 0.49 | 0.65; 0.67 |
| Rotation error, degrees | test_combos | 2.09 | 2.07; 2.08 |
| Direction accuracy, per axis | test_human moving | 94.8% | 91.5%; 92.3% |
| Direction accuracy, per axis | test_combos | 85.6% | 88.3%; 88.3% |
| Zoom read as dolly | test_scenes | 74.2% | 7.6%; 10.6% |
| Zoom read as dolly | test_combos | 83.2% | 43.2%; 36.8% |
| Handheld flag right | test_human moving | 58.5% | 78.3%; 77.3% |
| Handheld flag right | test_combos | 81.8% | 64.0%; 68.2% |

Every arm, metric and split, with intervals: `results/arms.csv`. Paired differences: `results/paired_vs_B.csv`. Per move type: `results/by_type.csv`.

**Where the head loses on held-out pairs.** On test_combos, by move type in the true card (D-GFH seed 0 against B): orbit 11% vs 54%, tilt 19% vs 53%, truck 24% vs 52%, roll 57% vs 81%, dolly 55% vs 78%. It wins on zoom (36% vs 12%) and pedestal (42% vs 31%), and ties on pan (47% vs 48%). Run one showed the same orbit failure (22% vs 55%); a wider training motion did not fix it.

**The path is not the problem being fixed.** Naming the head's corrected path with the same rules B uses gives 66.9% and 66.6% on test_human moving (B 69.5%), 46.2% and 47.4% on test_combos (B 50.4%), 62.8% and 62.8% on test_scenes (B 61.8%) and 54.2% and 54.8% on test_moving (B 53.8%). Any gain the head shows comes from its naming output, not from a better camera path.

## Results: re-shoot test (200 test_scenes clips)

Each arm's camera path re-rendered in the original scene and compared with the original frames. GT (true path re-rendered) is the noise floor; STATIC (camera held at frame 0) is the do-nothing floor. Seed 0 for each D mix.

| Row | PSNR | SSIM | LPIPS (lower is better) |
|---|---|---|---|
| GT | 65.06 | 1.000 | 0.000 |
| STATIC | 21.63 | 0.517 | 0.305 |
| B | 22.78 | 0.573 | 0.159 |
| D-G | 22.45 | 0.556 | 0.152 |
| D-F | 22.36 | 0.557 | 0.161 |
| D-H | 22.75 | 0.572 | 0.160 |
| D-GFH | 22.51 | 0.565 | 0.161 |

Paired against B: D-G is slightly better on LPIPS (-0.0071, CI -0.0131 to -0.0010) and worse on PSNR (-0.32 dB, CI -0.59 to -0.07) and SSIM (-0.017). D-GFH is worse on PSNR (-0.27 dB, CI -0.50 to -0.06) and SSIM (-0.008), with no LPIPS difference (+0.0019, CI -0.0031 to +0.0069). D-H, which barely trained, matches B. As in run one, there is no clear winner, and all arms sit far closer to the do-nothing floor than to the noise floor.

## What we did

**Data.** Every clip is a Blender 4.5.17 Cycles render (392 x 224, 16 frames over 4 seconds) of a procedural Poly Haven scene, as in run one. Three sources of camera motion, all filtered so no training clip contains any of the five held-out pairs (tilt+truck, pan+pedestal, zoom+pedestal, roll+dolly, orbit+tilt) as judged by the labeller on the path as placed:

| Source | What it is |
|---|---|
| G | Run one's move grammar (nine primitives, one to three per clip), run one's seeds |
| F | Free-form: each of seven channels (yaw, pitch, roll, forward, right, up, log focal) switched on with p = 0.35, each with its own size and easing, optional look-at, handheld shake on 30% |
| H | Human camera paths: 4-second windows of RealEstate10K camera poses (CC BY 4.0) scaled to the scene, plus Big Buck Bunny's shot cameras (scenes 01 to 07 for training, 08 to 13 for test) |

In 30% of training and val clips, one to three props move during the shot.

| Split | Clips |
|---|---|
| train_G, train_F, train_H | 6,000 each |
| val | 600 (200 per source) |
| test_scenes, test_combos | 1,000 and 500, run one's clips unchanged |
| test_human | 1,013 (1,000 RealEstate10K test-list windows, 13 Big Buck Bunny), 978 of them moving |
| test_moving | 500, grammar and free-form moves with moving props in every clip |

Leak check **passed**: 0 assets shared between training and test, no held-out pair in train or val, a held-out pair in every test_combos clip, no repeated first frame, and 0 of 3,744 training-side human-path videos on the test side (`raw/results/leak_check.txt`). Determinism check passed: 20 test_scenes clips re-rendered with scene specs and cameras identical to run one and 65.2 dB PSNR between two renders.

**Arms.** Same as run one: static floor, COLMAP 4.2 (pycolmap 4.2.1) with per-frame and shared focal, DA3-Base (commit `3d835ec`) plus rules, Qwen3-VL-8B zero-shot, and our head: a 6-layer transformer on frozen DA3-Base features and cameras that outputs a path correction and a move card. Four training mixes (D-G, D-F, D-H, D-GFH), 6,000 clips each, two seeds, 40 epochs, checkpoint with the best val loss kept. Handheld thresholds for rule-named arms are calibrated on val (B: 0.121°).

**Software.** torch 2.8.0+cu128, xformers 0.0.32.post2, Blender 4.5.17 LTS, LPIPS v0.1 (AlexNet). Full lists in `raw/env/`. Code in `harness/`; the run order is `setup2.sh`, `pilot2.sh`, then `run_render2.sh`, `baselines2.sh` and `final2.sh` in parallel.

## Compute

| Step | Wall time (UTC) |
|---|---|
| Setup, pools, pilots, freeze | 21:12 to 21:29 |
| Render 21,757 clips (four Blender processes) | 21:29 to 07:02 |
| DA3 on every clip, alongside rendering | to 07:02 |
| COLMAP (test_combos, test_human, val), Qwen3-VL | to 23:42 and 22:30 |
| Train 8 heads, four at a time (about 25 min each) | 07:03 to 07:54 |
| Predict, leak check, score | 07:54 to 07:58 |
| Re-shoot (200 clips x 7 rows) and scoring | 07:58 to 08:21 |
| Copy off, terminate | 08:21 to 08:25 |

One H100 PCIe 80 GB for the whole run. Terminated 08:25:33 UTC 25 Nov and confirmed terminated.

## Changes from the plan

Listed in full with times in `notes/ledger.md`. The ones that touch results:

- **Source F lost its start/stop times (pilot, before the freeze).** With each channel starting and stopping mid-clip, 84.8% of 400 shake-free F paths were labelled handheld, because the labeller's degree-5 fit cannot follow a move that starts mid-clip. Every channel now runs the full 4 seconds with its own easing. F is less varied than planned.
- **Human-path pools sampled differently (before any rendering).** One random clip per video gave only 296 test windows, because about half of RealEstate10K clips are under 4 seconds. Changed to clips of at least 4 seconds, round-robin over videos, at most two per video.
- **The render cut rule was not applied.** The plan says to cut each training source to 4,000 clips if rendering projects past 8.5 h. It ran about 9.5 h, but this was noticed only when 17,195 of 18,000 training clips were already rendered; cutting then would only have dropped data. Training stayed at 6,000 clips per mix.
- **D-H collapsed.** Both seeds reached their best val loss in the second of 40 epochs (11.4 and 11.5, against 3.8 to 5.1 for the other mixes), so the kept checkpoints are close to untrained. Kept as the plan requires; D-H was not the mix picked on val.
- **D-G did not reproduce run one's head.** It trains on the same 6,000 grammar seeds, yet scored 83.9% and 87.4% on test_scenes (run one 89.0% and 89.4%) and 45.8% and 46.6% on test_combos (run one 53.6% and 50.8%). What changed: the held-out-pair filter now applies to the placed path's card, 30% of training clips have moving props, and the checkpoint is picked on a val set that mixes all three sources. Which one causes the drop was not isolated.
- **Re-shoot has 7 rows, not 6**: one per training mix. No COLMAP rows, because COLMAP did not run on test_scenes in run two.
- **Human-path timing.** Windows are 16 samples evenly over 4.0 seconds (the grammar's timing); run one's 17 Big Buck Bunny windows used 0.25-second steps over 3.75 seconds. No run-one number changes.
- **val's human-path clips come from RealEstate10K only**, held out of train_H by video id; no Big Buck Bunny in val.
- **Phone clips were not shot**, as decided before the run. The gap to real footage with ground truth is still unmeasured.

## Limits

- **Real footage with ground truth is still missing.** test_human uses real camera paths, but replayed in rendered scenes. The image is still a Cycles render.
- **The human paths are mostly one genre.** 973 of 978 moving test_human windows are RealEstate10K: slow walk-throughs of homes with a fixed focal. Zoom metrics mean nothing there (zoom F1 is about 0.33 for every arm), and only 5 moving windows are Big Buck Bunny.
- **Labels come from thresholds.** Directions, sizes and handheld depend on the labeller's fixed thresholds (3° rotation, 0.03 of scene depth, 0.05 log focal). A different threshold changes every arm's score.
- **Scale.** Translation is in units of scene depth, and the re-shoot borrows the true depth for every arm.
- **test_moving is close to the training distribution** (grammar and free-form moves, moving props seen in training), so the head's +22 to +24 point lead there says it learned to ignore moving props in motion it knows, not that it generalises.

## What this means for publishing

- **Publish runs one and two together as one negative result.** DA3 plus rules is the tool: Apache-2.0 end to end, never failed a clip, and no head beat it on human camera paths or held-out move pairs. The learned namer helps only on motion like its training data.
- **Do not publish the head weights** as a camera move reader.
- **Publish the benchmark.** Scene generator, three motion sources, labeller, leak check, the re-shoot test with its two floors, and these results. The human-path test set (1,013 windows with exact cameras) is new.
- **One finding is worth a short note on its own:** DA3 reads a pure zoom as a dolly on 74% to 83% of zoom clips, and the D-GFH head on top cuts that to 8% to 43%. A tool that wants to name zooms could use the head for that one flag and DA3 plus rules for everything else. Not tested as a combined system here.

## Raw data

- Public repo folder: https://github.com/spheronai/research/tree/main/2026-11-25-camera-move-reader-2
- `raw/` (about 1.8 GB, mostly logs) is not stored in the repository. Its clip specs, exact cameras, ground-truth move cards and per-clip scores are published as Parquet in one archive (about 49 MB, configs `clips`, `predictions` and `run1`, dataset card in `README.md`). Download: https://huggingface.co/datasets/spheron/camera-move-cards (Hugging Face), or the same files as one archive: https://github.com/spheronai/research/releases/download/2026.11.25/study2-camera-move-reader-dataset.tar.gz. Model weights, logs and frames are not published.
- Column definitions: `DATA.md`.

## Files

- `PLAN.md`: frozen plan. `notes/plan.sha256`: hashes at the freeze (43 files).
- `notes/ledger.md`: run log, every event and change with times.
- `harness/`: all code. `analyze.py` rebuilds `results/` and `charts/` from `raw/`.
- `results/*.csv`: arms, paired differences, per move type, training, criteria, re-shoot.
- `charts/`: `exact-card-vs-B` (main) and `exact-card-by-split`, SVG and PNG. Drawn in the Spheron chart style (`harness/spheron_chart_style.py`).
- `raw/results/`: every metric with intervals, per-clip scores, leak check, re-shoot.
- `raw/runs/<mix>_s<seed>/`: weights (`model.pt`), training log, predictions on every split. `raw/runs/vlm/`: the VLM's raw answers.
- `raw/data/`: scene spec, move spec and exact cameras for all 21,717 clips. Frames were not kept; `harness/render_clips2.py` rebuilds any clip from its spec.
- `raw/pools/`: human-path window pools. `raw/env/`: GPU and package versions. `raw/logs/`: all logs.

## How to cite

```
Spheron Research (2026). Does a camera move reader trained on more than one kind of motion beat Depth Anything 3 plus rules? Measured 24-25 Nov 2026, published 25 Nov 2026, version 1.
https://www.spheron.network/research/camera-move-reader-vs-depth-anything-3/
Raw data: https://github.com/spheronai/research/tree/main/2026-11-25-camera-move-reader-2
```

```bibtex
@misc{spheron2026cameramovereader2,
  author       = {{Spheron Research}},
  title        = {Does a camera move reader trained on more than one kind of motion beat Depth Anything 3 plus rules?},
  year         = {2026},
  note         = {Measured 24-25 Nov 2026, published 25 Nov 2026, version 1},
  howpublished = {\url{https://www.spheron.network/research/camera-move-reader-vs-depth-anything-3/}},
  url          = {https://github.com/spheronai/research/tree/main/2026-11-25-camera-move-reader-2}
}
```

