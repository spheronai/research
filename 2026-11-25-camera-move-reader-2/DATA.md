# Data for study 2, run two: camera move reader with three training sources

Every CSV in `results/` is produced by `harness/analyze.py` from the files in `raw/`. Rerun it from this folder with:

```bash
uv run -q --with matplotlib --with numpy python harness/analyze.py
```

It reads `raw/results/results_*.json`, `raw/results/reshoot.json` and `raw/runs/D*/train_log.json`, and writes `results/*.csv` and `charts/*.{svg,png}`.

Charts use the Spheron chart style in `harness/spheron_chart_style.py`. Set `SPHERON_FONT_DIR` to a folder holding `SFPRODISPLAYREGULAR.otf` and `OCRAStd.otf` to render them in the brand faces; without it they fall back to system fonts.

## Arms

| Arm id | What it is |
|---|---|
| `REF_always_static` | Reference floor: the camera never moves, the card says "static" |
| `A1_colmap_per_image_focal` | COLMAP 4.2 (pycolmap), SIFT, exhaustive matching, one camera per frame, 3 seeds per clip; a clip fails unless all 16 frames register |
| `A2_colmap_shared_focal` | Same, one shared camera (fixed focal) |
| `B_da3_base_rules` | Depth Anything 3 (DA3-Base, commit `3d835ec`), feed-forward poses and focal, named by the rule labeller |
| `C_qwen3_vl_8b` | Qwen3-VL-8B-Instruct, zero-shot, greedy, run one's prompt. Names only, no path |
| `D_ours_<mix>_<seed>` | The trained head on frozen DA3-Base, names from its own card output. Mixes below; seeds `s0`, `s1` |
| `D_ours_<mix>_<seed>_names_via_rules` | The same head's corrected camera path, named by the rule labeller instead of the head's card output |

| Mix | Training clips |
|---|---|
| `DG` | 6,000 grammar clips (source G, run one's seeds) |
| `DF` | 6,000 free-form clips (source F) |
| `DH` | 6,000 human-path clips (source H: RealEstate10K poses and Big Buck Bunny shot cameras) |
| `DGFH` | 2,000 of each source, 6,000 in all |

COLMAP ran on test_combos, test_human and val only. Qwen3-VL ran on test_human and test_moving only. Rows for other splits are absent, not zero.

## Splits

| Split | Clips | What it holds |
|---|---|---|
| `val` | 600 | 200 per source (G, F, H). Picks the mix and each arm's handheld threshold |
| `test_scenes` | 1,000 | Run one's split, unchanged: grammar moves in held-out assets |
| `test_combos` | 500 | Run one's split, unchanged: the five held-out move pairs in held-out assets |
| `test_human` | 1,013 | Held-out human camera paths (1,000 RealEstate10K test-list windows, 13 Big Buck Bunny windows from scenes 08 to 13) in held-out assets |
| `test_moving` | 500 | Grammar and free-form moves (alternating) with 1 to 3 props moving in every clip, held-out assets, held-out pairs filtered out |

`subset` is `all`, or `moving` for clips whose ground-truth card is not "static". The S2 criterion uses `test_human` / `moving`.

## results/arms.csv

One row per split, subset and arm.

| Column | Meaning |
|---|---|
| `split`, `subset`, `arm` | As above |
| `n_clips` | Clips scored |
| `failure_rate` | Share of clips where the arm gave no usable path (COLMAP did not register all 16 frames, or a non-finite or non-positive focal). A failed clip is scored as "nothing moved" on every name, so it is right only on axes that truly did not move, and it is left out of the path metrics |
| `rot_err_deg` | Mean geodesic rotation error over frames 1 to 15, each relative to frame 0, degrees. Empty for the VLM |
| `ate` | RMS trajectory error after a similarity (Sim3) alignment, in units of the true median scene depth D; clips whose true camera travels under 0.03 D are left out |
| `zoom_err` | Mean absolute error of ln(focal / focal at frame 0) |
| `dir_acc` | Mean per-axis direction accuracy over 7 axes (pan, tilt, roll, dolly, truck, pedestal, zoom), each named one of two directions or none |
| `dir_acc_ci_lo`, `dir_acc_ci_hi` | Bootstrap 95% CI over clips |
| `zoom_f1` | Macro F1 over the three zoom classes (in, out, none). Near 0.33 for every arm on `test_human`, because RealEstate10K windows have a fixed focal and almost no zoom: ignore it there |
| `orbit_acc` | Accuracy of the orbit flag |
| `handheld_acc` | Accuracy of the handheld flag. B, A1, A2 and the `names_via_rules` rows use a threshold calibrated on val; D rows use the head's own flag |
| `size_acc` | Mean per-axis size class accuracy (none, small, medium, large) over all 7 axes. Empty for the VLM |
| `exact_card` | Share of clips where all 7 axis directions and the orbit flag match ground truth. Sizes, handheld and easing are scored separately and do not enter it (run one's definition) |
| `exact_card_ci_lo`, `exact_card_ci_hi` | Bootstrap 95% CI over clips (1,000 resamples) |
| `zoom_read_as_dolly` | On clips with a true zoom and no true dolly: share where the arm named a dolly in or out. Empty where no such clips exist |
| `handheld_threshold_deg` | The val-calibrated handheld threshold the arm used, degrees of residual rotation. Empty for D rows and the VLM |

## results/paired_vs_B.csv

One row per split, subset, arm and metric: the arm minus `B_da3_base_rules` on the same clips.

| Column | Meaning |
|---|---|
| `metric` | `exact_card` or `dir_acc` for the naming splits; `psnr`, `ssim` or `lpips` for split `reshoot_test_scenes` |
| `diff_vs_B` | Mean per-clip difference, arm minus B (0.01 = 1 point for exact card) |
| `ci_lo`, `ci_hi` | Paired bootstrap 95% CI (2,000 resamples of clips) |
| `n_clips` | Clips in both arms |

For `reshoot_test_scenes`, `arm` is `D_F`, `D_G`, `D_H` or `D_GFH` (seed 0). Lower LPIPS is better, so a negative `lpips` difference favours the head.

## results/by_type.csv

One row per split, subset, arm and group.

| Column | Meaning |
|---|---|
| `group` | A move type present in the ground-truth card (`pan`, `tilt`, `roll`, `dolly`, `truck`, `pedestal`, `zoom`, `orbit`, `dollyzoom`, `static`), or a slice: `handheld` / `not_handheld`, `source_G` / `source_F` / `source_H`, `human_re10k` / `human_bbb`, `moving_props` / `no_moving_props`, `gt_static` / `gt_moving`, `human` |
| `n_clips` | Clips in the group. A clip with two move types counts in both |
| `exact_card`, `dir_acc`, `rot_err_deg`, `failure_rate` | As in `arms.csv`, within the group |

## results/training.csv

One row per trained head.

| Column | Meaning |
|---|---|
| `run` | `<mix>_s<seed>` |
| `train_split` | Training clips as `split:count` |
| `seed` | Training seed |
| `best_val_loss` | Lowest total val loss over 40 epochs (the checkpoint kept) |
| `best_epoch` | Epoch of that loss, counting from 0 |
| `epochs` | Epochs run |
| `seconds` | Wall time of the run (four ran at once on one H100 PCIe) |

## results/criteria.csv

One row per mix: the three pre-registered criteria from `PLAN.md`.

| Column | Meaning |
|---|---|
| `mix`, `label` | Mix id and name |
| `val_exact_mean` | Exact card on val, mean of the two seeds. The mix with the highest value is the one judged |
| `S1_combos_diff_<seed>`, `S1_combos_ci_<seed>` | test_combos exact card, head minus B, paired, and its 95% CI. Passes if the CI is above zero |
| `S2_human_moving_diff_<seed>`, `S2_human_moving_ci_<seed>` | Same on test_human moving windows |
| `S3_test_scenes_exact_<seed>` | test_scenes exact card. Passes at 0.860 or above (run one's head, 0.890, minus 3 points) |
| `pass_<seed>` | All three pass for that seed |
| `picked_on_val` | True for the mix with the highest `val_exact_mean` |

## results/reshoot.csv

One row per re-shoot arm: 200 test_scenes clips re-rendered with each arm's camera path and compared to the original frames.

| Column | Meaning |
|---|---|
| `arm` | `GT` (the true path re-rendered, the noise floor), `STATIC` (camera held at frame 0, the do-nothing floor), `B`, `D_G`, `D_F`, `D_H`, `D_GFH` (seed 0) |
| `n_clips` | Clips scored |
| `psnr`, `ssim` | Mean over all frames and clips; higher is better |
| `lpips` | Mean LPIPS (AlexNet, v0.1); lower is better |
| `lpips_ci_lo`, `lpips_ci_hi` | Bootstrap 95% CI over clips |

Every arm borrows the true scene depth to set its translation scale, so the re-shoot tests direction and shape of the path, not scale recovery.

## raw/

| Path | Contents |
|---|---|
| `raw/results/results_<split>[_moving].json` | Every metric per arm, with intervals, by-type breakdowns and paired differences |
| `raw/results/per_clip_<split>.json` | Per-clip scores used for the paired bootstrap |
| `raw/results/eval_*.txt` | Human-readable score tables |
| `raw/results/leak_check.txt` | Leak check output (PASS) |
| `raw/results/reshoot.json`, `reshoot.txt`, `reshoot_jobs.jsonl` | Re-shoot scores and the job list that produced them |
| `raw/runs/<mix>_s<seed>/` | `model.pt` (weights), `train_log.json`, `pred_<split>.json` |
| `raw/runs/vlm/` | Qwen3-VL raw answers |
| `raw/data/<split>/<clip>/meta.json` | Scene spec, move spec, source, moving props and exact cameras for every clip (21,717 files). Frames were not kept; `harness/render_clips2.py` rebuilds any clip from its spec |
| `raw/pools/` | Human-path window pools (`H_train`, `H_val`, `H_test`) |
| `raw/logs/` | Render (gzipped), baseline, training and pipeline logs |
| `raw/env/` | `nvidia-smi -q`, `pip freeze` for both venvs, Blender and DA3 versions |
