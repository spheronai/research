# Data for study 2, run one: can a small model read a camera move well enough to re-shoot it?

Every CSV in `results/` is produced by `harness/analyze.py` from the files in `raw/`. Rerun it from this folder with:

```bash
uv run -q --with matplotlib --with numpy python harness/analyze.py
```

It reads `raw/results/results_*.json`, `raw/results/reshoot.json`, `raw/results/reshoot_paired.json` and `raw/runs/seed*/train_log.json`, and writes `results/*.csv` and `charts/*.{svg,png}`. The script was written after the run (25 Nov 2026). It reshapes numbers already in `raw/results/` and computes one new figure, the LPIPS interval in `reshoot.csv`.

Charts use the Spheron chart style in `harness/spheron_chart_style.py`. Set `SPHERON_FONT_DIR` to a folder holding `SFPRODISPLAYREGULAR.otf` and `OCRAStd.otf` to render them in the brand faces; without it they fall back to system fonts.

## Arms

| Arm id | What it is |
|---|---|
| `REF_always_static` | Reference floor: the camera never moves, the card says "static" |
| `A1_colmap_per_image_focal` | COLMAP 4.2 (pycolmap), SIFT, exhaustive matching, one camera per frame, 3 seeds per clip; a clip fails unless all 16 frames register |
| `A2_colmap_shared_focal` | Same, one shared camera (fixed focal) |
| `B_da3_base_rules` | Depth Anything 3 (DA3-Base), feed-forward poses and focal, named by the rule labeller |
| `C_qwen3_vl_8b` | Qwen3-VL-8B-Instruct, zero-shot, greedy, one fixed prompt. Names only, no path |
| `D_ours_seed0`, `D_ours_seed1` | The trained head on frozen DA3-Base, names from its own card output |
| `D_ours_seed<n>_names_via_rules` | The same head's corrected camera path, named by the rule labeller instead |

## Splits

| Split | Clips | What it holds |
|---|---|---|
| `test_scenes` | 1,000 | Grammar moves in assets never seen in training |
| `test_combos` | 500 | The same unseen assets, and five move pairs never seen in training |
| `human_moves` | 62 | Big Buck Bunny's shot cameras (4-second windows) replayed in test scenes. Added after the main results were known |

`subset` is `all`, or `moving` for the 17 `human_moves` windows whose true card is not "static". The `moving` rows have no static floor or `names_via_rules` rows.

## results/arms.csv

One row per split, subset and arm.

| Column | Meaning |
|---|---|
| `split`, `subset`, `arm` | As above |
| `n_clips` | Clips scored |
| `failure_rate` | Share of clips with no usable path (COLMAP did not register all 16 frames, or a camera no lens can have). A failed clip is scored as "nothing moved" on every axis and as wrong on the exact card, and it is left out of the path metrics |
| `unusable_camera_rate` | Share of clips where the arm returned a focal of zero or below, or non-finite values. Already counted in `failure_rate` |
| `rot_err_deg`, `rot_err_ci_lo`, `rot_err_ci_hi` | Mean geodesic rotation error over frames 1 to 15, each relative to frame 0, degrees, with bootstrap 95% CI. Empty for the VLM |
| `ate` | RMS trajectory error after a similarity (Sim3) alignment, in units of the true median scene depth; clips whose true camera travels under 0.03 of that depth are left out |
| `zoom_err` | Mean absolute error of ln(focal / focal at frame 0) |
| `dir_acc`, `dir_acc_ci_lo`, `dir_acc_ci_hi` | Mean per-axis direction accuracy over 7 axes (pan, tilt, roll, dolly, truck, pedestal, zoom), each named one of two directions or none, with bootstrap 95% CI |
| `zoom_f1` | Macro F1 over the three zoom classes (in, out, none) |
| `orbit_acc` | Accuracy of the orbit flag |
| `handheld_acc` | Accuracy of the handheld flag. B, A1, A2 and `names_via_rules` rows use a threshold calibrated on val; D rows use the head's own flag |
| `size_acc` | Mean per-axis size class accuracy (none, small, medium, large) over all 7 axes. Empty for the VLM |
| `exact_card`, `exact_card_ci_lo`, `exact_card_ci_hi` | Share of clips where all 7 axis directions and the orbit flag match ground truth, with bootstrap 95% CI (1,000 resamples). Sizes, handheld and easing do not enter it |
| `zoom_dir_acc_on_zoom_clips` | On clips with a true zoom: share where the zoom direction was right |
| `zoom_read_as_dolly` | On clips with a true zoom and no true dolly: share where the arm named a dolly in or out |
| `handheld_threshold_deg` | The val-calibrated handheld threshold, degrees of residual rotation. Empty for D rows and the VLM |

## results/by_type.csv

One row per split, subset, arm and group.

| Column | Meaning |
|---|---|
| `group` | A move type in the true card (`static`, `pan`, `tilt`, `roll`, `dolly`, `truck`, `pedestal`, `zoom`, `orbit`, `dollyzoom`), or `handheld` / `not_handheld`, or `human` (every `human_moves` clip) |
| `n_clips` | Clips in the group. A clip with two move types counts in both |
| `exact_card`, `dir_acc`, `rot_err_deg`, `failure_rate` | As in `arms.csv`, within the group |

## results/training.csv

| Column | Meaning |
|---|---|
| `run` | `seed0` or `seed1` |
| `train_split`, `seed` | Training split (6,000 clips) and seed |
| `best_val_loss` | Lowest total val loss over 40 epochs (the checkpoint kept) |
| `best_epoch` | Epoch of that loss, counting from 0 |
| `epochs` | Epochs run |
| `seconds` | Wall time of the run (both seeds ran at once on one H100 PCIe) |

## results/reshoot.csv

One row per re-shoot arm: 200 test_scenes clips re-rendered with each arm's camera path, frames 1 to 15 compared with the originals.

| Column | Meaning |
|---|---|
| `arm` | `GT` (true path re-rendered, the noise floor), `STATIC` (camera held at frame 0, the do-nothing floor), `A1`, `A2`, `B`, `D` (seed 0). A clip an arm failed is re-shot locked off |
| `n_clips` | Clips scored |
| `psnr`, `ssim` | Mean over clips; higher is better |
| `lpips`, `lpips_ci_lo`, `lpips_ci_hi` | Mean LPIPS (AlexNet, v0.1), lower is better, with a bootstrap 95% CI over clips (1,000 resamples, seed 0) computed by `analyze.py` |

## results/reshoot_paired.csv

Paired differences on the same 200 clips, from `raw/results/reshoot_paired.json`.

| Column | Meaning |
|---|---|
| `comparison` | `D_minus_B`, `B_minus_A2`, `D_minus_STATIC`, `B_minus_STATIC`: first arm minus second |
| `metric` | `psnr`, `ssim` or `lpips` |
| `mean_diff`, `ci_lo`, `ci_hi` | Mean per-clip difference and its bootstrap 95% CI. A negative LPIPS difference favours the first arm |
| `first_better_share` | Share of clips where the first arm scored better on that metric. The script that wrote this file ran on the VM and is not in `harness/` |

## raw/

| Path | Contents |
|---|---|
| `raw/results/results_*.json` | Every metric per arm, with intervals and by-type breakdowns |
| `raw/results/eval_*.txt` | Human-readable score tables |
| `raw/results/leak_check.txt` | Leak check output |
| `raw/results/reshoot*.json`, `reshoot.txt` | Re-shoot per-clip scores and paired differences |
| `raw/results/real_tos_cards.json` | Cards for 13 Tears of Steel shots (no ground truth) |
| `raw/results/reshoot_zoom_examples.png` | Three zoom clips: original, DA3 and the head at frames 0, 7 and 15 |
| `raw/runs/seed0`, `raw/runs/seed1` | `model.pt`, `train_log.json`, `pred_<split>.json` |
| `raw/runs/vlm/` | Qwen3-VL raw answers |
| `raw/data/<split>/<clip>/meta.json` | Scene spec, move spec and exact cameras for every clip (8,093 files). Frames were not kept; `harness/render_clips.py` rebuilds any clip |
| `raw/jobs/`, `raw/assets/manifest.json`, `raw/ext/` | Job lists, the Poly Haven asset manifest, Big Buck Bunny and Tears of Steel extracts |
| `raw/nvidia-smi-q.txt`, `raw/pip-freeze.txt` | GPU and package versions (run two keeps these under `raw/env/`) |
| `raw/logs/` | Render, baseline, training and pipeline logs |
