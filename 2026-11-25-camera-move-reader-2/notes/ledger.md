# Ledger

Times are UTC. Public copy: price, budget, account and deployment details were removed.

## 24 Nov 2026 (UTC; 25 Nov local)

- Plan approved without the optional phone clips.
- Harness built in `harness/` from run one's copy (`harness/r1/` keeps run one's drivers for reference). New or changed files:
  - `moves_free.py`: source F, free-form paths.
  - `human_pool.py`: source H, 4 s windows from RealEstate10K poses and Big Buck Bunny shot cameras, split as the plan says.
  - `render_clips2.py`: run one's scenes and renderer with the three sources, the held-out-pair filter on the placed path's card, and moving props. A G clip with no filter and no moving props takes run one's exact code path, so run one's test job files re-render unchanged.
  - `make_jobs2.py`: job lists. train_G reuses run one's train scene seeds and clip numbers.
  - `fetch_from_manifest.py`: fetches run one's exact Poly Haven ids from its manifest, instead of re-sampling.
  - `setup2.sh`: run one's setup, DA3 at run one's commit (3d835ec), RealEstate10K poses.
  - `train.py`: `--train-split` takes several splits with counts (`train_G:2000 ...`, first N clips by id). Model, losses, optimiser and checkpoint rule unchanged.
  - `evaluate.py`: scores every `runs/D*` directory, stores per-clip results, paired bootstrap D minus B, `--subset moving`, breakdowns by source and moving props.
  - `leak_check.py`: held-out pairs checked on the placed path's card too; human-path video ids and BBB scene files checked across sides.
  - `reshoot_prep.py` / `reshoot_score.py`: one D row per training mix (seed 0), no COLMAP rows (not run on test_scenes in run two), paired D minus B.
  - `det_check.py`, `pilot_stats.py`, `pilot2.sh`, `run_render2.sh`, `baselines2.sh`, `final2.sh`.
- Implementation choices the plan left open, fixed before any rendering:
  - Held-out pairs are matched on the card's named parts, the same way `labeler.card()` writes them (an orbit absorbs its pan and truck, a dolly zoom its dolly and zoom). A training clip is redrawn until its card has none, up to 24 tries.
  - RealEstate10K travel is scaled against the median ray depth at the placed first frame, the same D the labeller uses.
  - Human-path windows are 16 samples evenly over 4.0 s (run one's grammar timing). Run one's 17 human windows used 0.25 s steps over 3.75 s; run two's test_human windows are new, so this does not change a run-one number.
  - val's H clips come from RealEstate10K training videos held out of train_H by `sha1(video_id) % 10 == 0`; no Big Buck Bunny in val.
  - Re-shoot rows: GT, STATIC, B and one D row per training mix (7 rows, the plan said 6).
  - Determinism check: run one's rendered frames were not kept, so the pilot (a) compares re-rendered cameras, focal lengths, depth and scene specs with run one's saved meta.json, and (b) renders the same 20 clips twice here and compares pixels.
- 21:09 UTC: deployed 1x H100 PCIe 80 GB on Spheron, the same machine type as run one. Firewall: inbound TCP 22 only. Auto-terminate timer at 2026-11-25 11:10 UTC (14 h).
- 21:12 UTC: setup (`setup2.sh`): Blender 4.5.17, venv (torch 2.8 cu128), venv-da3 (DA3 at 3d835ec), RealEstate10K poses (tarball sha256 af18e4d5...91db; 71,556 train and 7,711 test clips, 6,559 and 696 videos, no video in both lists). Run one's 320 Poly Haven assets fetched from its manifest, none missing.
- Human-path pools, first try: one random clip per video gave only 296 test windows, because about half of RealEstate10K clips are shorter than 4 s (measured on the test list: median 3.9 s, 49.5% at 4 s or longer; frame step 0.033 s). Changed before any rendering to clips of at least 4 s, round-robin over videos, at most 2 per video. Pools: train 8,493 windows (8,448 RealEstate10K + 45 Big Buck Bunny), val 400 (RealEstate10K), test 1,013 (1,000 RealEstate10K + 13 Big Buck Bunny). Copied to `raw/pools/`.
- Job lists (`make_jobs2.py`): train_G, train_F, train_H 6,000 each; val 600 (200 per source); test_human 1,013; test_moving 500; run one's test_scenes 1,000 and test_combos 500 unchanged. 21,757 clips with the pilots.
- Pilot, determinism: 20 test_scenes clips re-rendered. Scene specs identical to run one 20/20; cameras, focal lengths and scene depth identical to run one's meta.json (max difference 0). Two renders on this machine: 65.2 dB mean PSNR, 63.9 dB minimum over 320 frames (run one measured about 65 dB). **Pass.**
- Pilot, sources (52 clips each): render 2.9 to 3.3 s per clip per process with 4 processes. Held-out-pair redraws: 20 (F) and 21 (H) over 52 clips. Moving props in 15 and 12 clips (target 30%). RealEstate10K median travel 0.08 of scene depth.
- **Pilot change to source F, before the freeze.** As first written, each F channel had its own start/stop time. 42 of 52 pilot F clips were labelled handheld though only 30% get shake. Tested on 400 shake-free F paths: 84.8% read as handheld with start/stop windows, 0% with full-length channels, 0% for the grammar. The labeller's degree-5 fit cannot follow a move that starts or stops mid-clip, so the residual crosses the handheld threshold. Start/stop times were dropped; every channel now runs over the full 4 s with its own easing. Re-run pilot: 14 of 52 handheld (27%). This removes one feature the plan listed for source F.
- Note on source H: 28 of 52 pilot H clips read as handheld. That is RealEstate10K's own frame-to-frame jitter (real gimbal and phone footage), scaled with the path, so it stays.
- Known asset issue carried from run one: Poly Haven `old_military_crate` fails to import ("Cannot apply to a multi user") and is skipped and recorded in the scene spec, as in run one.
- **Freeze:** SHA-256 of PLAN.md, harness, job lists and pools in `notes/plan.sha256` (43 files). Harness on the VM checked identical to the local copy (md5).
- ~21:29 UTC: full run started on the VM (`go.sh`): `run_render2.sh` (test splits first, then val, train_G, train_F, train_H), `baselines2.sh` (DA3 on every split, COLMAP on test_combos, test_human and val, Qwen3-VL on test_human and test_moving), `final2.sh` (waits, then trains, predicts, leak check, scores, re-shoot).
- 21:39 UTC: first 10 minutes rendered 702 clips (about 0.86 s per clip with DA3 alongside), which projected about 5.2 h for everything.
- The VM was not checked between 21:40 and 06:27 UTC. It kept running on its own. Split finish times: test_scenes 21:44, test_combos 21:51, test_human 22:06, test_moving 22:16, val 22:50, train_G 01:43, train_F 04:17; train_H at 5,195 of 6,000 at 06:27. COLMAP (test_combos, test_human, val) done 23:42; val COLMAP per-image: 333 ok, 267 failed. Qwen3-VL done 22:30. No tracebacks, no skipped clips.
- Rendering slowed on the training splits to about 1.6 s per clip overall (6.2 s per clip per process on train_H). Each Blender process grew to about 24 GB of RAM, because loaded textures and HDRIs stay in memory across scenes; the likely cause of the slowdown, not confirmed.
- **Deviation: the render cut rule was not applied.** The plan says to cut each training source to 4,000 clips if rendering projects past 8.5 h. Rendering will total about 9.4 h, so the trigger was hit, but this was not noticed until 06:27, when 17,195 of 18,000 training clips were already rendered. Cutting then would only have dropped data. Training stays at 6,000 clips per arm, as the plan's main design and run one.
- 07:02 UTC: rendering finished (train_H last). DA3 done on all splits at 07:02:49, 21,613 clips. Total render wall time about 9.5 h.
- 07:03 to 07:54 UTC: trained 8 runs (DG, DF, DH, DGFH x seeds 0 and 1), four at a time, 40 epochs each, about 25 min per run. Best epochs by val loss (counted from 0): DG 15 and 20, DF 13 and 12, DGFH 18 and 19. **DH collapsed**: best val loss at epoch 1, the second epoch, for both seeds (11.44 and 11.53, against 3.8 to 5.1 for the other mixes), so its checkpoint is close to untrained. Kept as is; the plan says to keep the best-val checkpoint, and DH is not the mix picked on val.
- 07:54 to 07:58 UTC: predicted every split, leak check **PASS** (0 shared assets; 0 held-out pairs in train or val; every test_combos clip has one; 0 repeated first frames; 3,744 train-side and 549 test-side human-path videos, 0 on both sides). Scored val, test_scenes, test_combos, test_human, test_human moving windows and test_moving.
- Mix picked on val (exact card, mean of two seeds): **D-GFH 74.3%** (D-G 69.5%, D-F 68.5%, D-H 31.9%; B 62.8%).
- Pre-registered criteria for D-GFH: S1 test_combos D minus B -13.6 and -13.8 points (paired 95% CI entirely below zero), **fail**. S2 test_human moving -14.3 and -12.7 points (CI below zero), **fail**. S3 test_scenes 85.9% and 86.6% against a floor of 86.0%: fails on seed 0, passes on seed 1. **No mix passes all three.** The outcome is the one the plan named in advance: publish runs one and two together as a negative result for the learned namer and stop this line.
- 07:58 to 08:21 UTC: re-shoot, 200 test_scenes clips x 7 rows (GT, STATIC, B, D_G, D_F, D_H, D_GFH; seed 0 for each D). Scored with paired intervals.
- 08:21 UTC: copied off results, weights, predictions, clip specs (21,717 meta.json), pools, env and logs (render logs gzipped, 1.3 GB). Grepped the copy for the VM's IP and for keys: none found. Rendered frames stayed on the VM, as planned.
- 08:25:33 UTC: machine terminated and confirmed terminated; no other machine running.
- Local: `harness/analyze.py` wrote `results/*.csv` (arms 119 rows, paired_vs_B 204, by_type 1,665, training 8, criteria 4, reshoot 7) and `charts/exact-card-vs-B` and `charts/exact-card-by-split` (SVG and PNG).
