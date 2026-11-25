# Study 2 plan: Can a small model read a camera move well enough to re-shoot it?

> **Public copy.** Lines about price, budget, region, deployment and account, and who approved the plan, were removed from this file before publication. Its SHA-256 therefore no longer matches the `PLAN.md` entry in `notes/plan.sha256`, which covers the unredacted plan. The harness files listed there are unchanged.

Status: **frozen, 24 Nov 2026, before any model was trained or any test split was scored.** Revised once from the first draft after the pilot; what changed and why is listed in "Changes from the first draft" below. This file is frozen (SHA-256 recorded in `notes/plan.sha256`) before any test split is scored. Changes after the freeze go in `REPORT.md` under "Deviations".

## Question

Given a short video shot, can one small open model return both (a) the per-frame camera path, including focal length, accurately enough to re-render the shot, and (b) the move named the way a camera operator would name it ("slow push-in, tilt up, eases out")? How does it compare with a structure-from-motion pipeline, a learned feed-forward camera estimator, and an open vision-language model asked to name the move?

Why it is open: [CameraBench](https://arxiv.org/abs/2504.15376) found that structure-from-motion recovers geometry but cannot name moves, and vision-language models name moves but miss the geometry. The tools that take a camera path as input (camera-controlled video models, Blender, Nuke) need the path as a file, and editors and shot lists need the name. We found no open tool that returns both. The prior-art check for Sep to Nov 2026 is in `notes/prior_art.md`.

What would change the answer for a reader: if a feed-forward camera estimator plus a simple naming rule matches our model on both outputs, the honest result is "you do not need a new model, here is the rule", and we publish that.

## Data

All training and test data is rendered by us in Blender from CC0 assets, so every frame has an exact camera. Licences were checked for every source before the run; `REPORT.md` names the licence of each source used.

- **Scenes.** A procedural generator (`harness/render_clips.py`, `sample_scene` and `build_scene`) builds a set from CC0 Poly Haven assets (170 models, 90 HDRIs, 60 ground textures, fetched by `harness/fetch_assets.py`): a textured ground, 10 to 28 props at random positions and scales, an HDRI for light and distant background. Each scene has a seed and its full spec is saved with every clip, so it can be rebuilt exactly.
- **Moves.** A move grammar (`harness/moves.py`) samples one to three primitives per clip: pan, tilt, roll, dolly (push in / pull out), truck, pedestal (crane up / down), zoom, orbit (arc), dolly zoom, or static. Each gets a size (small / medium / large), an easing curve (linear, ease in, ease out, ease in-out) and, for 30% of clips, handheld shake.
- **Clips.** 16 frames sampled evenly from a 4 second move at 24 fps, rendered at 392 x 224 with Blender 4.5.17 LTS, Cycles on the GPU (OptiX), 24 samples, GPU OpenImageDenoise. Scale comes from the median of a 24 x 14 grid of ray hits from the frame-0 camera (`D_median`). Four clips per scene, four Blender processes sharing the GPU.
- **Labels from geometry, no annotators.** A rule labeller (`harness/labeler.py`) reads the ground-truth path and writes the move name: per axis (pan, tilt, roll, dolly, truck, pedestal, zoom) a direction (negative, none, positive) and a size, plus flags for orbit, dolly zoom, handheld and easing. Thresholds were tuned on the 48 pilot clips only and are fixed in the file from here on. The same labeller turns any predicted path into a name, which gives every geometry baseline a naming score too.

### Splits (frozen before training)

| Split | Clips | What is held out |
|---|---|---|
| train | 6,000 | nothing |
| val | 500 | used for early stopping only |
| test-unseen-scenes | 1,000 | HDRIs and prop models never seen in training (15% of each asset pool reserved) |
| test-unseen-combos | 500 | move pairs never seen in training (tilt+truck, pan+pedestal, zoom+pedestal, roll+dolly, orbit+tilt; `HELDOUT_PAIRS` in `harness/moves.py`), on unseen assets |

The test asset pool is every asset whose `sha1(id) % 100 < 15`. Training reads only `train` and `val`; `train.py --predict` loads test clips without their ground truth (`need_gt=False`). Ground truth on test is read only by `harness/evaluate.py` and `harness/reshoot_*.py`. A red-team check confirms no asset id or held-out move pair leaks into train or val.

Out-of-domain checks, reported separately and not used for any choice:

- **Blender Open Movie shots** (CC BY). If budget and time allow after the main study, production `.blend` files with shot cameras (Big Buck Bunny, Sprite Fright) give ground truth on shots a human camera artist made. If not done, the report says so.

## Arms

| Arm | Method | Path output | Name output |
|---|---|---|---|
| A1 | COLMAP (pycolmap 4.2.1), SIFT, exhaustive matching, incremental mapping, one camera per frame so focal can change, 3 seeds per clip, a clip fails unless all 16 frames register | yes | via the rule labeller |
| A2 | Same, one shared camera (fixed focal), the usual default | yes | via the rule labeller |
| B | Depth Anything 3, DA3-Base (Apache-2.0), feed-forward poses and intrinsics, frame 0 as reference | yes | via the rule labeller |
| C | Qwen3-VL-8B-Instruct (Apache-2.0), zero-shot, greedy, one fixed prompt defining each term (`harness/vlm_baseline.py`), 16 frames as a 4 fps video | no | directions, orbit, handheld only |
| D | **Ours** (`harness/train.py`): a 6-layer transformer (width 256) trained on top of frozen DA3-Base. Input per frame: DA3's camera relative to frame 0 and its focal, plus DA3 backbone features (layer 11) pooled to 84 tokens. Output: a residual correction to DA3's per-frame camera and focal, and the move card. AdamW 3e-4, 40 epochs, batch 32, best val loss kept | yes | yes, directly (primary), and via the rule labeller (secondary) |
| D2 | Ours, second training seed (duplicate run) | yes | yes |

## Changes from the first draft (made after the pilot, before any training or test scoring)

- **Arm D sits on DA3-Base instead of DINOv2.** The pilot showed DA3 already gives a usable path but misreads zoom as a push-in (pilot_00009: true focal 65 to 85.5 mm, DA3 60.3 to 60.8 mm). A head that corrects a strong estimator is the more useful and more honest test than one that starts from scratch. It also makes arm B the exact ablation of arm D.
- **MultiCamVideo dropped.** 333 GB, no small sample available, and a fixed focal length, so it cannot test zoom.
- **COLMAP run two ways.** Per-frame focal (can represent zoom, diverged on the pilot: focal 0.0 to 153,967 mm) and shared focal (the usual default).
- **Props 10 to 28** instead of 6 to 20, so COLMAP and DA3 have enough texture to work with.
- **Handheld threshold calibrated per arm on val** (see Metrics).
- **Scale from a ray-hit grid**, not a depth render pass.

## Metrics

Path (on clips where a method returns a path; failures reported as a rate):

- Rotation error: mean geodesic angle between predicted and true per-frame rotation relative to frame 0, in degrees.
- Translation error: absolute trajectory error after similarity alignment (scale, rotation, translation), in units of median scene depth. We state this normalisation because papers differ.
- Zoom error: mean absolute error of the per-frame log focal ratio to frame 0 (the zoom profile), and separately the absolute focal error in mm. This separates zoom from dolly.
- ATE is reported only on clips where the true camera travels more than 0.03 scene-depth units; rotation and zoom errors use every clip.

Name:

- Per-axis accuracy (direction), macro F1 across all labels, and exact match of the full move card.
- Broken down by move type and by handheld vs not, and on zoom clips: zoom direction accuracy and how often a zoom with no dolly is called a dolly.
- Arms named by the rule labeller get one handheld threshold each, the value that maximises handheld accuracy on val for that arm, because noisy predicted paths would otherwise all read as handheld. All other thresholds are the labeller's.
- A failed clip counts as "no move, not handheld" for naming, and is left out of the path metrics, with the failure rate reported next to them.

**Re-shoot test** (test-unseen-scenes, 200 clips per arm with a path): rebuild the scene from its saved spec, render each arm's predicted move anchored at the true frame-0 camera, with translation rescaled to the true scene depth and focal following the predicted zoom profile from the true starting focal, and compare frames 1 to 15 with the original: PSNR, SSIM and LPIPS (AlexNet). A failed clip is re-shot locked off. Two reference rows: the true path re-rendered (noise floor; 65 dB PSNR in the smoke test) and a locked-off camera (do-nothing floor). Anyone with the released scenes can repeat it.

Uncertainty: 95% bootstrap intervals over clips (1,000 resamples). Differences inside the interval are reported as no measurable difference.

## Compute and time

GPU: 1x H100 PCIe 80 GB on Spheron, deployed 24 Nov 2026 17:08 UTC.

| Step | Estimate |
|---|---|
| Setup: Blender, Python stack, assets | 1 h |
| Pilot: 50 clips, render speed, labeller check | 0.5 h |
| Render 8,000 clips (128,000 frames), GPU and CPU in parallel | 4 to 6 h |
| Features and poses (DA3-Base) | 0.5 h |
| Baselines A, B, C on test splits | 2 to 3 h |
| Train D and D2 | 2 h |
| Re-shoot renders and scoring | 1 h |
| Out-of-domain checks, copy off, terminate | 1 h |
| **Total** | **12 to 15 GPU hours (estimate)** |

**Stop rule:** a hard spending stop for this deployment. If the render step projects past 8 h, cut train to 4,000 clips and record it as a deviation. An auto-terminate timer is set for 26 Nov 2026 05:00 UTC.

## Safety

- Firewall: inbound TCP 22 only. No service listens on a public port.
- Results copied off the VM after each step into `raw/`, so a late failure loses at most one step.
- At the end, the machine is terminated and confirmed terminated before the study is called done.

## What gets published

Model weights (Apache-2.0, if every input allows it), the scene generator and harness (MIT), the rendered dataset and split files (CC BY 4.0, if asset licences allow), all results, logs, the red-team note and this plan. A negative result is published in the same format.

## Risks

- **Synthetic to real gap.** Real lenses, motion blur, rolling shutter and moving people are absent or simplified. Out-of-domain checks measure it; the report does not claim real-footage accuracy beyond what is measured.
- **Static scenes only.** Props do not move, so the model never learns to ignore moving subjects. Stated as a limit.
- **Scale ambiguity.** One camera cannot measure absolute distance; translation is reported in scene-depth units.
- **A baseline may already solve it.** That is a valid result (see "What would change the answer").
- **Prior art may land first.** Checked in `notes/prior_art.md` on 24 Nov 2026.
