# Study 2, run two: does the camera move reader generalise once it trains on more than our move grammar?

> **Public copy.** Lines about price, budget, region, deployment and account, and who approved the plan, were removed from this file before publication. Its SHA-256 therefore no longer matches the `PLAN.md` entry in `notes/plan.sha256`, which covers the unredacted plan. The harness files listed there are unchanged.

Status: **approved 25 Nov 2026**, without the optional phone clips. Follows run one (`../2026-11-24-camera-move-reader/REPORT.md`). Freezes the same way: SHA-256 of this file and the harness in `notes/plan.sha256` before any model is trained or any test split is scored.

## Why a second run

Run one trained a small head on frozen Depth Anything 3 (DA3-Base) and found three things:

1. On held-out scenes with moves from the training grammar it beat every baseline: exact move card 89.0% vs 62.0% for DA3 plus naming rules, and pure zooms misread as dollies on 5% to 15% of zoom clips vs 75%.
2. On move pairs it never saw, the gain was gone: 53.6% vs 51.0% exact, intervals overlapping. On orbit plus tilt it was worse than its own input, 22% vs 55%.
3. On 17 moving windows of Big Buck Bunny's real shot cameras, no measurable gain: 71% and 65% vs 77%.

The likely cause is the training data, not the model. Every training clip was one to three named primitives drawn from a fixed list, and the head learned those combinations. Run two changes the training motion and holds the model fixed, so one variable moves.

## Question

With the model unchanged, does training on motion that is not built from our primitives make the head beat DA3 plus rules on moves it has not seen? Two kinds of unseen move count: compositions of primitives held out of training, and camera paths keyed or shot by people.

**What would change the answer.** If no training mix beats DA3 plus rules on the human-move tests, we publish run one and run two together as "the feed-forward estimator plus a rule is the tool; a learned namer did not generalise", and stop this line.

## Pre-registered success criteria (decided now, before any rendering)

The best training mix (picked on val only) counts as a success only if **all three** hold:

| # | Test | Criterion |
|---|---|---|
| S1 | test_combos (held-out pairs, unchanged from run one) | Exact card higher than B (DA3 plus rules), **paired** bootstrap 95% CI of the difference above zero |
| S2 | test_human (held-out human camera paths, moving windows only) | Exact card higher than B, paired 95% CI above zero |
| S3 | test_scenes (unchanged from run one) | Not worse than run one's head by more than 3 points of exact card (it was 89.0%) |

Handheld is judged separately. Run one showed the rule with a val-calibrated threshold beats the head's own output (90.4% vs 81% to 82%). Run two reports both and picks on val.

## What changes from run one

### 1. Three sources of training motion

Every source excludes the five held-out pairs (tilt+truck, pan+pedestal, zoom+pedestal, roll+dolly, orbit+tilt). The exclusion is by the labeller's card for the path as placed in the scene, not by how the path was generated, so no source can leak a held-out composition.

| Source | What it is | Licence |
|---|---|---|
| **G**: grammar | Run one's move grammar, unchanged (`harness/moves.py`) | ours |
| **F**: free-form | Smooth random camera paths. Each of the 7 axes (yaw, pitch, roll, three translations, log focal) switches on independently with probability 0.35, with its own amplitude, easing and start/stop time inside the 4 s. With probability 0.3 the camera keeps looking at a target point, which makes arcs and orbits. Every pair and triple of axes appears except the five held out. Nothing is a named primitive, so a combination cannot be memorised. | ours |
| **H**: human | 4 s windows of real camera trajectories. **RealEstate10K** camera poses, about 10,000 YouTube clips shot by people on gimbals and phones (720 MB of pose text files, no video needed), plus the **Big Buck Bunny** shot cameras from run one. Each is replayed in our scenes. RealEstate10K has no metric scale and a fixed focal length, so translation is scaled so travel is log-uniform between 0.02 and 0.6 of scene depth. The label is read from the path as placed, so the scale choice only changes which label a clip gets, never whether it is correct. | RealEstate10K: ["licensed by Google LLC under a Creative Commons Attribution 4.0 International License"](https://google.github.io/realestate10k/download.html). Big Buck Bunny: CC BY 3.0, Blender Foundation |

**Splits for human paths.**

- RealEstate10K is split by video id. Its own train/test split is respected, and our test windows come only from its test list.
- Big Buck Bunny is split by scene number: files `01` to `07` for training, `08` to `13` for test.

**Training arms.** Every arm has 6,000 clips, so only the motion source differs:

| Arm | Motion source |
|---|---|
| D-G | 6,000 grammar clips. Repeats run one, so it doubles as a reproducibility check. |
| D-F | 6,000 free-form clips |
| D-H | 6,000 human-path clips |
| D-GFH | 2,000 of each |

Two seeds each. The model, losses, optimiser, epochs and checkpoint rule are run one's, byte for byte (`harness/train.py`).

### 2. Moving subjects

In 30% of clips from every source, one to three props move as rigid bodies during the shot: up to 0.5 scene-depth units of travel and up to 90 degrees of turn, with easing. Labels describe the camera only. A separate test split checks that the reader ignores subject motion. Real shots are full of moving people, which run one never covered.

### 3. Scoring

- **Paired bootstrap for every comparison.** The scorer stores per-clip results and reports D minus B with a 95% interval on the same clips. Run one compared separate intervals, which is weaker.
- **Same metrics as run one.** Rotation error, ATE, zoom error, per-axis direction, zoom F1, exact card, handheld, zoom-read-as-dolly rate, by-type breakdowns, and the re-shoot test (200 test_scenes clips, LPIPS as the main re-shoot metric because run one showed PSNR cannot separate methods here).
- **One rule written down this time.** In the exact card, a failed clip counts as wrong. Run one did this but did not state it in the plan.

## Splits

| Split | Clips | Purpose |
|---|---|---|
| train_G, train_F, train_H | 6,000 each | training sources |
| val | 600 (200 per source) | early stopping, mix choice, handheld threshold |
| test_scenes | 1,000 | **same seeds and specs as run one**, re-rendered: unseen assets |
| test_combos | 500 | same as run one: unseen assets and held-out pairs |
| test_human | 1,000 RealEstate10K test-split windows + Big Buck Bunny `08` to `13` windows | unseen human camera paths on unseen assets |
| test_moving | 500 | unseen assets, grammar and free-form moves, moving props in every clip |

**Test assets.** All test splits use run one's held-out asset pool (`sha1(id) % 100 < 15`).

**Leak check.** Extends run one's:
- No asset is shared between test and train.
- No held-out pair appears in any training card.
- No RealEstate10K video id or Big Buck Bunny scene file appears on both sides.
- No duplicate first frames.

## Arms scored

The same arms as run one, so every number is comparable:

| Arm | What it is |
|---|---|
| REF | Static reference |
| A1, A2 | COLMAP with per-image focal and with shared focal. test_combos and test_human only, since COLMAP gains nothing from new training data and run one already measured it on test_scenes. |
| B | DA3-Base plus rules |
| C | Qwen3-VL-8B. test_human and test_moving only. |
| D | D-G, D-F, D-H and D-GFH, two seeds each |

## Real footage with ground truth (optional, needs a person, not GPU time)

Run one had no real-footage test with ground truth. The cheapest owned option: someone shoots about 60 four-second clips on an iPhone with an app that records ARKit camera poses per frame, for example pans, tilts, walks forward and back, sideways moves, orbits around a person, and handheld. This gives real lenses, motion blur, rolling shutter and moving people with a measured camera path, under our own licence. Zoom cannot be tested this way, because ARKit tracking uses one fixed lens.

**Open point.** We have not confirmed which app exports per-frame poses and intrinsics; Record3D's page does not say. This needs a 15-minute check on a phone before it goes in. If nobody shoots, the report says the gap to real footage is still unmeasured.

**Not in run two.** The Blender Studio Vault (production shot files under CC BY 4.0 behind a paid subscription) and CameraBench (labels tagged CC BY 4.0, videos not redistributed) are kept for a later study. The Vault needs a subscription and its shots are slow to render. CameraBench would need its videos fetched from their sources, each under its own terms.

## Compute and time (estimates)

GPU: 1x H100 PCIe 80 GB on Spheron, the same machine type as run one. Run one measured every rate used here.

| Step | Basis (run one measurement) | Estimate |
|---|---|---|
| Setup, assets, RealEstate10K poses, pilot of 50 clips per new source | run one setup took 25 min | 0.7 h |
| Render about 21,750 clips (18,000 train, 600 val, 3,000 test, 150 pilot), DA3 and COLMAP overlapped | 0.85 s per clip alone, about 1.2 s with DA3 alongside | 5 to 7.5 h |
| Qwen3-VL on test_human and test_moving | about 0.5 s per clip | 0.2 h |
| Train 8 runs (4 arms x 2 seeds), two at a time | about 14 min per run in pairs | 1 h |
| Predict, score, re-shoot (200 clips x 6 rows), leak check | run one: 25 min | 0.5 h |
| Slack | | 1 h |
| **Total** | | **8.5 to 11 GPU hours** |

**Stop rule:** a hard spending stop, the program default for a study. If the render projects past 8.5 h, cut each training source to 4,000 clips (D-GFH to 1,333 each) before training and record it. An auto-terminate timer is set at deploy for 14 h after start.

## Safety

Same as run one:
- Firewall: inbound TCP 22 only.
- Results copied off after each step.
- Terminate at the end and confirm the machine is terminated.
- Rendered frames stay on the VM and are not copied. Clip specs, model weights, predictions and results are copied.

## What gets published if it works

- **Weights.** The best head, under Apache-2.0. DA3-Base is Apache-2.0, Poly Haven is CC0, RealEstate10K poses are CC BY 4.0 (attribution in the model card), and Big Buck Bunny is CC BY 3.0 (attribution).
- **Code.** The harness, under MIT.
- **Data.** Clip specs and results, under CC BY 4.0.

If it does not work, the two runs are written up together as a negative result in the same format.

## Risks

- **RealEstate10K is one kind of footage.** It is mostly slow interior walk-throughs: forward dollies, gentle pans, almost no zoom or roll. It may teach "slow forward" more than general motion. Big Buck Bunny and free-form cover the rest. The per-source arms show which one helps.
- **Scale for RealEstate10K is assumed.** Labels stay consistent with the placed path, but the size mix is our choice. Stated in the report.
- **Moving props are rigid and slow.** They are not people. The real-footage clips, if shot, are the only check against people.
- **Re-rendering run one's test sets** relies on determinism. Run one measured 65 dB PSNR for a re-render; the pilot repeats that check on 20 clips before anything else.
- **A source that helps human moves may hurt the grammar tests.** The criteria above require all three, so a trade-off counts as a failure and is reported as one.
