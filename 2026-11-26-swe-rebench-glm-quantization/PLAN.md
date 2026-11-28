# Study 3 plan: does a quantized open coding model hold up on fresh SWE tasks?

Status: **written 26 Nov 2026, before Stage 1. Stage 0 ran the same day (see `PILOT.md`).** Two stages. Stage 0 is a pilot. Stage 1 is the study itself.

**Public copy.** Stage 1 ran 26 to 28 Nov 2026. Its results are in `REPORT.md`, which lists every departure from this plan. Prices, budgets, run estimates, approval steps, provider and account details are removed from this copy. The question, arms, margin, outcomes and analysis are as written before Stage 1.

## Question

On one rented 8x B200 node, with the same agent, prompt and tasks, how much of GLM-5.3-Flash's SWE-rebench resolved rate survives FP8 and NVFP4 against BF16? And how much node time does a resolved task take, next to the closed models on the same leaderboard?

Why it is open:

- **Demand.** Software is 39.7% of spend on one large LLM API marketplace, and six of its top eight apps are coding or agent harnesses (week ending 24 Nov 2026). GLM-5.3-Flash is one of the open models every inference provider leads with. Its NVFP4 checkpoints have about 177k downloads in 30 days across NVIDIA, RedHat and one community repo, and the GGUF has 739k.
- **Parity exists.** On the newest SWE-rebench window (15 May to 1 Jul 2026, 111 tasks), the top open model GLM-5.2 resolves 62.9% against 64.5% for Claude Fable 5. At pass@5 it leads, with 81.1% against 78.4%. These are independent numbers from the leaderboard.
- **Nobody has measured the deployed version on fresh tasks.**
  - imec (Oct 2026) tested Qwen3.8-27B and GLM-5.3-Flash on a 64-task SWE-Bench-Pro subset and on DeepSWE. It found "no clear, impactful accuracy loss", ran single runs with high unexplained variance, and **skipped GLM-5.3-Flash BF16** because it did not fit on 4x H200.
  - NVIDIA's model card reports BF16 82.58 against NVFP4 82.02 on Terminal-Bench 2.1. That is a vendor number, with no runs or intervals.
  - We found no quantization test of any model on SWE-rebench.

**What would change the answer for a reader.** If NVFP4 loses more than about 5 points of resolved rate against BF16, teams serving coding agents should pay for FP8. If it does not, NVFP4 halves the memory and the node count.

## Pre-registered criteria (fixed before Stage 1 starts)

For each quantized arm against BF16 (arm A), take the paired difference in resolved rate. Each task is averaged over its 5 runs, and the 95% interval comes from a bootstrap over tasks (10,000 resamples, seed 0).

| Verdict | Rule |
|---|---|
| No measurable loss | The interval's lower bound is above -5 points |
| Loss | The interval's upper bound is below 0 |
| Inconclusive at this size | Anything else, reported as such and never as a win |

The 5-point margin is set by what 111 tasks can resolve. The pilot measures how much tasks vary from run to run, and if the expected interval is wider than ±5 points, the plan says so before Stage 1 runs.

## Arms (Stage 1)

| Arm | Checkpoint | Weights | Role |
|---|---|---|---|
| A | `zai-org/GLM-5.3-Flash-BF16` | BF16, about 642 GB | Reference |
| B | `zai-org/GLM-5.3-Flash` | FP8, official, about 306 GiB | Precision arm |
| C | `nvidia/GLM-5.3-Flash-NVFP4` | NVFP4 | Precision arm |
| D | `zai-org/GLM-5.2-FP8` | FP8, 744B total / 40B active | **Calibration**: the one model with a leaderboard score on this exact window (62.9 ± 1.19) |
| E (optional) | `nvidia/GLM-5.2-NVFP4` | NVFP4 | Extends the precision result to the anchor model, if time allows |

- **Model facts:** GLM-5.3-Flash is 320B total and 18B active, MIT-licensed, released 25 Aug 2026. GLM-5.2 is MIT.
- **How the NVFP4 checkpoint was picked:** C is NVIDIA's checkpoint, the one its card reports numbers for. `tattrongvu/GLM-5.3-Flash-NVFP4` has more downloads (110,749 against 48,532), but it is a community repack. This departs from study 1's rule (most downloaded), and the report says so.
- **One hardware type:** every arm runs on Blackwell, so precision is the only thing that changes. H200 cannot run NVFP4 natively.
- **Commit SHAs:** pinned at download and written to `raw/env/`.

DeepSeek V4.1 Flash is left out. Its official checkpoint already ships MXFP4 experts with no BF16 release, so a precision sweep does not apply to it.

## Fixed setup

- **Tasks:**
  - SWE-rebench, July 2026 window, via Harbor: `ibragim-badertdinov/swe-rebench-07-2026@2026-07-r4`.
  - 111 tasks: Go 21, Java 21, Python 20, Rust 25, TypeScript 24.
  - Three tasks need network access (`vize-1508`, `agents-5944`, `cli-printing-press-2002`). All 111 are run, and the 108 offline tasks are also reported on their own.
- **Agent:**
  - mini-swe-agent through Harbor, using SWE-rebench's own published tool-mode system prompt.
  - The leaderboard's scaffold is not public, so this is the closest open equivalent. Arm D measures the gap.
  - Context 128K tokens, as on the leaderboard. Step limit 150. Wall limit 60 min per task.
  - Both limits are checked in the pilot. How often an arm hits either limit is reported per arm.
- **Sampling:**
  - Thinking on, with the model card's recommended temperature and top-p, recorded in `raw/env/`.
  - 5 runs per task per arm, seeds 0 to 4. The leaderboard also uses 5 runs.
- **Engine:**
  - vLLM. GLM-5.3-Flash needs 0.29.0 or later, which is nightly-only today, so the exact wheel or image digest is pinned in the pilot.
  - TP8 with expert parallel for NVFP4, following the vLLM recipe. Prefix caching on, because agents reuse their history.
  - Bound to the Docker bridge address only, never to a public interface.
- **Concurrency:** the number of tasks in flight is set in the pilot from vCPU and KV headroom, about 48 expected. It is held the same for every arm.

## Measurements per arm

- Resolved rate and SEM across the 5 runs, the same way the leaderboard reports them, plus pass@5.
- The paired difference against arm A, as set out in the criteria.
- Breakdown by language, descriptive only.
- Per task: steps, input and output tokens, wall time, and whether it hit the step or wall limit.
- Wall hours per arm at the fixed concurrency, and GPU power sampled every second.
- **Node time per resolved task:** arm wall hours ÷ resolved tasks across the 5 runs.
- **Calibration rule for arm D:**
  - If our GLM-5.2 FP8 resolved rate is within 5 points of the leaderboard's 62.9, the report puts GLM-5.3-Flash next to the leaderboard's closed rows, with the scaffold caveat.
  - If not, it compares precisions only and makes no closed-model claim.

## Stage 0: pilot

- **Purpose:**
  - Prove the harness runs end to end against a local vLLM.
  - Measure minutes, steps and tokens per task, and Docker image disk use.
  - Find the concurrency the node can hold.
- **What it runs:** GLM-5.3-Flash NVFP4 on 2x B200 spot at TP2. NVFP4 is about 200 GB, so it fits in 384 GB. 20 tasks (4 per language), 1 run each.
- **Where:** Spheron, 2x B200 spot. The node has only 256 GB of local disk, so a 600 GB network SSD volume is attached in the same region.
- **Time:** about 2 h, with the auto-terminate timer set to 3 h.
- **Output:**
  - The Stage 1 time estimate, recomputed from measured rates.
  - The run-to-run spread, which tells whether ±5 points is reachable.
  - A first resolved count, a harness check rather than a result.
- **Stop rule:** if the harness cannot complete a task against the local endpoint within the first hour, stop, save logs and ask.

## Stage 1: compute

- **Node:** 8x B200 SXM6 spot on Spheron. It has 160 vCPU and 1,792 GB RAM.
- **Why spot:** it was the 8x B200 offer available when this plan was written.
- **Storage:** a 3 TB network SSD in the same region holds the weights (about 1.9 TB across A to D) and the task images (about 220 GB). A reclaim then loses only in-flight tasks: redeploy, reattach and resume. Harbor writes one result per trial, so finished trials are skipped.

- **Lighter variant:** arms A to C only, 3 runs each. It drops the calibration, so the report makes no claim against closed models.

## Safety

- **Network:** vLLM is never on a public interface.
- **Results:** copied off after each arm.
- **Shutdown:**
  - Terminate the node, then confirm that it shows terminated.
  - Delete the volume.
  - Report the actual node hours.
- **Secrets:** no tokens on the VM. All checkpoints and the Harbor dataset are public.

## Risks

- **The scaffold differs from the leaderboard's.** Arm D measures by how much. If the offset is large, the closed-model comparison is dropped rather than adjusted.
- **Contamination.** Every task in the window predates GLM-5.3-Flash's release (25 Aug 2026), so the leaderboard would flag its absolute score. This does not affect the precision comparison, which is the main claim, because all arms see the same tasks. The report states it next to any absolute number.
- **The closed bar has moved.** The leaderboard has not been updated since July, so Fable 5.2 and Opus 5.5 are not on it. The report says the closed rows are dated.
- **vLLM nightly.** A bug can masquerade as a precision effect. The pilot runs the NVFP4 arm first, and Stage 1 runs all arms on one pinned build.
- **Spot reclaim.** It costs time, not results, because of the volume and Harbor's per-trial resume.
- **Agent variance.** imec saw large unexplained swings on DeepSWE. Five runs and paired statistics are the answer; the pilot measures the spread before Stage 1.
- **Terminal-Bench 4.1 is not in Stage 1.** It has GPU tasks and 8-hour timeouts. It is a possible follow-up once this harness exists.

## What gets published

- **Results:** per-trial results, trajectories summary (steps, tokens, wall time, resolved), harness config and prompt, under CC BY 4.0.
- **Code:** the harness under MIT.
- **Raw logs:** trajectories are published only after a scan for local details and secrets. Agent runs record shell output.
