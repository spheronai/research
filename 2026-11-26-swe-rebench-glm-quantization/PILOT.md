# Study 3 pilot notes (Stage 0)

Run 26 Nov 2026, 04:06 to 05:23 UTC. GLM-5.3-Flash NVFP4 on 2x B200, 20 SWE-rebench tasks, 1 run each. A re-pilot of the 8 context-limited tasks with a SWE-rebench-style agent config ran 06:55 to 08:04 UTC, and a third run of the same tasks with a 600-second command timeout ran 08:47 to 09:52 UTC (see "Re-pilot" and "Third re-pilot" below). The resolved counts here are harness checks, not results: one run per task, a subset of the 111 tasks.

Public copy: the pilot's trial folders and raw files (`pilot/`, `repilot/`, `repilot-t600/`) are not included, and prices, provider and account details are removed.

## What ran

| Item | Value |
|---|---|
| Node | 2x B200 SXM6 spot, 60 vCPU, 340 GB RAM, 200 GB local disk |
| Weights | `nvidia/GLM-5.3-Flash-NVFP4`, commit `1d00738646cbb0c6d0a9eb403dac5f366f1473a7`, on a 400 GB shared NVMe volume |
| Engine | `vllm/vllm-openai:glm53-flash-x86_64-cu130`, digest `sha256:0674a3ea3c971706cea310bca71daadff68ea1717c758d2f54affa5def001b86`, reports `0.28.1rc1.dev580+g385dce36b` |
| Serving | TP2 with expert parallel, fp8 KV cache, 131,072 context, 32 max sequences, prefix caching on, GPU memory 0.85 |
| Harness | Harbor 0.27.0, mini-swe-agent 2.4.6, dataset `ibragim-badertdinov/swe-rebench-07-2026@2026-07-r4` |
| Tasks | 20, 4 per language (`raw/env/pilot_tasks.txt`), 10 at a time, 1 run each |

Files: `pilot/jobs/` (per-trial results and trajectories), `pilot/raw/pilot_trials.csv` (one row per trial), `pilot/raw/power.csv` (both GPUs every 5 s), `pilot/raw/vllm_metrics.txt` (engine counters every 30 s), `pilot/raw/env/` (scripts, patch, versions).

## Harness: works end to end

- **Tool calls parse.** vLLM's GLM tool-call and reasoning parsing (`--tool-call-parser glm47`, `--reasoning-parser glm45`) returns clean `tool_calls` (`raw/smoke.json`).
- **All 20 trials completed and were verified.** None failed in setup, so the stop rule did not trigger.
- **One fix was needed.** The SWE-rebench images set `XDG_DATA_HOME=/workspace/.local/share`, and some set `UV_OFFLINE=1`, so Harbor's mini-swe-agent install put the binary off `PATH` and failed with exit 127. A one-line patch to Harbor's install command (`raw/env/harbor-mini-swe-agent.patch`) exports `UV_TOOL_BIN_DIR` and `UV_OFFLINE=0`. Stage 1 must apply it to the pinned Harbor.

## Measured

| Per trial (n=20) | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| Trial wall, minutes | 17.8 | 15.6 | 5.4 | 36.8 |
| Agent time, minutes | 16.7 | 14.9 | 5.1 | 35.5 |
| Verifier time, minutes | 0.6 | 0.3 | 0.0 | 2.2 |
| Model calls | 110 | 114 | 44 | 173 |
| Input tokens | 7.35M | 7.98M | 1.38M | 13.4M |
| Output tokens | 35.3k | 34.6k | 9.6k | 57.2k |

- **Whole job:** 45 min 55 s wall for 20 trials at 10 concurrent. That is 356 trial-minutes, an average of 7.7 in flight, because the tail ran with only a few trials left.
- **The workload is prefill, not decode.** 147.0M prompt tokens against 0.71M generated, about 208 to 1. Prefix caching hit 95.4% of prompt tokens (140.2M of 147.0M), so the engine prefilled only about 6.8M tokens fresh.
- **GPU headroom:** average utilisation 71% and 511 W per GPU (peak 868 W), KV cache never above about 8% of 7.67M tokens. vLLM reported room for 58 concurrent 128K sequences on two GPUs.
- **CPU is set per task:** each task container is limited to 1 CPU by its `task.toml`, and Rust, Go and Java builds pinned that core at 100%. Host load stayed around 7 of 60 vCPUs.
- **Images:** about 3 GB each. Disk use on the root disk peaked at about 140 GB of 193 GB with 20 images cached.
- **Load time:** the NVFP4 weights (about 205 GB) took about 12 minutes from container start to ready, reading from the shared volume at about 250 MB/s. CUDA graph capture and FlashInfer autotuning added about 2 minutes.
- **Download:** 205 GB from Hugging Face to the volume in about 15 minutes, 200 to 320 MB/s.

### Outcomes

| Exit | Trials | Resolved |
|---|---:|---:|
| Submitted | 11 | 10 |
| Hit 128K context | 8 | 2 |
| Repeated format error | 1 | 0 |
| **Total** | **20** | **12** |

Two trials report `ApiRateLimitError`. Both are the same 400 context-length error, which litellm mislabels. Harbor ran the verifier on whatever the agent had changed when it stopped, which is why two context-limited trials still resolved.

## What has to change before Stage 1

1. **The context limit decides 40% of trials.** mini-swe-agent keeps the whole history, and 8 of 20 trials filled 128K, after 105 to 170 model calls. That is too large a share to leave as noise. If one precision is more verbose, it hits the wall sooner, and the resolved rate would partly measure history length rather than skill.

   What SWE-rebench itself does, from its July 2026 trajectories (`ibragim-bad/swe_rebench_07_2026_trajectories`) and its About page:
   - It does not compact history either. A run that fills 128K ends with `exit_context`, which happened in 13 of 416 rows in one shard checked, about 3%.
   - It has no low step cap. GLM-5.2's longest run took 214 steps, and other models went to 492. The plan's step limit of 150 would cut runs the leaderboard lets finish, so Stage 1 uses 250.
   - It does not run mini-swe-agent. Its own scaffold reads files through a 100-line window (`open`, `goto`, `scroll_*`) and edits with `edit`/`replace`, which keeps each observation small. That is the most likely reason its overflow rate is about 3% where ours is 40%.

   mini-swe-agent already cuts any output over 10,000 characters, so the overflow here comes from volume: 130 to 180 observations averaging 2,000 to 3,000 characters each.

   Proposed Stage 1 agent config, written as `repilot/swerebench_mini.yaml`:
   - SWE-rebench's production system and issue prompt, adapted. Their special commands section is dropped, because the commands do not exist here, and `submit` becomes mini's submit command.
   - Observations cut at 5,000 characters, standing in for their file window.
   - Step limit 250, 128K context, no compaction.
   - Context-limit rate and format-error rate reported per arm as secondary outcomes. A precision that breaks tool formatting more often is a real finding, not noise.

   The re-pilot ran this config on the 8 context-limited tasks (below). It helped, but not enough: 3 of the 7 finished trials still filled 128K. Shorter observations are not the main lever. Each observation came out about 18% smaller, but runs still took 130 to 160 steps, and much of that is polling. mini-swe-agent kills any shell command after 30 seconds, so agents push Rust, Go and Java builds into the background and poll them with `sleep 25; tail log`. That cost up to 30 steps per trial (median 18), and 69 commands still hit the limit. Stage 1 therefore adds one more change, tested in the third re-pilot:
   - Command timeout raised from 30 to 600 seconds (`environment.timeout: 600`), so a build runs in one step. Each task's own 50-minute agent limit still applies.

   It removed the polling: no command timed out, and `sleep` calls fell from 133 to 9 across the same tasks. It did not remove the overflow. The 3 Rust tasks still filled 128K, because a Rust build or test prints more per step than the other languages, and the history reached 460k to 500k characters either way. Stage 1 keeps the 600-second timeout and reports the context-limit rate per arm. Rust is 25 of the 111 tasks, so this caps the absolute resolved rate for every arm alike; it does not bias the comparison between precisions. Any further fix for Rust is undecided and untested.
2. **Pin the KV cache dtype across arms.** NVIDIA's recipe casts the KV cache to fp8, and the pilot followed it. If arm C keeps fp8 KV while A and B use BF16 KV, the comparison mixes two precision changes. All arms use the same KV dtype. The default should be the engine's auto (BF16), with NVFP4 plus fp8 KV as an optional extra row.
3. **Run-to-run spread was not measured.** The pilot ran one attempt per task, as the plan specified, so it cannot say how much tasks flip between runs. The best available estimate comes from the leaderboard: GLM-5.2's SEM of 1.19 points over 5 runs implies a per-run SD of about 2.7 points. If arms vary like that, the difference of two 5-run means has an SE of about 1.7 points, about ±3.4 at 95%, before pairing tightens it. So the ±5 point margin looks reachable, but it is an inference, not a measurement.
4. **Engine version.** The `glm53-flash` image reports `0.28.1rc1.dev580`, not 0.29.0 as the plan assumed. Stage 1 pins this digest for every arm.

## Stage 1 timing

555 rollouts per arm at a mean of 17.8 trial-minutes is about 9,880 trial-minutes. On 8x B200 at 48 concurrent, the load per GPU is close to the pilot's (12 in flight per two GPUs against 10), so per-rollout time should hold. The re-pilot's 8 longest tasks averaged 19.3 min, and with the 600-second command timeout the same 8 tasks averaged 19.4 min, so the timeout did not make trials longer.

## Re-pilot

The 8 context-limited tasks, run once with `repilot/swerebench_mini.yaml` on the same 2x B200 offer. Same image digest, Harbor 0.27.0 with the same patch, 8 at a time. Files: `repilot/jobs/`, `repilot/raw/repilot_trials.csv`, `repilot/raw/power.csv`, `repilot/raw/vllm_metrics.txt`, `repilot/raw/env/`.

### First attempt: stopped at 06:11 UTC, before any task ran

Loading the weights stalled at about 18 MB/s. After 18 minutes only 5 of 33 shards had loaded, so the run was stopped. The second attempt showed the volume was not the cause:
- A fresh volume in the same region read at 667 MB/s sequentially with 8 streams (`repilot/raw/env/volume-speed.txt`), yet vLLM's 64 loader threads still pulled only 18 to 29 MB/s from it. The slow part is how the loader reads a network filesystem, not the disk.
- Reading the shards first with 8 parallel sequential `dd`s put all 205 GB in page cache in 4 min 22 s, about 0.78 GB/s. vLLM then loaded from memory and was serving 10 minutes after its container started (12 in the first pilot).

For Stage 1:
- Warm the page cache with parallel sequential reads before each arm starts vLLM. An 8x B200 node has 1.8 TB of RAM, so even the 642 GB BF16 weights fit.
- Keep the sequential read check as a go/no-go, and recreate the volume if it reads under about 300 MB/s.

### Second attempt: 07:30 to 08:03 UTC

The first launch failed at once. Harbor refuses `--base-url` when a custom agent config is passed, and takes the endpoint from the config's `api_base` instead; ours already set it. With the flag dropped, the trial logs confirm the custom config loaded after `-c mini`, and that agents reached the local engine.

| Task | Language | Pilot | Re-pilot | Re-pilot calls | Pilot resolved | Re-pilot resolved |
|---|---|---|---|---:|---:|---:|
| fallow-762 | Rust | 128K context | 128K context | 158 | 0 | 1 |
| fallow-821 | Rust | 128K context | 128K context | 157 | 0 | 0 |
| kestra-16450 | Java | 128K context | Cut at step 201 | n/a | 0 | not verified |
| typescript-go-4194 | Go | 128K context | Submitted | 91 | 0 | 1 |
| waza-247 | Go | 128K context | Submitted | 136 | 1 | 1 |
| any-llm-1121 | Python | 128K context | Submitted | 128 | 0 | 1 |
| orca-4331 | TypeScript | 128K context | 128K context | 135 | 1 | 1 |
| vize-769 | Rust | 128K context | Repeated format error | 142 | 0 | 0 |

- **Resolved:** 5 of the 7 that finished, against 2 of 8 in the pilot. One run each, on tasks picked because they had failed, with the prompt changed as well. Treat it as a sign the harness is closer to the leaderboard's, not as an effect size.
- **Context limit:** 3 of 7 still filled 128K, against 8 of 8. Kestra had not hit it by step 201. It was stopped at 08:03 to leave time to copy results before the node's 08:15 auto-terminate, and its log and working-tree diff are kept in `kestra-*/cut/`.
- **Why the 5K cut was not enough:** on these tasks the mean observation fell from about 2,180 to about 1,790 characters, but each trial still built 450k to 500k characters of history over 130 to 160 steps. The command timeout is the larger lever (see "What has to change", item 1).
- **Per trial (n=7):** 19.3 trial-minutes, 135 model calls, 9.38M input and 38.9k output tokens on average.
- **Job:** 33 min wall. The engine saw 81.5M prompt tokens against 0.31M generated, with prefix caching hitting 95.7%. GPU utilisation averaged 46% at 415 W per GPU, lower than the pilot because agents spent much of the run polling builds.

## Third re-pilot: 600-second command timeout, 09:15 to 09:44 UTC

The same 8 tasks, run once with `repilot-t600/swerebench_mini.yaml`. It is the re-pilot config with one setting changed (`environment.timeout: 600`) and one line added to the prompt saying a command is stopped after 10 minutes. Same offer, image digest, Harbor patch and weights commit. Files: `repilot-t600/jobs/`, `repilot-t600/raw/t600_trials.csv`, `repilot-t600/raw/power.csv`, `repilot-t600/raw/vllm_metrics.txt`, `repilot-t600/raw/env/`.

Load: the volume read at 758 MB/s and wrote at 262 MB/s in the go/no-go check (`raw/env/volume-speed.txt`). The download took 17 minutes, the page-cache warm-up 3 min 19 s, and vLLM was serving 4 minutes after that.

| Task | Language | Re-pilot | Third re-pilot | Calls | Trial min | Resolved |
|---|---|---|---|---:|---:|---:|
| fallow-762 | Rust | 128K context | 128K context | 146 | 14.9 | 0 |
| fallow-821 | Rust | 128K context | 128K context | 192 | 28.8 | 0 |
| kestra-16450 | Java | Cut at step 201 | Submitted | 143 | 20.6 | 1 |
| typescript-go-4194 | Go | Submitted | Submitted | 82 | 24.6 | 1 |
| waza-247 | Go | Submitted | Submitted | 142 | 14.3 | 1 |
| any-llm-1121 | Python | Submitted | Submitted | 108 | 21.2 | 1 |
| orca-4331 | TypeScript | 128K context | Submitted | 115 | 17.7 | 1 |
| vize-769 | Rust | Repeated format error | 128K context | 131 | 13.4 | 0 |

| Same 8 tasks | Pilot | Re-pilot | Third re-pilot |
|---|---:|---:|---:|
| Resolved | 2 of 8 | 5 of 7 finished | 5 of 8 |
| Filled 128K | 8 of 8 | 3 of 7 finished | 3 of 8 |
| Commands that timed out | not counted | 69 | 0 |
| `sleep` calls | not counted | 133 | 9 |

- **The polling is gone.** Every non-Rust task submitted, kestra included, and fallow-821 is the only one with more than 2 `sleep` calls (7).
- **Rust still overflows.** All 3 Rust tasks filled 128K, at 131 to 192 calls. Their histories reached 463k to 498k characters; the 5 that submitted stopped at 385k to 434k. The fallow-762 resolve in the re-pilot came from its partial diff and did not repeat.
- **Same caveat as before:** one run each, on tasks picked because they failed. It shows the harness behaves, not an effect size.
- **Per trial (n=8):** 19.4 trial-minutes, 132 model calls, 8.82M input and 36.9k output tokens on average.
- **Job:** 28 min 46 s wall. The engine saw 70.6M prompt tokens against 0.30M generated, with prefix caching hitting 95.4%. GPU utilisation averaged 57% at 441 W per GPU (peak 841 W), up from 46% and 415 W, because agents now wait on the model instead of on `sleep`.

## Shutdown

All four nodes show `terminated`, at 05:23, 06:11, 08:04 and 09:52 UTC, and all four volumes were deleted.

## Deviations from the plan

- **Node:** the pilot ran on a 2x B200 spot node. Its inbound port 8000 was closed with iptables except from the Docker bridge and loopback.
- **Volume:** 400 GB, not 600 GB. Only the NVFP4 weights were needed.
- **Agent config:** mini-swe-agent's default prompt and limits, not SWE-rebench's prompt, step limit 150 and 60-minute wall. Harbor applied each task's own 50-minute agent timeout. No trial reached it.
- **vLLM version:** `0.28.1rc1.dev580`, see above.
- **Copying results:** results were copied off each node and the checksums matched.
- **Re-pilot timing:** kestra-16450 was cut before it finished, to stay inside the node's auto-terminate window.
