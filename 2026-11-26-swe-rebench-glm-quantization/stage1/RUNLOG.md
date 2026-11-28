# Stage 1 run log

Notes kept during the run, for the report's deviations section. All times UTC.

## Nodes

| Node | Ran | Ended by |
|---|---|---|
| 1 | 26 Nov 11:10 to 14:47 | spot reclaim during arm C |
| 2 | 26 Nov 15:16 to 22:04 | spot reclaim during arm B |
| 3 | 27 Nov 02:42 to 06:24 | spot reclaim during arm B |
| 4 | 27 Nov 07:27, never finished booting | spot reclaim before start |
| 5 | 27 Nov 07:43 to 11:12 | spot reclaim during arm B |
| 6 | 27 Nov 12:24 to 18:42 | spot reclaim during arm A |
| 7 | 27 Nov 18:46 to 28 Nov 01:11 | spot reclaim during arm D |
| 8 | 28 Nov 02:05 to 03:47 | stopped during arm D |

Nodes 1 and 2 are covered by `raw/env/setup2.sh` and `raw/env/quarantine_node2.sh`.

## Node 3 reclaim, 27 Nov

- 317 arm B trials had a `result.json` on the volume. Every trial that finished at or before 06:22:20 has a normal verifier result.
- From 06:23:11 to 06:23:47, 59 trials were written up as failures with no reward (agent exit 255, rate-limit and network errors from the dying vLLM, then docker compose errors). That is the reclaim, not the model.
- `raw/env/node5.sh` moved those 59 to `aborted/stage1-B-node3-reclaim/` so `harbor jobs resume` reruns them, leaving 258 arm B trials. Exclude the aborted folder from the analysis, as with node 2's.
- The task images were meant to be saved to the volume after node 2, but `/mnt/vol/images/` was empty, so node 5 pulled all 111 again. It took 12 minutes with no Docker Hub throttling.
- Node 5: setup done 08:05, vLLM ready 08:13, arm B resumed.

## Node 5 reclaim, 27 Nov

- Reclaimed at 11:12:49, still in arm B. At 10:50 arm B had 474 of 555 results (300 resolved), every one with a verifier reward.
- 69 container-level OOM kills on node 5 before the reclaim, all inside task containers (mostly `rustc` and `rust-lld`, some `tursodb` and the agent itself). Host memory stayed at about 515 of 1,763 GB used.
- One arm B trial, `HKUDS__nanobot-4274`, ended with `AgentAuthenticationError` and no reward. Count it as unresolved and list it.
- Node 6 deployed at 12:24.
- On the volume every arm B trial had a `result.json` (555). The last normal one finished at 11:10:32; all 48 from 11:11:30 to 11:11:36 have no reward (network, rate-limit and exit-code errors from the dying vLLM). `raw/env/node6.sh` with `CUTOFF=2026-11-27T11:11:00` moved those 48 to `aborted/stage1-B-node5-reclaim/`, leaving 507.
- Because harbor saw all 555 as finished, node 5's chain logged `HARBOR_EXIT 0 11:11:44` and `ARM_DONE B 11:11:54`, then began arm A's warm-up before the reclaim at 11:12:49 (no `jobs/stage1-A` was created). The `ARM_DONE B` line in `logs/arm-B.out` was renamed `ARM_VOID_B` before node 6's chain started, so arm B resumes instead of being skipped. `logs/summary-B.txt` from node 5 is stale and is rewritten when B finishes.

## Arm B done, 27 Nov

- Node 6: setup done 12:45, vLLM ready 12:53, the 48 reruns finished and harbor exited at 14:01. `ARM_DONE B 14:01:23`.
- `summary-B.txt`: 555 trials, 351 resolved (63.2%), 147 hit context. Arm C for reference: 364 of 555 (65.6%).
- Results copied off the node and checked.
- Arm A vLLM ready at 14:07:19.

## Node 6 reclaim, 27 Nov

- Arm A ran at about 90 trials an hour on node 6, every result with a verifier reward. At 18:11 it had 356 results (222 resolved, 62.4%).
- Reclaimed at 18:42:51, still in arm A. Node 7 was deployed at 18:46 from the same spot offer.
- On the volume arm A had 492 trial folders and 457 `result.json` files. The last normal result finished at 18:41:12; all 62 from 18:41:15 to 18:41:51 have no reward (agent timeout, authentication, rate-limit, network and exit-code errors, then `RuntimeError`s from the dying vLLM). `raw/env/node7.sh` with `ARM=A CUTOFF=2026-11-27T18:41:13` moved those 62 to `aborted/stage1-A-node6-reclaim/`, leaving 395 (243 resolved).
- Node 6's chain had not logged `HARBOR_EXIT` or `ARM_DONE A`, so there was nothing to void; `node7.sh` checks for it anyway.
- Node 7: quarantine done 18:52:58, setup (images pulled again) done 19:09:29, chain restarted, arm A vLLM ready 19:15:27 and harbor resumed the remaining 160 trials.

## Arm A done, 27 Nov

- `ARM_DONE A 21:23:48` on node 7, every result with a verifier reward.
- `summary-A.txt`: 555 trials, 349 resolved (62.9%), 64 hit context. For reference: arm B 351 (63.2%), arm C 364 (65.6%).
- 11 container-level OOM kills on node 7 by 20:12, all inside task containers. Host memory stayed at about 500 of 1,763 GB used.
- Results copied off the node and checked.
- Arm D vLLM loaded the GLM-5.2 FP8 weights at 21:27:23 (89.33 GiB per GPU, TP8 with expert parallel), so the architecture is supported by this vLLM build.

## Arm D at 48 concurrent, and node 7 reclaim, 27 to 28 Nov

- Arm D vLLM ready 21:33:00, harbor started at 48 concurrent as for arms A to C.
- By 22:35, 50 trials had finished: 48 with `Agent execution timed out after 3000.0 seconds` (each task's own 50-minute agent limit), 2 normal, 4 resolved. The timed-out trials had made a median of 44 agent steps (11 to 66). Arm A's trials finish in a median of about 68. vLLM's KV cache sat at 95-99% with 7 to 14 requests waiting and about 160 to 410 generated tokens/s across all requests.
- At that rate arm D would have taken about 11 h and scored far below the leaderboard's 62.9 because of throughput, not model quality, so it could not serve as the calibration arm.
- On 28 Nov arm D was restarted at **16 concurrent for 2 attempts per task (222 trials)**. This is a deviation: arms A to C ran at 48 concurrent with 5 attempts. The plan's calibration rule (within 5 points of 62.9) is applied to the 2-attempt mean, with its wider interval reported.
- Node 7 was reclaimed at 01:11:44 on 28 Nov, still running arm D at 48. Node 8 deployed at 02:05 from the same spot offer.
- `raw/env/node8.sh` moves every arm D trial from the 48-concurrent run (finished or in flight), with that run's `raw/D` files and arm D logs, to `aborted/stage1-D-c48/`. None of it is used in the analysis. `raw/env/arm.sh` now sets `-n 16 -k 2` for arm D only.
- On node 8 it moved 295 arm D trial folders (274 with a result at the reclaim) at 02:11:40, then setup began. Setup done 02:28:14, arm D vLLM ready 02:37:02, harbor started as `-n 16 -k 2`.

## Arm D at 16 concurrent, stopped, 28 Nov

- 16 concurrent did not fix it. Of the first 20 trials to finish (02:45 to 03:43), 9 hit the 3000 s agent limit and 9 resolved.
- The cause is memory, not concurrency. vLLM gave GLM-5.2 FP8 a KV cache of 670,208 tokens on 8x B200 (5.11 full 128K contexts), against 6,427,427 for arm A's GLM-5.3-Flash BF16. With fewer than 6 long histories fitting at once, vLLM evicts and re-prefills them, so long trials run out of time regardless of how many run in parallel.
- `./stop_arm.sh D c16-b200` at 03:43:05 moved the run to `aborted/stage1-D-c16-b200/` (36 trial folders, 31 results, 11 of them cancelled by the stop). None of it is used in the analysis.
- Node 8 was stopped. Dropping arm D drops only the leaderboard calibration and the closed-model comparison; arms A to C do not depend on it.
- Before termination the rest of the volume (scripts, logs, tasks, raw files and every set-aside trial) was copied off the node and checked.
- Node 8 terminated at 03:47:50.
- On 28 Nov arm D was **dropped**. Stage 1 reports arms A to C only. The leaderboard calibration and the comparison with closed models on the leaderboard are left out, and the report says why: on 8x B200, GLM-5.2 FP8 leaves too little KV cache to run this agent within each task's time limit, so its score here would measure memory, not the model.
- The volume (4 TB network SSD) was deleted at about 03:50 on 28 Nov, after the copies above were checked. No Stage 1 nodes or volumes remain.

## Stage 1 node time

- Nodes 1 to 8 ran about 32 h in all. The volume was attached for about 40.7 h, 26 Nov 11:10 to 28 Nov 03:50.
- Six spot reclaims re-ran in-flight trials and re-pulled the task images each time.
