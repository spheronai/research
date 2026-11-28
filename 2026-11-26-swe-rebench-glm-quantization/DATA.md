# Data for study 3: GLM-5.3-Flash BF16 vs FP8 vs NVFP4 on SWE-rebench

Every CSV in `results/` is a copy of what `stage1/analysis/analyze.py` writes (run `python3 analysis/analyze.py` from `stage1/`) from the Harbor trial folders of each arm (`result.json`, the agent trajectory, `mini-swe-agent.txt`, `exception.txt` and the verifier's `test-stdout.txt`). It uses the Python standard library only. Trials in any `aborted/` folder (cut by a spot reclaim, or from the dropped arm D) are excluded.

Charts are drawn from the CSVs by `harness/charts.py` in the Spheron chart style (`harness/spheron_chart_style.py`). Run it from this folder with `uv run --with matplotlib python harness/charts.py`. Set `SPHERON_FONT_DIR` to a folder holding `SFPRODISPLAYREGULAR.otf` and `OCRAStd.otf` to render the brand faces; without it they fall back to system fonts.

## Arms

| Arm | Label | Checkpoint (commit) | KV cache |
|---|---|---|---|
| A | BF16 | `zai-org/GLM-5.3-Flash-BF16` (`f9fed3bbc822`) | BF16 |
| B | FP8 | `zai-org/GLM-5.3-Flash` (`46a050bec130`) | BF16 |
| C | NVFP4 | `nvidia/GLM-5.3-Flash-NVFP4` (`1d00738646cb`) | BF16 (forced; vLLM picks FP8 KV for this checkpoint by default) |

Arm D (`zai-org/GLM-5.2-FP8`, `f33c6dc501ee`) was started and dropped; none of its trials is in these files.

Scopes: `all_111` is every task. `offline_108` leaves out the three tasks that need network access (`vize-1508`, `agents-5944`, `cli-printing-press-2002`).

Rates and differences are fractions from 0 to 1 unless the column name says otherwise. Multiply by 100 for percent or points.

## results/trials_classified.csv

One row per scored trial: 1,665 rows, 555 per arm (111 tasks x 5 attempts).

| Column | Meaning |
|---|---|
| `arm` | A, B or C |
| `trial` | Harbor trial name: task id plus a random suffix |
| `task` | SWE-rebench task id |
| `language` | Task language: go, java, python, rust, typescript |
| `started_at` | Trial start, UTC, from `result.json` |
| `resolved` | 1 if the verifier's reward is 1, else 0. A trial with no reward counts as 0 |
| `reward_missing` | 1 if the trial has no verifier reward |
| `exit_status` | mini-swe-agent's own exit status (`Submitted`, `ContextWindowExceededError`, `LimitsExceeded`, `RepeatedFormatError` and so on). Empty if the agent did not record one |
| `exception` | Harbor's exception type for the trial, if any. Harbor mislabels the engine's 400 context-length error as `ApiRateLimitError` or `UnknownApiError`; `stop_reason` does not rely on this column for that |
| `stop_reason` | One bucket per trial, see below |
| `model_calls` | Model requests in the trajectory |
| `input_tokens` | Prompt tokens summed over all calls (history is resent each call) |
| `output_tokens` | Generated tokens summed over all calls, reasoning included |
| `trial_min` | Trial wall time, minutes, agent plus verifier |
| `tool_chars` | Characters of tool output in the final history |
| `assistant_chars` | Characters of model text in the final history, reasoning included |
| `reasoning_chars` | Characters of the model's reasoning in the final history |
| `tool_calls` | Tool calls in the trajectory |
| `repeated_calls` | Tool calls whose arguments are identical to the call before |
| `oom_signal` | 1 if the agent log or test output matches a kill or allocation failure (`Killed`, signal 9, exit 137, out of memory). A text match, not the kernel log |
| `attempt` | 0 to 4: order of the trial among its task's attempts in this arm, by `started_at` (0 is the first run) |

`stop_reason` values, checked in this order:

| Value | Rule |
|---|---|
| `context_limit` | Exit status or log tail shows the 128K context error |
| `time_limit` | Harbor's `AgentTimeoutError` (the task's 50-minute agent limit) |
| `step_limit` | Exit status `LimitsExceeded` (250 steps) |
| `format_errors` | Exit status `RepeatedFormatError` |
| `submitted` | The agent submitted |
| `verifier_error` | The agent submitted but Harbor reports `RewardFileNotFoundError` |
| `auth_error` | Harbor's `AgentAuthenticationError`, when none of the above applies |
| `safety_refusal` | Harbor's `AgentSafetyRefusalError` |
| `endpoint_error` | Harbor's `NetworkConnectionError`, `ApiRateLimitError` or `UnknownApiError` |
| `other_agent_exit` | Anything else, mostly a non-zero agent exit |

`auth_error` and `safety_refusal` are zero in every arm: the two trials Harbor labelled `AgentAuthenticationError` had already filled the context window and count as `context_limit`.

## results/arms.csv

One row per scope and arm.

| Column | Meaning |
|---|---|
| `scope` | `all_111` or `offline_108` |
| `arm`, `precision` | Arm letter and label |
| `trials` | Scored trials |
| `resolved` | Resolved trials |
| `resolved_rate` | `resolved / trials` |
| `sem` | Sample standard deviation of the 5 per-run rates divided by the square root of 5 |
| `run_rates` | The 5 per-run resolved rates, space-separated, attempt 0 to 4 |
| `pass_at_5` | Share of tasks resolved in at least one attempt |
| `resolved_all_5` | Share of tasks resolved in all five attempts |

## results/paired_vs_A.csv

The pre-registered comparison. One row per scope and quantized arm.

| Column | Meaning |
|---|---|
| `scope`, `arm`, `precision` | As above |
| `vs` | Reference arm: A (BF16) |
| `tasks` | Tasks compared |
| `diff` | Mean over tasks of (this arm's per-task resolved share minus A's) |
| `ci95_lo`, `ci95_hi` | 95% percentile interval from 10,000 bootstrap resamples over tasks, seed 0 |
| `verdict` | `no measurable loss` if `ci95_lo` > -0.05, `loss` if `ci95_hi` < 0, else `inconclusive at this size` |

## results/paired_vs_A_without_oom.csv

Same as `paired_vs_A.csv` for `all_111`, after removing every trial with `oom_signal = 1`. Task shares are over the trials left; a task with no trials left in one of the two arms drops out, which is why `tasks` is under 111.

## results/oom_signal.csv

| Column | Meaning |
|---|---|
| `arm` | A, B or C |
| `trials_with_oom_signal` | Trials with `oom_signal = 1` |
| `resolved_among_them` | How many of those resolved |

## results/stop_reasons.csv

One row per `stop_reason`. For each arm, `<arm>_trials` is how many trials ended that way and `<arm>_resolved` how many of those still resolved.

## results/context_probe.csv

What the context-limited trials looked like, per arm. "At hit" means over trials whose `stop_reason` is `context_limit`.

| Column | Meaning |
|---|---|
| `context_trials` | Trials that filled 128K |
| `tasks_with_any_context_hit` | Tasks with at least one such trial |
| `median_calls_at_hit` | Median `model_calls` at the hit |
| `median_tool_chars_at_hit`, `median_assistant_chars_at_hit`, `median_reasoning_chars_at_hit` | Median characters of tool output, model text and reasoning in the history at the hit |
| `median_repeated_calls_at_hit` | Median `repeated_calls` at the hit |
| `median_calls_other` | Median `model_calls` over all other trials |
| `trials_at_150_calls` | Trials in the arm with 150 or more model calls |
| `format_error_exits` | Trials with `stop_reason = format_errors` |
| `context_hit_resolved_share` | Share of context-limited trials that still resolved |
| `median_output_tokens_all` | Median `output_tokens` per trial, all trials |
| `median_output_tokens_per_call_all` | Median of `output_tokens / model_calls`, all trials |

## results/by_language.csv

| Column | Meaning |
|---|---|
| `language` | Task language |
| `tasks` | Tasks in that language |
| `A_rate`, `B_rate`, `C_rate` | Resolved rate per arm over that language's trials |

## results/throughput.csv

| Column | Meaning |
|---|---|
| `arm` | A, B or C |
| `trial_hours` | Sum of `trial_min` over the arm, in hours |
| `node_hours_at_48` | `trial_hours / 48`, the trials kept in flight. Used because spot reclaims split arms across nodes |
| `resolved` | Resolved trials |
| `node_minutes_per_resolved` | `node_hours_at_48 x 60 / resolved` |
| `median_trial_min` | Median `trial_min` |
