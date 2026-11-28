"""Stage 1 analysis, arms A to C, as pre-registered in PLAN.md.

Reads each arm's trials from node-copy/<arm>/jobs/stage1-<arm>/ (trials in aborted/ folders are never read)
and writes CSVs plus summary.md next to this file. Standard library only, so it runs with any python3:

    python3 analysis/analyze.py            (from stage1/)

Arm D was dropped (RUNLOG.md, 28 Nov), so there is no calibration and no closed-model comparison.
"""
import collections
import csv
import json
import math
import os
import random
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE1 = os.path.dirname(HERE)
ARMS = {"A": "BF16", "B": "FP8", "C": "NVFP4"}
REF = "A"
RUNS = 5
BOOT = 10_000
MARGIN = -0.05
CONCURRENCY = 48
NETWORK_TASKS = ("vize-1508", "agents-5944", "cli-printing-press-2002")

CTX_RE = re.compile(r"maximum context length is \d+ tokens", re.I)
OOM_RE = re.compile(
    r"(^|\s)Killed(\s|$)|signal: 9|SIGKILL|exit code 137|returned non-zero exit status 137|"
    r"out of memory|Cannot allocate memory|std::bad_alloc|memory allocation of \d+ bytes failed",
    re.I | re.M,
)


def read(path, limit=None):
    try:
        with open(path, errors="replace") as f:
            return f.read() if limit is None else f.read()[-limit:]
    except OSError:
        return ""


def lang_map():
    m = {}
    for line in open(os.path.join(STAGE1, "node-copy", "rest", "lang.txt")):
        p = line.split()
        if len(p) >= 2:
            m[p[1]] = p[0]
    return m


def classify(exit_status, exc, tail):
    """Why a trial's agent stopped, from the agent's own log first and Harbor's label second.
    Harbor labels some context-length 400s from vLLM as ApiRateLimitError or UnknownApiError."""
    if "ContextWindowExceeded" in exit_status or CTX_RE.search(tail):
        return "context_limit"
    if exc == "AgentTimeoutError":
        return "time_limit"
    if exit_status == "LimitsExceeded":
        return "step_limit"
    if exit_status == "RepeatedFormatError":
        return "format_errors"
    if exit_status == "Submitted" and not exc:
        return "submitted"
    if exit_status == "Submitted" and exc == "RewardFileNotFoundError":
        return "verifier_error"
    if exc == "AgentAuthenticationError":
        return "auth_error"
    if exc == "AgentSafetyRefusalError":
        return "safety_refusal"
    if exc in ("NetworkConnectionError", "ApiRateLimitError", "UnknownApiError"):
        return "endpoint_error"
    return "other_agent_exit"


def load_arm(arm, langs):
    base = os.path.join(STAGE1, "node-copy", arm, "jobs", f"stage1-{arm}")
    trials = []
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        rf = os.path.join(d, "result.json")
        if "__" not in name or not os.path.exists(rf):
            continue
        r = json.load(open(rf))
        task = r["task_name"].split("/")[-1]
        vr = r.get("verifier_result") or {}
        reward = (vr.get("rewards") or {}).get("reward")
        exc = (r.get("exception_info") or {}).get("exception_type") or ""
        tj = os.path.join(d, "agent", "mini-swe-agent.trajectory.json")
        info, msgs = {}, []
        if os.path.exists(tj):
            try:
                j = json.load(open(tj))
                info, msgs = j.get("info") or {}, j.get("messages") or []
            except ValueError:
                pass
        exit_status = info.get("exit_status") or ""
        tail = read(os.path.join(d, "agent", "mini-swe-agent.txt"), 4000) + read(os.path.join(d, "exception.txt"), 4000)
        agent_txt = read(os.path.join(d, "agent", "mini-swe-agent.txt"))
        test_txt = read(os.path.join(d, "verifier", "test-stdout.txt"))
        # Context probe: where the 128K went.
        tool_chars = sum(len(str(m.get("content") or "")) for m in msgs if m.get("role") == "tool")
        asst = [m for m in msgs if m.get("role") == "assistant"]
        asst_chars = sum(len(str(m.get("content") or "")) + len(str(m.get("reasoning_content") or "")) for m in asst)
        reason_chars = sum(len(str(m.get("reasoning_content") or "")) for m in asst)
        cmds = []
        for m in asst:
            for tc in m.get("tool_calls") or []:
                cmds.append(json.dumps((tc.get("function") or {}).get("arguments"), sort_keys=True))
        repeats = sum(1 for i in range(1, len(cmds)) if cmds[i] == cmds[i - 1])
        ar = r.get("agent_result") or {}
        trials.append(
            dict(
                arm=arm,
                trial=name,
                task=task,
                language=langs.get(task, ""),
                started_at=r.get("started_at") or "",
                resolved=int(reward == 1.0),
                reward_missing=int(reward is None),
                exit_status=exit_status,
                exception=exc,
                stop_reason=classify(exit_status, exc, tail),
                model_calls=(info.get("model_stats") or {}).get("api_calls") or 0,
                input_tokens=ar.get("n_input_tokens") or 0,
                output_tokens=ar.get("n_output_tokens") or 0,
                trial_min=minutes(r.get("started_at"), r.get("finished_at")),
                tool_chars=tool_chars,
                assistant_chars=asst_chars,
                reasoning_chars=reason_chars,
                tool_calls=len(cmds),
                repeated_calls=repeats,
                oom_signal=int(bool(OOM_RE.search(agent_txt) or OOM_RE.search(test_txt))),
            )
        )
    # Attempt index 0..4 by start time within a task (harbor -k 5 does not record a seed).
    by_task = collections.defaultdict(list)
    for t in trials:
        by_task[t["task"]].append(t)
    for ts in by_task.values():
        for i, t in enumerate(sorted(ts, key=lambda x: x["started_at"])):
            t["attempt"] = i
    return trials


def minutes(a, b):
    if not a or not b:
        return 0.0
    from datetime import datetime

    f = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
    return (f(b) - f(a)).total_seconds() / 60


def task_means(trials, keep=lambda t: True):
    acc = collections.defaultdict(list)
    for t in trials:
        if keep(t):
            acc[t["task"]].append(t["resolved"])
    return {k: sum(v) / len(v) for k, v in acc.items() if v}


def arm_stats(trials, tasks):
    ts = [t for t in trials if t["task"] in tasks]
    runs = []
    for i in range(RUNS):
        rr = [t["resolved"] for t in ts if t["attempt"] == i]
        runs.append(sum(rr) / len(rr))
    tm = task_means(ts)
    return dict(
        trials=len(ts),
        resolved=sum(t["resolved"] for t in ts),
        rate=sum(t["resolved"] for t in ts) / len(ts),
        sem=st.stdev(runs) / math.sqrt(RUNS),
        run_rates=runs,
        pass_at_5=sum(1 for v in tm.values() if v > 0) / len(tm),
        pass_all_5=sum(1 for v in tm.values() if v == 1) / len(tm),
    )


def boot_ci(diffs, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOT))
    return means[int(0.025 * BOOT)], means[int(0.975 * BOOT) - 1]


def verdict(lo, hi):
    if lo > MARGIN:
        return "no measurable loss"
    if hi < 0:
        return "loss"
    return "inconclusive at this size"


def paired(ref, arm, keep=lambda t: True):
    a, b = task_means(ref, keep), task_means(arm, keep)
    tasks = sorted(set(a) & set(b))
    diffs = [b[k] - a[k] for k in tasks]
    lo, hi = boot_ci(diffs)
    return dict(tasks=len(tasks), diff=sum(diffs) / len(diffs), lo=lo, hi=hi, verdict=verdict(lo, hi))


def write_csv(name, rows):
    if not rows:
        return
    with open(os.path.join(HERE, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def main():
    langs = lang_map()
    data = {arm: load_arm(arm, langs) for arm in ARMS}
    for arm, ts in data.items():
        tasks = collections.Counter(t["task"] for t in ts)
        assert len(tasks) == 111 and set(tasks.values()) == {RUNS}, (arm, len(tasks), set(tasks.values()))
    all_tasks = set(t["task"] for t in data[REF])
    offline = {k for k in all_tasks if not any(n in k for n in NETWORK_TASKS)}
    assert len(offline) == 108, len(offline)

    write_csv("trials_classified.csv", [t for arm in ARMS for t in data[arm]])

    arms_rows, paired_rows = [], []
    for scope, tasks in (("all_111", all_tasks), ("offline_108", offline)):
        for arm, label in ARMS.items():
            s = arm_stats(data[arm], tasks)
            arms_rows.append(
                dict(
                    scope=scope, arm=arm, precision=label, trials=s["trials"], resolved=s["resolved"],
                    resolved_rate=round(s["rate"], 4), sem=round(s["sem"], 4),
                    run_rates=" ".join(f"{x:.4f}" for x in s["run_rates"]),
                    pass_at_5=round(s["pass_at_5"], 4), resolved_all_5=round(s["pass_all_5"], 4),
                )
            )
            if arm != REF:
                p = paired(
                    [t for t in data[REF] if t["task"] in tasks], [t for t in data[arm] if t["task"] in tasks]
                )
                paired_rows.append(
                    dict(scope=scope, arm=arm, precision=label, vs="A (BF16)", tasks=p["tasks"],
                         diff=round(p["diff"], 4), ci95_lo=round(p["lo"], 4), ci95_hi=round(p["hi"], 4),
                         verdict=p["verdict"])
                )
    write_csv("arms.csv", arms_rows)
    write_csv("paired_vs_A.csv", paired_rows)

    # By language, descriptive only.
    lang_rows = []
    for lang in sorted(set(langs.values())):
        tasks = {k for k in all_tasks if langs.get(k) == lang}
        row = dict(language=lang, tasks=len(tasks))
        for arm in ARMS:
            ts = [t for t in data[arm] if t["task"] in tasks]
            row[f"{arm}_rate"] = round(sum(t["resolved"] for t in ts) / len(ts), 4)
        lang_rows.append(row)
    write_csv("by_language.csv", lang_rows)

    # Why the agent stopped, by actual message, with the resolved share in each bucket.
    reasons = ["submitted", "format_errors", "context_limit", "time_limit", "step_limit", "endpoint_error",
               "other_agent_exit", "auth_error", "safety_refusal", "verifier_error"]
    stop_rows = []
    for reason in reasons:
        row = dict(stop_reason=reason)
        for arm in ARMS:
            ts = [t for t in data[arm] if t["stop_reason"] == reason]
            row[f"{arm}_trials"] = len(ts)
            row[f"{arm}_resolved"] = sum(t["resolved"] for t in ts)
        stop_rows.append(row)
    write_csv("stop_reasons.csv", stop_rows)
    harbor_vs_actual = collections.Counter(
        (t["arm"], t["exception"]) for arm in ARMS for t in data[arm]
        if t["stop_reason"] == "context_limit" and "ContextWindowExceeded" not in t["exit_status"]
    )

    # OOM sensitivity: drop trials whose agent log or test output shows a kill or allocation failure.
    oom_rows = []
    for arm in ARMS:
        n = sum(t["oom_signal"] for t in data[arm])
        oom_rows.append(dict(arm=arm, trials_with_oom_signal=n,
                             resolved_among_them=sum(t["resolved"] for t in data[arm] if t["oom_signal"])))
    keep = lambda t: not t["oom_signal"]
    oom_paired = []
    for arm in ARMS:
        if arm == REF:
            continue
        p = paired(data[REF], data[arm], keep)
        oom_paired.append(dict(arm=arm, precision=ARMS[arm], tasks=p["tasks"], diff=round(p["diff"], 4),
                               ci95_lo=round(p["lo"], 4), ci95_hi=round(p["hi"], 4), verdict=p["verdict"]))
    write_csv("oom_signal.csv", oom_rows)
    write_csv("paired_vs_A_without_oom.csv", oom_paired)

    # Context probe: what filled the window in trials that hit it, and how often each arm hits it per task.
    probe_rows = []
    for arm in ARMS:
        ctx = [t for t in data[arm] if t["stop_reason"] == "context_limit"]
        rest = [t for t in data[arm] if t["stop_reason"] != "context_limit"]
        med = lambda xs, k: round(st.median(x[k] for x in xs), 1) if xs else ""
        probe_rows.append(dict(
            arm=arm, context_trials=len(ctx),
            tasks_with_any_context_hit=len({t["task"] for t in ctx}),
            median_calls_at_hit=med(ctx, "model_calls"),
            median_tool_chars_at_hit=med(ctx, "tool_chars"),
            median_assistant_chars_at_hit=med(ctx, "assistant_chars"),
            median_reasoning_chars_at_hit=med(ctx, "reasoning_chars"),
            median_repeated_calls_at_hit=med(ctx, "repeated_calls"),
            median_calls_other=med(rest, "model_calls"),
            trials_at_150_calls=sum(1 for t in data[arm] if t["model_calls"] >= 150),
            format_error_exits=sum(1 for t in data[arm] if t["stop_reason"] == "format_errors"),
            context_hit_resolved_share=round(sum(t["resolved"] for t in ctx) / len(ctx), 4) if ctx else "",
            median_output_tokens_all=round(st.median(t["output_tokens"] for t in data[arm]), 1),
            median_output_tokens_per_call_all=round(st.median(
                t["output_tokens"] / t["model_calls"] for t in data[arm] if t["model_calls"]), 1),
        ))
    write_csv("context_probe.csv", probe_rows)
    ctx_task = lambda arm: collections.Counter(t["task"] for t in data[arm] if t["stop_reason"] == "context_limit")
    a_ctx, b_ctx = ctx_task("A"), ctx_task("B")
    shift = sorted(((b_ctx[k] - a_ctx[k], k) for k in all_tasks), reverse=True)
    paired_ctx_tasks = [k for k in all_tasks]
    ctx_diff = [b_ctx[k] / RUNS - a_ctx[k] / RUNS for k in paired_ctx_tasks]
    ctx_lo, ctx_hi = boot_ci(ctx_diff)

    # Throughput: trial hours, and node hours at 48 in flight (the arms ran across reclaims, so wall clock is not comparable).
    thr_rows = []
    for arm in ARMS:
        th = sum(t["trial_min"] for t in data[arm]) / 60
        res = sum(t["resolved"] for t in data[arm])
        thr_rows.append(dict(arm=arm, trial_hours=round(th, 1), node_hours_at_48=round(th / CONCURRENCY, 2),
                             resolved=res, node_minutes_per_resolved=round(th / CONCURRENCY * 60 / res, 2),
                             median_trial_min=round(st.median(t["trial_min"] for t in data[arm]), 1)))
    write_csv("throughput.csv", thr_rows)

    unresolved_odd = [t for arm in ARMS for t in data[arm] if t["stop_reason"] in ("auth_error", "verifier_error")]

    # summary.md
    L = []
    L.append("# Stage 1 results, arms A to C\n")
    L.append("Generated by `analysis/analyze.py` from the trials in `node-copy/<arm>/jobs/`. Trials in any `aborted/` folder are excluded. Arm D was dropped (see `RUNLOG.md`), so this has no leaderboard calibration and no comparison with closed models.\n")
    L.append("Every task in this window predates GLM-5.3-Flash's release (25 Aug 2026), so its absolute scores may be inflated by contamination. That does not affect the precision comparison, the main result, because all three arms ran the same tasks with the same agent.\n")
    L.append("## Resolved rate\n")
    L.append("| Arm | Precision | Resolved | Rate | SEM (5 runs) | pass@5 | Resolved in all 5 |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for r in arms_rows:
        if r["scope"] == "all_111":
            L.append(f"| {r['arm']} | {r['precision']} | {r['resolved']} / {r['trials']} | {pct(r['resolved_rate'])}% | {pct(r['sem'])} | {pct(r['pass_at_5'])}% | {pct(r['resolved_all_5'])}% |")
    L.append("\nRuns are the 1st to 5th attempt at each task by start time. Harbor's `-k 5` does not record a seed.\n")
    L.append("## Pre-registered verdict: paired difference against BF16\n")
    L.append("Per task mean over 5 runs, quantized minus BF16, 95% interval from 10,000 bootstrap resamples over tasks (seed 0). No measurable loss if the lower bound is above -5 points; loss if the upper bound is below 0.\n")
    L.append("| Scope | Arm | Difference | 95% interval | Verdict |")
    L.append("|---|---|---:|---|---|")
    for r in paired_rows:
        L.append(f"| {r['scope'].replace('_', ' ')} | {r['arm']} ({r['precision']}) | {r['diff'] * 100:+.1f} pts | {r['ci95_lo'] * 100:+.1f} to {r['ci95_hi'] * 100:+.1f} | {r['verdict']} |")
    L.append("\n## By language (descriptive only)\n")
    L.append("| Language | Tasks | A | B | C |")
    L.append("|---|---:|---:|---:|---:|")
    for r in lang_rows:
        L.append(f"| {r['language']} | {r['tasks']} | {pct(r['A_rate'])}% | {pct(r['B_rate'])}% | {pct(r['C_rate'])}% |")
    L.append("\n## Why the agent stopped (from each trial's own log)\n")
    L.append("Trials, with how many of them still resolved: the verifier scores the repository as the agent left it, so a trial that ran out of context or time can still pass.\n")
    L.append("| Stop reason | A | B | C |")
    L.append("|---|---:|---:|---:|")
    for r in stop_rows:
        if any(r[f"{a}_trials"] for a in ARMS):
            L.append(f"| {r['stop_reason'].replace('_', ' ')} | " + " | ".join(f"{r[f'{a}_trials']} ({r[f'{a}_resolved']})" for a in ARMS) + " |")
    L.append("")
    if harbor_vs_actual:
        L.append("Harbor's own label is wrong for some context hits: vLLM's \"maximum context length is 131072 tokens\" 400 reached Harbor as " +
                 ", ".join(f"{v} `{exc or 'no exception'}` in arm {arm}" for (arm, exc), v in sorted(harbor_vs_actual.items())) +
                 ". They are counted as context hits above.\n")
    if unresolved_odd:
        L.append("Counted as unresolved, listed per the plan: " + ", ".join(f"`{t['trial']}` (arm {t['arm']}, {t['stop_reason'].replace('_', ' ')})" for t in unresolved_odd) + ".\n")
    L.append("## Context limit: why FP8 hits it more\n")
    L.append("| Arm | Context hits | Tasks with a hit | Median calls at hit | Median tool output chars at hit | Median model chars at hit | Median reasoning chars at hit | Median repeated calls at hit | Median output tokens per call (all trials) |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in probe_rows:
        L.append(f"| {r['arm']} | {r['context_trials']} | {r['tasks_with_any_context_hit']} | {r['median_calls_at_hit']} | {r['median_tool_chars_at_hit']} | {r['median_assistant_chars_at_hit']} | {r['median_reasoning_chars_at_hit']} | {r['median_repeated_calls_at_hit']} | {r['median_output_tokens_per_call_all']} |")
    L.append(f"\nPer task, B minus A context-hit share: {sum(ctx_diff) / len(ctx_diff) * 100:+.1f} pts, 95% interval {ctx_lo * 100:+.1f} to {ctx_hi * 100:+.1f} (same bootstrap). Tasks where B hit context most often relative to A: " +
             ", ".join(f"`{k}` (A {a_ctx[k]}, B {b_ctx[k]})" for d, k in shift[:6] if d > 0) + ".\n")
    pr = {r["arm"]: r for r in probe_rows}
    L.append("FP8 is not wordier. Per call it writes fewer output tokens than BF16 "
             f"({pr['B']['median_output_tokens_per_call_all']} against {pr['A']['median_output_tokens_per_call_all']}), and at the hit its trials hold less model text and about the same tool output. "
             "What differs is length: a trial fills the 128K window at about 150 calls whatever the precision, and "
             f"{pr['B']['trials_at_150_calls']} FP8 trials reached 150 calls against {pr['A']['trials_at_150_calls']} for BF16 ({pr['C']['trials_at_150_calls']} for NVFP4). "
             f"Part of that is BF16 stopping early more often on repeated format errors ({pr['A']['format_error_exits']} trials against {pr['B']['format_error_exits']} for FP8). "
             "A context hit is not a lost task either: the verifier scores the repository as the agent left it, and "
             + ", ".join(f"{pct(pr[a]['context_hit_resolved_share'], 0)}% of arm {a}'s context-hit trials resolved" for a in ARMS) + ".\n")
    L.append("## OOM sensitivity\n")
    L.append("Trials whose agent log or test output shows a kill or an allocation failure (`Killed`, signal 9, exit 137, out of memory). This is a text match, not the kernel log, which was not kept per trial.\n")
    L.append("| Arm | Trials with the signal | Resolved among them |")
    L.append("|---|---:|---:|")
    for r in oom_rows:
        L.append(f"| {r['arm']} | {r['trials_with_oom_signal']} | {r['resolved_among_them']} |")
    L.append("\nPaired difference against BF16 with those trials removed (task means over the trials left):\n")
    L.append("| Arm | Tasks | Difference | 95% interval | Verdict |")
    L.append("|---|---:|---:|---|---|")
    for r in oom_paired:
        L.append(f"| {r['arm']} ({r['precision']}) | {r['tasks']} | {r['diff'] * 100:+.1f} pts | {r['ci95_lo'] * 100:+.1f} to {r['ci95_hi'] * 100:+.1f} | {r['verdict']} |")
    L.append("\n## Throughput\n")
    L.append("Arms ran across spot reclaims, so wall clock per arm is not comparable. Trial hours summed, divided by the 48 trials in flight, give node hours per arm.\n")
    L.append("| Arm | Trial hours | Node hours at 48 | Node minutes per resolved trial | Median trial minutes |")
    L.append("|---|---:|---:|---:|---:|")
    for r in thr_rows:
        L.append(f"| {r['arm']} | {r['trial_hours']} | {r['node_hours_at_48']} | {r['node_minutes_per_resolved']} | {r['median_trial_min']} |")
    L.append("\nFor reference, arm C ran uninterrupted on node 2 from vLLM ready at 15:50:42 to `ARM_DONE` at 21:52:09 on 26 Nov: 6.0 h.\n")
    open(os.path.join(HERE, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
