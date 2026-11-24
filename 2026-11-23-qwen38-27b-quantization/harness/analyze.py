"""Turn raw/ into results/*.csv and charts/*.svg for study 1.

Run from the study folder:  uv run --with matplotlib python harness/analyze.py
Everything here is computed from files in raw/. Nothing is typed in by hand.
"""
import csv, glob, json, math, os, statistics as st
from collections import defaultdict

ARMS = {"A": "BF16", "B": "FP8", "C": "NVFP4", "D": "INT4-AWQ",
        "E": "FP8 + FP8 KV", "F": "NVFP4 as shipped (FP8 KV)"}
MAIN = ["A", "B", "C", "D"]
TPOT_MAX_MS, TTFT_P95_MAX_MS = 50.0, 2000.0
os.makedirs("results", exist_ok=True); os.makedirs("charts", exist_ok=True)


def write(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("wrote", path, len(rows))


# ---------- quality: per-sample scores, paired against BF16 ----------
# eval name -> (lm-eval task prefix, metric key, filter)
EVALS = {
    "gsm8k": ("gsm8k", "exact_match", "strict-match"),
    "mmlu_pro": ("mmlu_pro", "exact_match", "custom-extract"),
    "math500": ("minerva_math500", "math_verify", "none"),
    "ifeval": ("ifeval", "prompt_level_strict_acc", "none"),
}


def samples(arm, ev):
    task, metric, filt = EVALS[ev]
    out = {}
    for path in glob.glob(f"raw/{arm}/evals/{ev}/*/samples_*.jsonl"):
        for line in open(path):
            s = json.loads(line)
            if s.get("filter", filt) != filt:
                continue
            if metric not in s:
                continue
            out[(os.path.basename(path).split("_20")[0], s["doc_id"])] = float(s[metric])
    return out


def niah(arm):
    out = {}
    path = f"raw/{arm}/evals/niah.jsonl"
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            out[(f"{r['task']}@{r['length']}", r["i"])] = r["score"]
    return out


quality = []
base = {ev: samples("A", ev) for ev in EVALS}
base["niah"] = niah("A")
for arm in ARMS:
    for ev in list(EVALS) + ["niah"]:
        cur = niah(arm) if ev == "niah" else samples(arm, ev)
        if not cur:
            continue
        n = len(cur); acc = sum(cur.values()) / n
        se = math.sqrt(acc * (1 - acc) / n) if n > 1 else 0
        keys = [k for k in cur if k in base[ev]]
        if arm != "A" and keys:
            diffs = [cur[k] - base[ev][k] for k in keys]
            d = sum(diffs) / len(diffs)
            dse = st.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0
        else:
            d, dse = 0.0, 0.0
        quality.append({"arm": arm, "format": ARMS[arm], "eval": ev, "n": n, "correct": round(acc * n, 3), "accuracy": round(acc, 8),
                        "stderr": round(se, 8), "delta_vs_bf16": round(d, 8), "paired_delta_stderr": round(dse, 8),
                        "paired_n": len(keys) if arm != "A" else n,
                        "significant_2se": "yes" if arm != "A" and abs(d) > 2 * dse and dse > 0 else "no"})
write("results/quality.csv", quality)

# NIAH per task and length
nrows = []
for arm in ARMS:
    path = f"raw/{arm}/evals/niah_summary.json"
    if os.path.exists(path):
        s = json.load(open(path))
        for k, v in s["accuracy"].items():
            task, length = k.split("@")
            nrows.append({"arm": arm, "format": ARMS[arm], "task": task, "length_tokens": int(length),
                          "n": s["n"][k], "accuracy": v})
write("results/niah.csv", nrows)

# ---------- throughput, latency, power ----------
def power_samples(arm):
    rows = []
    path = f"raw/{arm}/logs/power.csv"
    if not os.path.exists(path):
        return rows
    import datetime as dt
    for r in csv.reader(open(path)):
        if not r or r[0].startswith("timestamp"):
            continue
        try:
            t = dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
            rows.append((t, float(r[1])))
        except ValueError:
            pass
    return rows


tp = []
for arm in ARMS:
    windows = {}
    wpath = f"raw/{arm}/bench/windows.csv"
    if os.path.exists(wpath):
        for c, rep, kind, t in csv.reader(open(wpath)):
            windows[(int(c), int(rep), kind)] = float(t)
    pw = power_samples(arm)
    for path in sorted(glob.glob(f"raw/{arm}/bench/c*_r*.json")):
        name = os.path.basename(path)[:-5]
        c, rep = int(name.split("_")[0][1:]), int(name.split("_")[1][1:])
        b = json.load(open(path))
        watts, joules_per_tok = None, None
        t0, t1 = windows.get((c, rep, "start")), windows.get((c, rep, "end"))
        if pw and t0 and t1:
            # nvidia-smi timestamps are local VM time; the VM runs UTC
            inside = [w for t, w in pw if t0 <= t <= t1]
            if inside:
                watts = sum(inside) / len(inside)
                joules_per_tok = watts * (t1 - t0) / b["total_output_tokens"]
        tp.append({"arm": arm, "format": ARMS[arm], "concurrency": c, "repeat": rep,
                   "prompts": b.get("num_prompts") or b.get("completed"), "completed": b.get("completed"),
                   "output_tok_s": round(b["output_throughput"], 1),
                   "ttft_p50_ms": round(b["median_ttft_ms"], 1), "ttft_p95_ms": round(b["p95_ttft_ms"], 1),
                   "tpot_p50_ms": round(b["median_tpot_ms"], 2), "itl_p50_ms": round(b["median_itl_ms"], 2),
                   "gpu_power_w": round(watts, 1) if watts else "",
                   "joules_per_output_token": round(joules_per_tok, 3) if joules_per_tok else ""})
write("results/throughput_runs.csv", tp)

summary = []
by = defaultdict(list)
for r in tp:
    by[(r["arm"], r["concurrency"])].append(r)
for (arm, c), rs in sorted(by.items()):
    med = lambda k: st.median(float(r[k]) for r in rs if r[k] != "")
    o = med("output_tok_s")
    jpt = [float(r["joules_per_output_token"]) for r in rs if r["joules_per_output_token"] != ""]
    summary.append({"arm": arm, "format": ARMS[arm], "concurrency": c, "repeats": len(rs),
                    "output_tok_s_median": round(o, 1),
                    "output_tok_s_min": min(r["output_tok_s"] for r in rs),
                    "output_tok_s_max": max(r["output_tok_s"] for r in rs),
                    "per_user_tok_s": round(1000 / med("tpot_p50_ms"), 1),
                    "ttft_p50_ms": round(med("ttft_p50_ms"), 1), "ttft_p95_ms": round(med("ttft_p95_ms"), 1),
                    "tpot_p50_ms": round(med("tpot_p50_ms"), 2),
                    "joules_per_output_token": round(st.median(jpt), 3) if jpt else "",
                    "meets_target": "yes" if med("tpot_p50_ms") <= TPOT_MAX_MS and med("ttft_p95_ms") <= TTFT_P95_MAX_MS else "no"})
write("results/throughput_summary.csv", summary)

json.dump({"quality": quality, "niah": nrows, "throughput": summary},
          open("results/all.json", "w"), indent=1)

# ---------- charts ----------
try:
    import matplotlib; matplotlib.use("svg"); import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("charts skipped: run with uv run --with matplotlib")
COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "D": "#6250d6"}  # validated categorical slots
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "svg.hashsalt": "study1"})
SVG_META = {"Creator": None, "Date": None, "Format": None, "Type": None}  # no tool or timestamp metadata
PNG_META = {"Software": None}


def label_svg(path, title, desc):
    """Give the SVG an accessible title and description for screen readers."""
    from html import escape
    s = open(path).read()
    i = s.index(">", s.index("<svg")) + 1
    open(path, "w").write(s[:i] + f"\n <title>{escape(title)}</title>\n <desc>{escape(desc)}</desc>" + s[i:])

# 1. accuracy change vs BF16, per eval, with paired 2-SE bars
evs = [e for e in ["gsm8k", "mmlu_pro", "math500", "ifeval", "niah"] if any(q["eval"] == e for q in quality)]
LABEL = {"gsm8k": "GSM8K", "mmlu_pro": "MMLU-Pro (subset)", "math500": "MATH-500", "ifeval": "IFEval",
         "niah": "Needle 8k+32k"}
fig, ax = plt.subplots(figsize=(8, 4.2))
w = 0.25
for j, arm in enumerate(["B", "C", "D"]):
    xs, ys, es = [], [], []
    for i, e in enumerate(evs):
        q = next((q for q in quality if q["arm"] == arm and q["eval"] == e), None)
        if q:
            xs.append(i + (j - 1) * w); ys.append(q["delta_vs_bf16"] * 100); es.append(2 * q["paired_delta_stderr"] * 100)
    ax.bar(xs, ys, width=w - 0.03, color=COL[arm], label=ARMS[arm], zorder=2)
    ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
ax.axhline(0, color=INK2, linewidth=1)
ax.set_xticks(range(len(evs))); ax.set_xticklabels([LABEL[e] for e in evs])
ax.set_ylabel("Accuracy change vs BF16 (points)")
ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, loc="upper left")
ax.set_ylim(-5.5, 6)
ax.set_title("Qwen3.8-27B on 1x RTX PRO 6000: accuracy change vs BF16 (bars: 2 paired standard errors)",
             fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig("charts/accuracy-delta.svg", metadata=SVG_META); fig.savefig("charts/accuracy-delta.png", dpi=160, metadata=PNG_META)
label_svg("charts/accuracy-delta.svg", "Accuracy change versus BF16 per eval",
          "Bars: accuracy change in points for FP8, NVFP4 and INT4-AWQ against BF16 on each eval, with two paired "
          "standard errors as error bars. Numbers are in results/quality.csv.")

# 2. output tok/s vs concurrency
fig, ax = plt.subplots(figsize=(7, 4.2))
for arm in MAIN:
    rs = sorted([r for r in summary if r["arm"] == arm], key=lambda r: r["concurrency"])
    if rs:
        ax.plot([r["concurrency"] for r in rs], [r["output_tok_s_median"] for r in rs], color=COL[arm], linewidth=2,
                marker="o", markersize=5, label=ARMS[arm])
        ax.annotate(ARMS[arm], (rs[-1]["concurrency"], rs[-1]["output_tok_s_median"]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
ax.set_xscale("log", base=2); ax.set_xticks([1, 4, 16, 64, 128]); ax.set_xticklabels(["1", "4", "16", "64", "128"])
ax.set_xlabel("Concurrent requests (1,024 in / 512 out)"); ax.set_ylabel("Output tokens per second")
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.legend(frameon=False)
ax.set_title("Throughput by checkpoint, vLLM 0.34.0, median of 3 runs", fontsize=10, color=INK, loc="left")
fig.tight_layout(); fig.savefig("charts/throughput.svg", metadata=SVG_META); fig.savefig("charts/throughput.png", dpi=160, metadata=PNG_META)
label_svg("charts/throughput.svg", "Output tokens per second by checkpoint and concurrency",
          "Lines: median output tokens per second of three runs at 1, 4, 16, 64 and 128 concurrent requests for "
          "BF16, FP8, NVFP4 and INT4-AWQ. Numbers are in results/throughput_summary.csv.")

print("charts done")
