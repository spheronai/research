"""Study 3 charts, from results/*.csv, in the Spheron brand style.

Run from the study folder:
  SPHERON_FONT_DIR=<folder with SFPRODISPLAYREGULAR.otf and OCRAStd.otf> \
    uv run --with matplotlib python harness/charts.py
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import spheron_chart_style as st  # noqa: E402

st.setup()
import matplotlib.pyplot as plt  # noqa: E402

os.makedirs("charts", exist_ok=True)


def rows(name):
    with open(os.path.join("results", name)) as f:
        return list(csv.DictReader(f))


# 1. Main chart: paired difference against BF16, with the pre-registered -5 point margin.
paired = rows("paired_vs_A.csv")
order = [("all_111", "B"), ("all_111", "C"), ("offline_108", "B"), ("offline_108", "C")]
label = {
    ("all_111", "B"): "FP8\nall 111 tasks",
    ("all_111", "C"): "NVFP4\nall 111 tasks",
    ("offline_108", "B"): "FP8\n108 offline",
    ("offline_108", "C"): "NVFP4\n108 offline",
}
fig = plt.figure(figsize=(8.0, 4.9))
ax = fig.add_axes([0.11, 0.13, 0.85, 0.58])
ax.axhspan(-9, -5, color=(1, 1, 1, 0.04), zorder=0, linewidth=0)
ax.axhline(-5, color=st.INK_DIM, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
ax.text(3.45, -5.35, "-5 pt margin: a lower bound under this line\nwould not count as no measurable loss",
        ha="right", va="top", fontsize=8.5, color=st.INK_MUTED)
st.style_ax(ax, zero=0)
for i, key in enumerate(order):
    r = next(p for p in paired if (p["scope"], p["arm"]) == key)
    d, lo, hi = (float(r[k]) * 100 for k in ("diff", "ci95_lo", "ci95_hi"))
    role = "finding" if key[1] == "C" else "second"
    st.point(ax, i, d, lo, hi, role=role)
    ax.annotate(f"{d:+.1f}  ({lo:+.1f} to {hi:+.1f})", (i, hi), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9, color=st.INK)
ax.set_xticks(range(len(order)), [label[k] for k in order])
ax.set_xlim(-0.6, 3.6)
ax.set_ylim(-9, 10)
ax.set_yticks(range(-8, 11, 2))
ax.set_ylabel("Resolved rate vs BF16, points")
st.frame(fig, tag="Spheron Research / study 3",
         title="Neither FP8 nor NVFP4 lost measurable accuracy against BF16",
         subtitle="GLM-5.3-Flash on 111 SWE-rebench tasks, 5 runs each, 8x B200. Paired by task, 95% bootstrap interval.")
st.save(fig, "charts/paired-vs-bf16")

# 2. Why trials stopped: context limit, repeated format errors and the 50-minute agent limit.
stops = {r["stop_reason"]: r for r in rows("stop_reasons.csv")}
cats = [("context_limit", "Filled 128K context"), ("format_errors", "Repeated format errors"),
        ("time_limit", "Hit 50-min agent limit")]
arms = [("A", "BF16", "reference"), ("B", "FP8", "finding"), ("C", "NVFP4", "other")]
fig = plt.figure(figsize=(8.0, 4.6))
ax = fig.add_axes([0.09, 0.12, 0.88, 0.58])
st.style_ax(ax)
w = 0.26
for j, (arm, name, role) in enumerate(arms):
    c, _ = st.ROLE[role]
    for i, (k, _) in enumerate(cats):
        n = int(stops[k][f"{arm}_trials"])
        x = i + (j - 1) * w
        ax.bar(x, n, width=w * 0.86, color=c, linewidth=0, zorder=2)
        ax.text(x, n + 3, str(n), ha="center", va="bottom", fontsize=8.5, color=st.INK)
ax.set_xticks(range(len(cats)), [c for _, c in cats])
ax.set_ylim(0, 175)
ax.set_ylabel("Trials of 555")
handles = [plt.Rectangle((0, 0), 1, 1, color=st.ROLE[r][0], label=n) for _, n, r in arms]
fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.975, 0.80), ncol=3, fontsize=9)
st.frame(fig, tag="Spheron Research / study 3",
         title="FP8 runs filled the context window more often",
         subtitle="Trials per arm by why the agent stopped. A stopped trial can still resolve: the verifier scores the repo as left.")
st.save(fig, "charts/stop-reasons")
print("wrote charts/paired-vs-bf16 and charts/stop-reasons")
