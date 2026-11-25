"""Turn run two's raw result JSON into tidy CSVs and charts.

usage (from the study folder): uv run --with matplotlib python harness/analyze.py
Reads raw/results/*.json and raw/runs/*/train_log.json. Writes results/*.csv and charts/*.{svg,png}.
"""

import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "raw")
RES = os.path.join(HERE, "results")
CH = os.path.join(HERE, "charts")
MIXES = ["DG", "DF", "DH", "DGFH"]
MIX_LABEL = {"DG": "D-G grammar", "DF": "D-F free-form", "DH": "D-H human", "DGFH": "D-GFH mix"}
TESTS = [("test_scenes", None), ("test_combos", None), ("test_human", "moving"), ("test_human", None), ("test_moving", None), ("val", None)]


def tag(split, subset):
    return split + (f"_{subset}" if subset else "")


def load(split, subset):
    p = os.path.join(RAW, "results", f"results_{tag(split, subset)}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def write(name, rows, cols):
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    print("wrote", name, len(rows))


def r4(x):
    return None if x is None else round(float(x), 4)


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(CH, exist_ok=True)
    arms_rows, paired_rows, type_rows = [], [], []
    data = {}
    for split, subset in TESTS:
        d = load(split, subset)
        if not d:
            continue
        data[tag(split, subset)] = d
        for arm, v in d["arms"].items():
            p, n = v.get("path", {}), v["names"]
            arms_rows.append({
                "split": split, "subset": subset or "all", "arm": arm, "n_clips": v["n"],
                "failure_rate": r4(v["failure_rate"]), "rot_err_deg": r4(p.get("rot_err_deg_mean")),
                "ate": r4(p.get("ate_mean")), "zoom_err": r4(p.get("zoom_err_mean")),
                "dir_acc": r4(n["dir_acc_mean"]), "dir_acc_ci_lo": r4((n.get("dir_acc_ci") or [None])[0]),
                "dir_acc_ci_hi": r4((n.get("dir_acc_ci") or [None, None])[1]),
                "zoom_f1": r4(n["zoom_f1"]), "orbit_acc": r4(n["orbit_acc"]), "handheld_acc": r4(n["handheld_acc"]),
                "size_acc": r4(n.get("size_acc_mean")), "exact_card": r4(n["exact_card"]),
                "exact_card_ci_lo": r4(n["exact_card_ci"][0]), "exact_card_ci_hi": r4(n["exact_card_ci"][1]),
                "zoom_read_as_dolly": r4(n.get("dolly_false_on_zoom_only")),
                "handheld_threshold_deg": r4(v.get("handheld_threshold_deg")),
            })
            for metric, pv in v.get("paired_vs_B", {}).items():
                paired_rows.append({"split": split, "subset": subset or "all", "arm": arm, "metric": metric,
                                    "diff_vs_B": r4(pv["diff"]), "ci_lo": r4(pv["ci"][0]), "ci_hi": r4(pv["ci"][1]), "n_clips": pv["n"]})
            for g, gv in v.get("by_type", {}).items():
                type_rows.append({"split": split, "subset": subset or "all", "arm": arm, "group": g, "n_clips": gv["n"],
                                  "exact_card": r4(gv["exact_card"]), "dir_acc": r4(gv["dir_acc"]),
                                  "rot_err_deg": r4(gv["rot_err_deg"]), "failure_rate": r4(gv["fail"])})
    write("arms.csv", arms_rows, list(arms_rows[0].keys()))
    write("paired_vs_B.csv", paired_rows, list(paired_rows[0].keys()))
    write("by_type.csv", type_rows, list(type_rows[0].keys()))

    # training logs
    tl = []
    for f in sorted(glob.glob(os.path.join(RAW, "runs", "D*", "train_log.json"))):
        j = json.load(open(f))
        vals = [e["val"]["total"] for e in j["log"]]
        tl.append({"run": os.path.basename(os.path.dirname(f)), "train_split": " ".join(j["args"]["train_split"]),
                   "seed": j["args"]["seed"], "best_val_loss": r4(j["best_val"]), "best_epoch": int(np.argmin(vals)),
                   "epochs": len(vals), "seconds": round(j["seconds"])})
    if tl:
        write("training.csv", tl, list(tl[0].keys()))

    # mix choice on val (exact card, mean of the two seeds, direct names) and the three criteria
    def exact(t, arm):
        d = data.get(t)
        return d["arms"][arm]["names"]["exact_card"] if d and arm in d["arms"] else None

    def pv(t, arm):
        d = data.get(t)
        return d["arms"][arm].get("paired_vs_B", {}).get("exact_card") if d and arm in d["arms"] else None

    crit = []
    for m in MIXES:
        seeds = [f"D_ours_{m}_s0", f"D_ours_{m}_s1"]
        val = [exact("val", a) for a in seeds]
        row = {"mix": m, "label": MIX_LABEL[m], "val_exact_mean": r4(np.mean([x for x in val if x is not None])) if any(val) else None}
        for a in seeds:
            s = a[-2:]
            p1, p2 = pv("test_combos", a), pv("test_human_moving", a)
            ts = exact("test_scenes", a)
            row[f"S1_combos_diff_{s}"] = r4(p1["diff"]) if p1 else None
            row[f"S1_combos_ci_{s}"] = f"{p1['ci'][0]:+.3f} to {p1['ci'][1]:+.3f}" if p1 else None
            row[f"S2_human_moving_diff_{s}"] = r4(p2["diff"]) if p2 else None
            row[f"S2_human_moving_ci_{s}"] = f"{p2['ci'][0]:+.3f} to {p2['ci'][1]:+.3f}" if p2 else None
            row[f"S3_test_scenes_exact_{s}"] = r4(ts)
            row[f"pass_{s}"] = bool(p1 and p2 and ts is not None and p1["ci"][0] > 0 and p2["ci"][0] > 0 and ts >= 0.890 - 0.03)
        crit.append(row)
    best = max(crit, key=lambda r: r["val_exact_mean"] or -1)
    for r in crit:
        r["picked_on_val"] = r["mix"] == best["mix"]
    write("criteria.csv", crit, list(crit[0].keys()))

    # re-shoot
    rp = os.path.join(RAW, "results", "reshoot.json")
    if os.path.exists(rp):
        j = json.load(open(rp))
        rows = [{"arm": a, "n_clips": v["n"], "psnr": r4(v.get("psnr")), "ssim": r4(v.get("ssim")), "lpips": r4(v.get("lpips")),
                 "lpips_ci_lo": r4(v["lpips_ci"][0]) if "lpips_ci" in v else None,
                 "lpips_ci_hi": r4(v["lpips_ci"][1]) if "lpips_ci" in v else None} for a, v in j["arms"].items()]
        for a, pvv in j.get("paired_vs_B", {}).items():
            for k, x in pvv.items():
                rows_p = {"arm": a, "metric": k, "diff_vs_B": r4(x["diff"]), "ci_lo": r4(x["ci"][0]), "ci_hi": r4(x["ci"][1]), "n_clips": x["n"]}
                paired_rows.append({"split": "reshoot_test_scenes", "subset": "all", **rows_p})
        write("reshoot.csv", rows, list(rows[0].keys()))
        write("paired_vs_B.csv", paired_rows, list(paired_rows[0].keys()))

    charts(data, crit)


def charts(data, crit):
    import spheron_chart_style as st
    st.setup()
    plt = st.plt
    tag = "Spheron Research // Study 2 // Run two"
    panels = [("test_scenes", "Unseen scenes (grammar moves)"), ("test_combos", "Unseen move pairs (S1)"),
              ("test_human_moving", "Human camera paths, moving (S2)"), ("test_moving", "Moving props")]
    role = {"DGFH": "finding"}

    # main: D minus B, paired 95% CI, per mix and seed
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 5.2), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.70, bottom=0.10, wspace=0.08)
    for ax, (t, title) in zip(axes, panels):
        d = data.get(t)
        st.style_ax(ax, zero=0)
        ax.set_title(title, fontsize=10, color=st.INK_MUTED, loc="left", pad=10)
        ax.set_xticks(range(len(MIXES)))
        ax.set_xticklabels([MIX_LABEL[m].split(" ")[0] for m in MIXES])
        ax.set_xlim(-0.6, len(MIXES) - 0.4)
        if not d:
            continue
        for i, m in enumerate(MIXES):
            for j, s in enumerate(("s0", "s1")):
                p = d["arms"].get(f"D_ours_{m}_{s}", {}).get("paired_vs_B", {}).get("exact_card")
                if not p:
                    continue
                st.point(ax, i + (j - 0.5) * 0.3, 100 * p["diff"], 100 * p["ci"][0], 100 * p["ci"][1],
                         role.get(m, "other"), "o" if s == "s0" else "s")
    axes[0].set_ylabel("Exact move card, head minus DA3 + rules\n(points, paired 95% CI)")
    st.frame(fig, tag=tag, title="Each training mix against DA3 + rules: ahead on familiar motion, behind on new pairs and human paths",
             subtitle="Above zero means the head names more moves exactly. Circle seed 0, square seed 1. D-GFH was picked on val.")
    fig.legend(handles=st.legend_handles([("D-GFH, picked on val", "finding", None), ("Other mixes", "other", "o")]),
               loc="upper right", bbox_to_anchor=(0.975, 0.955), ncol=2, fontsize=9)
    st.save(fig, os.path.join(CH, "exact-card-vs-B"))

    # supporting: absolute exact card per split (mixes are the mean of two seeds)
    names = ["REF_always_static", "B_da3_base_rules"] + [f"D_ours_{m}" for m in MIXES]
    short = ["Static", "B"] + [MIX_LABEL[m].split(" ")[0] for m in MIXES]
    rl = {"B_da3_base_rules": "reference", "D_ours_DGFH": "finding"}
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 5.2), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.70, bottom=0.10, wspace=0.08)
    for ax, (t, title) in zip(axes, panels):
        d = data.get(t)
        st.style_ax(ax)
        ax.set_title(title, fontsize=10, color=st.INK_MUTED, loc="left", pad=10)
        for k, nm in enumerate(names):
            if not d:
                continue
            if nm.startswith("D_ours"):
                v = [d["arms"][f"{nm}_{s}"]["names"]["exact_card"] for s in ("s0", "s1") if f"{nm}_{s}" in d["arms"]]
                if v:
                    st.point(ax, k, 100 * float(np.mean(v)), role=rl.get(nm, "other"))
            elif nm in d["arms"]:
                n = d["arms"][nm]["names"]
                st.point(ax, k, 100 * n["exact_card"], 100 * n["exact_card_ci"][0], 100 * n["exact_card_ci"][1], rl.get(nm, "other"))
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(short, fontsize=8.5)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_ylim(-3, 100)
    axes[0].set_ylabel("Exact move card (%)")
    st.frame(fig, tag=tag, title="Exact move card per test set",
             subtitle="B and Static with 95% CI. Training mixes are the mean of two seeds; per-seed intervals are in results/arms.csv.")
    fig.legend(handles=st.legend_handles([("D-GFH, picked on val", "finding", None), ("B, DA3 + rules", "reference", None),
                                          ("Other arms", "other", None)]),
               loc="upper right", bbox_to_anchor=(0.975, 0.955), ncol=3, fontsize=9)
    st.save(fig, os.path.join(CH, "exact-card-by-split"))
    print("charts written")


if __name__ == "__main__":
    main()
