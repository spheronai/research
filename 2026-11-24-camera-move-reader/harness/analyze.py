"""Turn run one's raw result JSON into tidy CSVs and charts.

usage (from the study folder): uv run -q --with matplotlib --with numpy python harness/analyze.py
Reads raw/results/*.json and raw/runs/seed*/train_log.json. Writes results/*.csv and charts/*.{svg,png}.
Written after the run (25 Nov 2026); it only reshapes numbers already in raw/results, it computes no new metric
except the re-shoot LPIPS interval, which uses the same 1,000-resample bootstrap as harness/reshoot_score.py.
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
# (file stem, split name in the CSV, subset)
FILES = [("results_test_scenes", "test_scenes", "all"), ("results_test_combos", "test_combos", "all"),
         ("results_human_moves", "human_moves", "all"), ("results_human_moves_moving_only", "human_moves", "moving")]
ARM_LABEL = {"REF_always_static": "Static floor", "A1_colmap_per_image_focal": "A1: COLMAP, per-frame focal",
             "A2_colmap_shared_focal": "A2: COLMAP, shared focal", "B_da3_base_rules": "B: DA3 + rules",
             "C_qwen3_vl_8b": "C: Qwen3-VL-8B", "D_ours_seed0": "D: ours, seed 0", "D_ours_seed1": "D: ours, seed 1"}


def r4(x):
    return None if x is None else round(float(x), 4)


def ci(v, i):
    return r4(v[i]) if v else None


def write(name, rows):
    cols = list(rows[0].keys())
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    print("wrote", name, len(rows))


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(CH, exist_ok=True)
    data, arms_rows, type_rows = {}, [], []
    for stem, split, subset in FILES:
        d = json.load(open(os.path.join(RAW, "results", stem + ".json")))
        data[(split, subset)] = d
        for arm, v in d["arms"].items():
            p, n = v.get("path", {}), v["names"]
            arms_rows.append({
                "split": split, "subset": subset, "arm": arm, "n_clips": v["n"],
                "failure_rate": r4(v["failure_rate"]), "unusable_camera_rate": r4(v.get("unusable_camera_rate")),
                "rot_err_deg": r4(p.get("rot_err_deg_mean")), "rot_err_ci_lo": ci(p.get("rot_ci"), 0), "rot_err_ci_hi": ci(p.get("rot_ci"), 1),
                "ate": r4(p.get("ate_mean")), "zoom_err": r4(p.get("zoom_err_mean")),
                "dir_acc": r4(n["dir_acc_mean"]), "dir_acc_ci_lo": ci(n.get("dir_acc_ci"), 0), "dir_acc_ci_hi": ci(n.get("dir_acc_ci"), 1),
                "zoom_f1": r4(n["zoom_f1"]), "orbit_acc": r4(n["orbit_acc"]), "handheld_acc": r4(n["handheld_acc"]),
                "size_acc": r4(n.get("size_acc_mean")), "exact_card": r4(n["exact_card"]),
                "exact_card_ci_lo": ci(n.get("exact_card_ci"), 0), "exact_card_ci_hi": ci(n.get("exact_card_ci"), 1),
                "zoom_dir_acc_on_zoom_clips": r4(n.get("zoom_dir_acc_on_zoom_clips")),
                "zoom_read_as_dolly": r4(n.get("dolly_false_on_zoom_only")),
                "handheld_threshold_deg": r4(v.get("handheld_threshold_deg")),
            })
            for g, gv in v.get("by_type", {}).items():
                type_rows.append({"split": split, "subset": subset, "arm": arm, "group": g, "n_clips": gv["n"],
                                  "exact_card": r4(gv["exact_card"]), "dir_acc": r4(gv["dir_acc"]),
                                  "rot_err_deg": r4(gv.get("rot_err_deg")), "failure_rate": r4(gv.get("fail"))})
    write("arms.csv", arms_rows)
    write("by_type.csv", type_rows)

    tl = []
    for f in sorted(glob.glob(os.path.join(RAW, "runs", "seed*", "train_log.json"))):
        j = json.load(open(f))
        vals = [e["val"]["total"] for e in j["log"]]
        tl.append({"run": os.path.basename(os.path.dirname(f)), "train_split": j["args"]["train_split"], "seed": j["args"]["seed"],
                   "best_val_loss": r4(j["best_val"]), "best_epoch": int(np.argmin(vals)), "epochs": len(vals),
                   "seconds": round(j["seconds"])})
    write("training.csv", tl)

    # re-shoot: means from per-clip scores, LPIPS interval by bootstrap over clips (seed 0, 1,000 resamples)
    rs = json.load(open(os.path.join(RAW, "results", "reshoot.json")))["per_clip"]
    rng = np.random.default_rng(0)
    rows = []
    for a, v in rs.items():
        lp = np.asarray(v["lpips"], float)
        bs = [lp[rng.integers(0, len(lp), len(lp))].mean() for _ in range(1000)]
        rows.append({"arm": a, "n_clips": len(v["clips"]), "psnr": r4(np.mean(v["psnr"])), "ssim": r4(np.mean(v["ssim"])),
                     "lpips": r4(lp.mean()), "lpips_ci_lo": r4(np.percentile(bs, 2.5)), "lpips_ci_hi": r4(np.percentile(bs, 97.5))})
    write("reshoot.csv", rows)
    pj = json.load(open(os.path.join(RAW, "results", "reshoot_paired.json")))
    prow = []
    for comp, mets in pj.items():
        for m, x in mets.items():
            prow.append({"comparison": comp, "metric": m, "mean_diff": r4(x["mean_diff"]), "ci_lo": r4(x["ci"][0]),
                         "ci_hi": r4(x["ci"][1]), "first_better_share": r4(x.get("a_better_share"))})
    write("reshoot_paired.csv", prow)
    charts(data)


def charts(data):
    import spheron_chart_style as st
    st.setup()
    plt = st.plt
    arms = list(ARM_LABEL)
    short = ["Static", "A1", "A2", "B", "C", "D s0", "D s1"]
    role = {"D_ours_seed0": ("finding", "o"), "D_ours_seed1": ("finding", "s"), "B_da3_base_rules": ("reference", None)}
    panels = [(("test_scenes", "all"), "Unseen scenes, 1,000 clips"),
              (("test_combos", "all"), "Unseen move pairs, 500 clips"),
              (("human_moves", "moving"), "Big Buck Bunny cameras, 17 moving windows")]

    # main: exact card with 95% CI, every arm, three test sets
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.70, bottom=0.12, wspace=0.08)
    for ax, (key, title) in zip(axes, panels):
        d = data[key]["arms"]
        st.style_ax(ax)
        for i, a in enumerate(arms):
            if a not in d:
                continue
            n = d[a]["names"]
            r, m = role.get(a, ("other", None))
            st.point(ax, i, 100 * n["exact_card"], 100 * n["exact_card_ci"][0], 100 * n["exact_card_ci"][1], r, m)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(short)
        ax.set_xlim(-0.6, len(arms) - 0.4)
        ax.set_ylim(-3, 100)
        ax.set_title(title, fontsize=10, color=st.INK_MUTED, loc="left", pad=10)
    axes[0].set_ylabel("Exact move card (%, 95% CI)")
    st.frame(fig, tag="Spheron Research // Study 2 // Run one",
             title="Exact move card by arm: the head wins on its own grammar, not on new pairs or human moves",
             subtitle="A1, A2: COLMAP. B: Depth Anything 3 + naming rules. C: Qwen3-VL-8B. D: our head on DA3, two seeds.")
    fig.legend(handles=st.legend_handles([("D, our head (seed 0, seed 1)", "finding", None), ("B, DA3 + rules", "reference", None),
                                          ("Other arms", "other", None)]),
               loc="upper right", bbox_to_anchor=(0.975, 0.955), ncol=3, fontsize=9)
    st.save(fig, os.path.join(CH, "exact-card-by-arm"))

    # supporting: zoom clips read as a dolly, horizontal bars per arm
    show = ["A1_colmap_per_image_focal", "B_da3_base_rules", "C_qwen3_vl_8b", "D_ours_seed0", "D_ours_seed1"]
    colr = {"D_ours_seed0": st.BLUE, "D_ours_seed1": st.BLUE, "B_da3_base_rules": st.WHITE}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.70, bottom=0.17, wspace=0.12)
    for ax, (key, lab) in zip(axes, [(("test_scenes", "all"), "Unseen scenes"), (("test_combos", "all"), "Unseen move pairs")]):
        d = data[key]["arms"]
        ys = [100 * (d[a]["names"].get("dolly_false_on_zoom_only") or 0) for a in show]
        yy = list(range(len(show)))[::-1]
        ax.barh(yy, ys, height=0.56, color=[colr.get(a, st.FADED) for a in show], zorder=2)
        for y, v in zip(yy, ys):
            ax.text(v + 1.5, y, f"{v:.0f}%", va="center", fontsize=9, color=st.INK_MUTED)
        ax.set_yticks(yy)
        ax.set_yticklabels([ARM_LABEL[a] for a in show])
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Zoom clips read as a dolly (%)")
        ax.set_title(lab, fontsize=10, color=st.INK_MUTED, loc="left", pad=10)
    st.frame(fig, tag="Spheron Research // Study 2 // Run one",
             title="How often a pure zoom is reported as a dolly",
             subtitle="Clips with a true zoom and no true dolly. Lower is better.")
    st.save(fig, os.path.join(CH, "zoom-read-as-dolly"))
    print("charts written")


if __name__ == "__main__":
    main()
