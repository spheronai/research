"""Score every arm on a split against ground truth. Run by the evaluation step only.

Rule-based arms (any method that returns a path) are named by labeler.py.
Their handheld threshold is calibrated on the val split per method (the
one threshold that maximises val handheld accuracy), so noisy paths are not
all called handheld by a threshold tuned on clean ground truth.

usage: venv/bin/python evaluate.py --split test_scenes --out results/ [--subset moving]

Run two additions: every run directory under --runs with a pred_<split>.json
is scored as a D arm; per-clip exact-card and direction results are stored in
per_clip_<split>.json; every D arm gets a paired bootstrap of D minus B on the
same clips; --subset moving keeps only clips whose ground-truth card is not
static (criterion S2); breakdowns by motion source and by moving props.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom  # noqa: E402
import labeler  # noqa: E402

AX = labeler.AXES
DATA = "/ephemeral/study/data"
RNG = np.random.default_rng(0)


def gt_split(split):
    out = {}
    for c in sorted(glob.glob(os.path.join(DATA, split, "*", "meta.json"))):
        m = json.load(open(c))
        G, gf = np.array(m["cameras"]), np.array(m["focal_mm"])
        lab = labeler.label(G, gf, m["D_median"])
        out[m["clip_id"]] = {"c2w": G, "focal": gf, "D": m["D_median"], "lab": lab,
                             "t": labeler.targets(lab), "prims": m["move"]["primitives"],
                             "handheld_gen": m["move"]["handheld"], "source": m.get("source", "G"),
                             "moving_props": len(m.get("moving_props", [])) > 0,
                             "kind": m["move"].get("kind")}
    return out


def path_arm(split, fname):
    out = {}
    for z in glob.glob(os.path.join(DATA, split, "*", fname)):
        cid = os.path.basename(os.path.dirname(z))
        d = np.load(z)
        if "failed" in d.files and bool(d["failed"]):
            out[cid] = {"failed": True}
        else:
            e = {"failed": False, "c2w": d["c2w"], "focal": d["focal_mm"], "D": float(d["D"])}
            out[cid] = e if usable(e) else {"failed": True, "unusable": True}
    return out


def usable(e):
    """A returned camera no lens can have (focal <= 0, NaN, depth <= 0) counts as a failure."""
    return (np.all(np.isfinite(e["c2w"])) and np.all(np.isfinite(e["focal"])) and np.all(np.asarray(e["focal"]) > 0)
            and np.isfinite(e["D"]) and e["D"] > 0)


def json_arm(path):
    if not os.path.exists(path):
        return {}
    d = json.load(open(path))
    out = {}
    for k, v in d.items():
        e = {"failed": False}
        if "c2w" in v:
            e.update(c2w=np.array(v["c2w"]), focal=np.array(v["focal_mm"]), D=float(v["D"]))
            if not usable(e):
                e = {"failed": True, "unusable": True}
        if v.get("names") is not None:
            e["names"] = v["names"]
        elif "parse_ok" in v:
            e["failed"] = True
        out[k] = e
    return out


def resid(p):
    return labeler.label(p["c2w"], p["focal"], p["D"])["rot_resid_rms"]


def calibrate(val_gt, val_pred):
    xs, ys = [], []
    for k, g in val_gt.items():
        p = val_pred.get(k)
        if p and not p["failed"] and "c2w" in p:
            xs.append(resid(p))
            ys.append(g["t"]["handheld"])
    if not xs:
        return None
    xs, ys = np.array(xs), np.array(ys)
    best = max(np.unique(np.concatenate([xs, [1e9]])), key=lambda th: float(np.mean((xs > th) == ys)))
    return float(best)


def names_from_path(p, hh):
    lab = labeler.label(p["c2w"], p["focal"], p["D"], hh_rot=hh, hh_trans=1e9 if hh is not None else None)
    return labeler.targets(lab)


def f1_macro(y, p, k=3):
    f = []
    for c in range(k):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def boot(vals, fn=np.mean, n=1000):
    vals = np.asarray(vals)
    if len(vals) == 0:
        return None
    idx = RNG.integers(0, len(vals), (n, len(vals)))
    s = np.array([fn(vals[i]) for i in idx])
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def score(gt, pred, names_mode, hh=None):
    """names_mode: 'rules' (from path), 'direct' (model heads), 'vlm' (direction only)."""
    keys = sorted(gt)
    res = {"n": len(keys)}
    failed = [k for k in keys if k not in pred or pred[k]["failed"]]
    res["failure_rate"] = len(failed) / len(keys)
    res["unusable_camera_rate"] = sum(1 for k in keys if k in pred and pred[k].get("unusable")) / len(keys)
    ok = [k for k in keys if k not in failed]
    # path
    if ok and "c2w" in pred[ok[0]]:
        pm = [geom.path_metrics(pred[k]["c2w"], pred[k]["focal"], pred[k]["D"], gt[k]["c2w"], gt[k]["focal"], gt[k]["D"]) for k in ok]
        rot = [m["rot_err_deg"] for m in pm]
        ate = [m["ate"] for m in pm if m["ate"] is not None]
        zoom = [m["zoom_err"] for m in pm]
        zclips = [m["zoom_err"] for k, m in zip(ok, pm) if gt[k]["t"]["zoom_dir"] != 1]
        res["path"] = {
            "rot_err_deg_mean": float(np.mean(rot)), "rot_err_deg_median": float(np.median(rot)), "rot_ci": boot(rot),
            "ate_mean": float(np.mean(ate)) if ate else None, "ate_median": float(np.median(ate)) if ate else None, "ate_ci": boot(ate),
            "zoom_err_mean": float(np.mean(zoom)), "zoom_err_ci": boot(zoom),
            "zoom_err_on_zoom_clips": float(np.mean(zclips)) if zclips else None,
            "focal_abs_err_mean": float(np.mean([m["focal_abs_err"] for m in pm])),
            "static_drift_median": float(np.median([m["static_drift"] for m in pm if m["static_drift"] is not None])) if any(m["static_drift"] is not None for m in pm) else None,
        }
    # names: failed clips count as all-none / not handheld
    def get_names(k):
        if k in failed:
            return None
        p = pred[k]
        if names_mode == "rules":
            return names_from_path(p, hh)
        return p["names"]
    Y, P = {}, {}
    fields = [f"{a}_dir" for a in AX] + ["orbit", "handheld"]
    if names_mode != "vlm":
        fields += [f"{a}_size" for a in AX] + ["dollyzoom", "easing"]
    for f in fields:
        Y[f], P[f] = [], []
    exact = []
    moving = []
    for k in keys:
        g = gt[k]["t"]
        n = get_names(k)
        for f in fields:
            default = 1 if (f.endswith("_dir") or f == "orbit") else 0
            Y[f].append(g[f])
            P[f].append(default if n is None else int(n.get(f, default)))
        core = [f"{a}_dir" for a in AX] + ["orbit"]
        exact.append(all((n is not None) and int(n.get(f, 1)) == g[f] for f in core))
        moving.append(g["has_motion"])
    Y = {f: np.array(v) for f, v in Y.items()}
    P = {f: np.array(v) for f, v in P.items()}
    moving = np.array(moving, bool)
    dir_acc = {a: float(np.mean(Y[f"{a}_dir"] == P[f"{a}_dir"])) for a in AX}
    per_clip_dir = np.mean(np.stack([Y[f"{a}_dir"] == P[f"{a}_dir"] for a in AX]), 0)
    res["names"] = {
        "dir_acc_mean": float(np.mean(per_clip_dir)), "dir_acc_ci": boot(per_clip_dir),
        "dir_acc": dir_acc,
        "dir_f1_mean": float(np.mean([f1_macro(Y[f"{a}_dir"], P[f"{a}_dir"]) for a in AX])),
        "zoom_f1": f1_macro(Y["zoom_dir"], P["zoom_dir"]),
        "zoom_f1_ci": boot(np.arange(len(keys)), fn=lambda i: f1_macro(Y["zoom_dir"][i], P["zoom_dir"][i]), n=500),
        "orbit_acc": float(np.mean(Y["orbit"] == P["orbit"])),
        "handheld_acc": float(np.mean(Y["handheld"] == P["handheld"])),
        "exact_card": float(np.mean(exact)), "exact_card_ci": boot(np.array(exact, float)),
    }
    if names_mode != "vlm":
        res["names"]["size_acc_mean"] = float(np.mean([np.mean(Y[f"{a}_size"] == P[f"{a}_size"]) for a in AX]))
        res["names"]["dollyzoom_acc"] = float(np.mean(Y["dollyzoom"] == P["dollyzoom"]))
        res["names"]["easing_acc_moving"] = float(np.mean(Y["easing"][moving] == P["easing"][moving])) if moving.any() else None
    # zoom clips only: did it get the zoom right
    zc = Y["zoom_dir"] != 1
    if zc.any():
        res["names"]["zoom_dir_acc_on_zoom_clips"] = float(np.mean(Y["zoom_dir"][zc] == P["zoom_dir"][zc]))
        res["names"]["dolly_false_on_zoom_only"] = float(np.mean(P["dolly_dir"][zc & (Y["dolly_dir"] == 1)] != 1)) if (zc & (Y["dolly_dir"] == 1)).any() else None
    # breakdown by generated move type and by handheld
    rot_by = dict(zip(ok, rot)) if "path" in res else {}
    zoom_by = dict(zip(ok, zoom)) if "path" in res else {}
    groups = {"static": [k for k in keys if not gt[k]["prims"]]}
    for prim in sorted({p for k in keys for p in gt[k]["prims"]}):
        groups[prim] = [k for k in keys if prim in gt[k]["prims"]]
    for src in sorted({gt[k]["source"] for k in keys}):
        groups[f"source_{src}"] = [k for k in keys if gt[k]["source"] == src]
    for kind in sorted({gt[k]["kind"] for k in keys if gt[k]["kind"]}):
        groups[f"human_{kind}"] = [k for k in keys if gt[k]["kind"] == kind]
    groups["moving_props"] = [k for k in keys if gt[k]["moving_props"]]
    groups["no_moving_props"] = [k for k in keys if not gt[k]["moving_props"]]
    groups["gt_static"] = [k for k in keys if not gt[k]["t"]["has_motion"]]
    groups["gt_moving"] = [k for k in keys if gt[k]["t"]["has_motion"]]
    groups["handheld"] = [k for k in keys if gt[k]["handheld_gen"]]
    groups["not_handheld"] = [k for k in keys if not gt[k]["handheld_gen"]]
    pos = {k: i for i, k in enumerate(keys)}
    exact = np.array(exact, float)
    res["by_type"] = {}
    for gname, ks in groups.items():
        if not ks:
            continue
        ix = [pos[k] for k in ks]
        r = [rot_by[k] for k in ks if k in rot_by]
        z = [zoom_by[k] for k in ks if k in zoom_by]
        res["by_type"][gname] = {
            "n": len(ks), "fail": sum(k in failed for k in ks) / len(ks),
            "dir_acc": float(np.mean(per_clip_dir[ix])), "exact_card": float(np.mean(exact[ix])),
            "rot_err_deg": float(np.mean(r)) if r else None, "zoom_err": float(np.mean(z)) if z else None,
        }
    res["_per_clip"] = {"keys": keys, "exact": exact.tolist(), "dir": per_clip_dir.tolist()}
    return res


def paired(a, b, n=2000):
    """Mean of a - b on the same clips, with a 95% paired bootstrap interval."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(1)
    idx = rng.integers(0, len(d), (n, len(d)))
    s = d[idx].mean(1)
    return {"diff": float(d.mean()), "ci": [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))], "n": int(len(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--runs", default="/ephemeral/study/runs")
    ap.add_argument("--subset", default=None, choices=[None, "moving"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    gt = gt_split(a.split)
    if a.subset == "moving":
        gt = {k: v for k, v in gt.items() if v["t"]["has_motion"]}
    tag = a.split + (f"_{a.subset}" if a.subset else "")
    val_gt = gt_split("val")
    arms = {}
    # reference floor: a camera that never moves and a card that says "static"
    still = {f"{x}_dir": 1 for x in AX} | {f"{x}_size": 0 for x in AX} | {"orbit": 1, "dollyzoom": 0, "handheld": 0, "easing": 0}
    floor = {k: {"failed": False, "c2w": np.tile(np.eye(4), (16, 1, 1)), "focal": np.full(16, 36.0), "D": 1.0, "names": still} for k in gt}
    arms["REF_always_static"] = score(gt, floor, "direct")
    for name, fname in (("A1_colmap_per_image_focal", "colmap.npz"), ("A2_colmap_shared_focal", "colmap_single.npz"), ("B_da3_base_rules", "da3.npz")):
        pred = path_arm(a.split, fname)
        if not pred:
            continue
        hh = calibrate(val_gt, path_arm("val", fname))
        arms[name] = {"handheld_threshold_deg": hh, **score(gt, pred, "rules", hh)}
    vlm = json_arm(os.path.join(a.runs, "vlm", f"pred_{a.split}.json"))
    if vlm:
        arms["C_qwen3_vl_8b"] = score(gt, vlm, "vlm")
    for run in sorted(glob.glob(os.path.join(a.runs, "D*"))):
        seed = os.path.basename(run)
        p = json_arm(os.path.join(run, f"pred_{a.split}.json"))
        if not p:
            continue
        arms[f"D_ours_{seed}"] = score(gt, p, "direct")
        hh = calibrate(val_gt, json_arm(os.path.join(run, "pred_val.json")))
        arms[f"D_ours_{seed}_names_via_rules"] = {"handheld_threshold_deg": hh, **score(gt, p, "rules", hh)}
    per_clip = {k: v.pop("_per_clip") for k, v in arms.items()}
    if "B_da3_base_rules" in per_clip:
        b = per_clip["B_da3_base_rules"]
        for k, v in arms.items():
            if k.startswith("D_") and per_clip[k]["keys"] == b["keys"]:
                v["paired_vs_B"] = {"exact_card": paired(per_clip[k]["exact"], b["exact"]),
                                    "dir_acc": paired(per_clip[k]["dir"], b["dir"])}
    out = {"split": a.split, "subset": a.subset, "n_clips": len(gt), "arms": arms}
    json.dump(out, open(os.path.join(a.out, f"results_{tag}.json"), "w"), indent=1)
    json.dump(per_clip, open(os.path.join(a.out, f"per_clip_{tag}.json"), "w"))
    for k, v in arms.items():
        p = v.get("path", {})
        n = v["names"]
        print(f"{k:38s} fail {v['failure_rate']:.3f} rot {p.get('rot_err_deg_mean', float('nan')):6.2f} ate {p.get('ate_mean') or float('nan'):.3f} "
              f"zoomerr {p.get('zoom_err_mean', float('nan')):.3f} | dir {n['dir_acc_mean']:.3f} zoomF1 {n['zoom_f1']:.3f} exact {n['exact_card']:.3f} hh {n['handheld_acc']:.3f}"
              + (f" | D-B exact {v['paired_vs_B']['exact_card']['diff']:+.3f} [{v['paired_vs_B']['exact_card']['ci'][0]:+.3f}, {v['paired_vs_B']['exact_card']['ci'][1]:+.3f}]" if "paired_vs_B" in v else ""))


if __name__ == "__main__":
    main()
