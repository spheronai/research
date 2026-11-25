"""Build the re-shoot jobs: put each arm's predicted move back into the original scene.

The predicted path is used relative to its own frame 0 and anchored at the
true frame-0 camera, so every arm starts from the same framing. Translation is
rescaled from the arm's own depth unit to the true median scene depth (a user
re-shooting knows how far away their subject is). Focal length follows the
predicted zoom profile from the true starting focal. A clip an arm failed on is
re-shot with a locked-off camera, which is what a user would get from a
tool that returned nothing.

Arms: GT (re-render of the truth: the noise floor), STATIC (locked-off, the
do-nothing floor), A1, A2, B, D seed0. The VLM names moves but returns no path,
so it is not in this test.

usage: venv/bin/python reshoot_prep.py --n 200 --out /ephemeral/study/reshoot/jobs.jsonl
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom  # noqa: E402

DATA = "/ephemeral/study/data"
SPLIT = "test_scenes"


def anchor(pred_c2w, pred_f, pred_D, G0, gf0, gD):
    rel = geom.relative(pred_c2w)
    rel[:, :3, 3] *= gD / max(pred_D, 1e-6)
    mats = np.einsum("ij,njk->nik", G0, rel)
    pf = np.asarray(pred_f, float)
    return mats, gf0 * pf / pf[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="/ephemeral/study/reshoot/jobs.jsonl")
    ap.add_argument("--runs", default="/ephemeral/study/runs")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    clips = sorted(os.path.basename(os.path.dirname(m)) for m in glob.glob(os.path.join(DATA, SPLIT, "*", "meta.json")))
    pick = sorted(np.random.default_rng(0).choice(clips, size=min(a.n, len(clips)), replace=False).tolist())
    d_pred = {}
    pj = os.path.join(a.runs, "seed0", f"pred_{SPLIT}.json")
    if os.path.exists(pj):
        d_pred = json.load(open(pj))
    n_jobs, fails = 0, {}
    with open(a.out, "w") as fo:
        for cid in pick:
            cdir = os.path.join(DATA, SPLIT, cid)
            m = json.load(open(os.path.join(cdir, "meta.json")))
            G = np.array(m["cameras"])
            gf = np.array(m["focal_mm"])
            gD = m["D_median"]
            arms = {"GT": (G, gf), "STATIC": (np.repeat(G[:1], 16, 0), np.full(16, gf[0]))}
            for arm, fname in (("A1", "colmap.npz"), ("A2", "colmap_single.npz"), ("B", "da3.npz")):
                z = os.path.join(cdir, fname)
                d = np.load(z) if os.path.exists(z) else None
                if d is None or ("failed" in d.files and bool(d["failed"])):
                    arms[arm] = arms["STATIC"]
                    fails[arm] = fails.get(arm, 0) + 1
                else:
                    arms[arm] = anchor(d["c2w"], d["focal_mm"], float(d["D"]), G[0], gf[0], gD)
            if cid in d_pred:
                p = d_pred[cid]
                arms["D"] = anchor(np.array(p["c2w"]), p["focal_mm"], float(p["D"]), G[0], gf[0], gD)
            for arm, (mats, foc) in arms.items():
                fo.write(json.dumps({"clip_id": cid, "split": SPLIT, "arm": arm, "scene_seed": m["scene_seed"],
                                     "scene": m["scene"], "mats": np.asarray(mats).tolist(),
                                     "focals": [float(x) for x in foc]}) + "\n")
                n_jobs += 1
    print(f"wrote {n_jobs} jobs for {len(pick)} clips; failures re-shot locked-off: {fails}")


if __name__ == "__main__":
    main()
