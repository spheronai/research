"""Score re-shoots against the original clip: PSNR, SSIM, LPIPS on frames 1-15.

Frame 0 is shared by construction (every arm is anchored at the true first
camera), so it is left out.

usage: venv/bin/python reshoot_score.py --out results/reshoot.json
"""

import argparse
import glob
import json
import os

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity

DATA = "/ephemeral/study/data"
RS = "/ephemeral/study/reshoot"
RNG = np.random.default_rng(0)


def load(d):
    return np.stack([np.asarray(Image.open(os.path.join(d, f"f{i:02d}.png")).convert("RGB"), np.float32) / 255 for i in range(1, 16)])


def boot(v, n=1000):
    v = np.asarray(v)
    idx = RNG.integers(0, len(v), (n, len(v)))
    s = v[idx].mean(1)
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/reshoot.json")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    net = lpips.LPIPS(net="alex").cuda().eval()
    arms = sorted(os.path.basename(p) for p in glob.glob(os.path.join(RS, "*")) if os.path.isdir(p))
    clips = sorted(os.path.basename(p) for p in glob.glob(os.path.join(RS, "GT", "*")))
    per = {arm: {"psnr": [], "ssim": [], "lpips": []} for arm in arms}
    keys = {}
    for cid in clips:
        ref = load(os.path.join(DATA, "test_scenes", cid))
        rt = torch.tensor(ref).permute(0, 3, 1, 2).cuda() * 2 - 1
        for arm in arms:
            d = os.path.join(RS, arm, cid)
            if not os.path.exists(os.path.join(d, "f15.png")):
                continue
            x = load(d)
            mse = ((x - ref) ** 2).mean((1, 2, 3))
            keys.setdefault(arm, []).append(cid)
            per[arm]["psnr"].append(float(np.mean(10 * np.log10(1 / np.maximum(mse, 1e-10)))))
            per[arm]["ssim"].append(float(np.mean([structural_similarity(ref[i], x[i], channel_axis=2, data_range=1.0) for i in range(len(x))])))
            with torch.no_grad():
                xt = torch.tensor(x).permute(0, 3, 1, 2).cuda() * 2 - 1
                per[arm]["lpips"].append(float(net(xt, rt).mean()))
    out = {"n_clips": len(clips), "arms": {}}
    for arm, m in per.items():
        out["arms"][arm] = {"n": len(m["psnr"]), **{k: float(np.mean(v)) for k, v in m.items() if v},
                            **{f"{k}_ci": boot(v) for k, v in m.items() if v}}
        print(f"{arm:8s} n {len(m['psnr'])} PSNR {np.mean(m['psnr']):.2f} SSIM {np.mean(m['ssim']):.3f} LPIPS {np.mean(m['lpips']):.3f}")
    # run two: paired D minus B on the clips both re-shot
    out["paired_vs_B"] = {}
    if "B" in per:
        bi = {c: i for i, c in enumerate(keys.get("B", []))}
        for arm in arms:
            if not arm.startswith("D"):
                continue
            ai = {c: i for i, c in enumerate(keys.get(arm, []))}
            common = [c for c in ai if c in bi]
            out["paired_vs_B"][arm] = {}
            for k in ("psnr", "ssim", "lpips"):
                d = np.array([per[arm][k][ai[c]] - per["B"][k][bi[c]] for c in common])
                out["paired_vs_B"][arm][k] = {"diff": float(d.mean()), "ci": boot(d), "n": len(d)}
            print(f"{arm} - B paired: " + " ".join(f"{k} {v['diff']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]" for k, v in out["paired_vs_B"][arm].items()))
    json.dump({**out, "per_clip": {arm: {**m, "clips": keys.get(arm, [])} for arm, m in per.items()}}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
