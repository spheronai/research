"""Pilot check 1: run one's test clips re-render the same.

a) The camera path, focal lengths and scene depth of each re-rendered clip equal
   run one's meta.json (copied to ref/r1meta), so the scene and move streams
   are reproduced exactly.
b) Two renders of the same clips on this machine agree pixel-wise (PSNR).
usage: venv/bin/python det_check.py det_a/test_scenes det_b/test_scenes ref/r1meta/test_scenes
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

a_dir, b_dir, ref_dir = sys.argv[1:4]
cam_diff, foc_diff, d_diff, psnr, scene_same, n = [], [], [], [], 0, 0
for m in sorted(glob.glob(os.path.join(a_dir, "*", "meta.json"))):
    cid = os.path.basename(os.path.dirname(m))
    A = json.load(open(m))
    R = json.load(open(os.path.join(ref_dir, cid, "meta.json")))
    n += 1
    cam_diff.append(float(np.abs(np.array(A["cameras"]) - np.array(R["cameras"])).max()))
    foc_diff.append(float(np.abs(np.array(A["focal_mm"]) - np.array(R["focal_mm"])).max()))
    d_diff.append(abs(A["D_median"] - R["D_median"]))
    scene_same += int(A["scene"]["props"] == R["scene"]["props"] and A["scene"]["hdri"] == R["scene"]["hdri"])
    bdir = os.path.join(b_dir, cid)
    for i in range(16):
        x = np.asarray(Image.open(os.path.join(os.path.dirname(m), f"f{i:02d}.png")), np.float32) / 255
        y = np.asarray(Image.open(os.path.join(bdir, f"f{i:02d}.png")), np.float32) / 255
        mse = float(((x - y) ** 2).mean())
        psnr.append(10 * np.log10(1 / max(mse, 1e-12)))
print(f"clips {n}; scene spec identical to run one: {scene_same}/{n}")
print(f"max |camera - run one| {max(cam_diff):.2e}, max |focal - run one| {max(foc_diff):.2e}, max |D - run one| {max(d_diff):.2e}")
print(f"render A vs render B PSNR over {len(psnr)} frames: mean {np.mean(psnr):.1f} dB, min {np.min(psnr):.1f} dB")
ok = scene_same == n and max(cam_diff) < 1e-9 and max(foc_diff) < 1e-9 and np.min(psnr) > 40
print("DET_PASS" if ok else "DET_FAIL")
