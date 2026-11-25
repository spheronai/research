"""Human camera moves: cut 4-second windows from Big Buck Bunny's original shot cameras
and replay them in held-out test scenes.

Source: the 2D scene files in download.blender.org/demo/bbb/blender.zip (CC BY 3.0,
Blender Foundation | www.blender.org), dumped by bbb_cam.py. These are the moves a
human layout artist keyed for the film. Only the motion is reused; the images are
rendered in our own held-out scenes, so this tests the motion distribution, not the
look of real footage.

Per window: 16 samples over 4 s at the file's frame rate, one camera throughout, a
perspective camera, and at least 34 of 336 depth rays hitting geometry at the first
frame (the same rule as the main renders). The path is taken relative to its first
frame, translation divided by the film's median scene depth, then placed at the
first camera of a test_scenes clip and multiplied by that clip's scene depth. Focal
length is the film's (lens x 36 / sensor width). At most 3 non-overlapping windows
per scene file.

usage: venv/bin/python human_moves_prep.py   (writes reshoot-style jobs and meta.json files)
"""

import glob
import json
import os

import numpy as np

R = "/ephemeral/study"
OUT_SPLIT = "human_moves"


def ortho(M):
    M = np.array(M, float)
    U, _, Vt = np.linalg.svd(M[:3, :3])
    R_ = U @ Vt
    if np.linalg.det(R_) < 0:
        U[:, -1] *= -1
        R_ = U @ Vt
    out = np.eye(4)
    out[:3, :3] = R_
    out[:3, 3] = M[:3, 3]
    return out


def main():
    test = sorted(glob.glob(f"{R}/data/test_scenes/*/meta.json"))
    jobs, n, skipped = [], 0, {"cam_switch": 0, "ortho": 0, "depth": 0}
    for jf in sorted(glob.glob(f"{R}/ext/bbb/cams/*.json")):
        d = json.load(open(jf))
        fr = {x["f"]: x for x in d["frames"]}
        if not fr:
            continue
        fps = d["fps"]
        stride = fps / 4.0
        span = int(round(15 * stride))
        taken, last_end = 0, -10**9
        for f0s, (dep, hits) in sorted(((int(k), v) for k, v in d["depth"].items())):
            if taken >= 3:
                break
            if f0s < last_end:
                continue
            idx = [int(round(f0s + i * stride)) for i in range(16)]
            if any(i not in fr for i in idx):
                continue
            ws = [fr[i] for i in idx]
            if len({w["cam"] for w in ws}) > 1:
                skipped["cam_switch"] += 1
                continue
            if ws[0].get("type", "PERSP") != "PERSP":
                skipped["ortho"] += 1
                continue
            if dep is None or hits < 34 or dep <= 0:
                skipped["depth"] += 1
                continue
            mats = np.stack([ortho(w["m"]) for w in ws])
            rel = np.einsum("ij,njk->nik", np.linalg.inv(mats[0]), mats)
            rel[:, :3, 3] /= dep
            focal = np.array([w["lens"] * 36.0 / w["sensor"] for w in ws])
            host = json.load(open(test[n % len(test)]))
            G0 = np.array(host["cameras"][0])
            Dh = host["D_median"]
            relw = rel.copy()
            relw[:, :3, 3] *= Dh
            world = np.einsum("ij,njk->nik", G0, relw)
            cid = f"{OUT_SPLIT}_{n:05d}"
            src = os.path.basename(jf)[:-5]
            os.makedirs(f"{R}/data/{OUT_SPLIT}/{cid}", exist_ok=True)
            meta = {"clip_id": cid, "split": OUT_SPLIT, "pool": "test", "scene_seed": host["scene_seed"],
                    "scene": host["scene"], "host_clip": host["clip_id"],
                    "move": {"primitives": ["human"], "handheld": False,
                             "source": f"Big Buck Bunny scene file 2d/{src}.blend, frames {idx[0]}-{idx[-1]} at {fps} fps, camera {ws[0]['cam']}"},
                    "bbb_depth": dep, "bbb_depth_hits": hits,
                    "cameras": world.tolist(), "focal_mm": focal.tolist(), "D_median": Dh,
                    "sensor_width_mm": 36.0, "res": host["res"]}
            json.dump(meta, open(f"{R}/data/{OUT_SPLIT}/{cid}/meta.json", "w"))
            jobs.append({"clip_id": cid, "split": OUT_SPLIT, "arm": OUT_SPLIT, "scene_seed": host["scene_seed"],
                         "scene": host["scene"], "mats": world.tolist(), "focals": focal.tolist()})
            n += 1
            taken += 1
            last_end = idx[-1]
    with open(f"{R}/reshoot/human_jobs.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"windows {n} skipped {skipped}")


if __name__ == "__main__":
    main()
