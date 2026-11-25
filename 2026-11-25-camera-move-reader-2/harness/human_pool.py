"""Run two, source H: cut 4-second windows from human camera trajectories.

Sources
  RealEstate10K camera poses (CC BY 4.0, Google LLC): one .txt per clip, first
  line the YouTube URL, then per frame: timestamp_us fx fy cx cy k1 k2 and a
  3x4 world-to-camera matrix (OpenCV axes), intrinsics normalised by image size.
  Big Buck Bunny shot cameras (CC BY 3.0, Blender Foundation), dumped in run one
  by bbb_cam.py: per-frame camera-to-world (Blender axes), lens, sensor, and the
  median ray depth every 10 frames.

Each window is 16 samples evenly over 4.0 s (run one's grammar timing),
expressed relative to its first camera (Blender axes). RealEstate10K has no
metric scale: its translation is stored raw and scaled at render time (travel
log-uniform 0.02 to 0.6 of scene depth). Big Buck Bunny translation is divided
by the film's own depth at the window start, so it is already in depth units.

Splits (fixed before rendering)
  RealEstate10K: its own train and test lists. A video id that appears in both
  is dropped from train. Train videos are split again into ours-train and
  ours-val by sha1(video_id) % 10 == 0 -> val. Only clips of at least 4 s are
  used (about half). One window per clip; clips taken round-robin over videos
  (one per video, then a second), at most 2 per video.
  Big Buck Bunny: scene files 01-07 -> train, 08-13 -> test, 00 unused.
  Non-overlapping windows; a window needs one camera throughout, a perspective
  lens and >= 34 of 336 depth rays hitting at its first frame.

usage: venv/bin/python human_pool.py --re10k ext/re10k/RealEstate10K --bbb ext/bbb/cams --out pools
"""

import argparse
import glob
import hashlib
import json
import os

import numpy as np

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
T = np.linspace(0.0, 4.0, 16)
TOL_S = 0.06
N_TEST_RE10K = 1000


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


def rel_of(mats):
    return np.einsum("ij,njk->nik", np.linalg.inv(mats[0]), mats)


def read_re10k(path):
    with open(path) as f:
        url = f.readline().strip()
        rows = np.array([[float(x) for x in l.split()] for l in f if l.strip()])
    vid = url.split("v=")[-1].split("&")[0] if "v=" in url else url.rsplit("/", 1)[-1]
    return vid, rows


def re10k_window(rows, rng):
    ts = (rows[:, 0] - rows[0, 0]) / 1e6
    if ts[-1] < 4.0:
        return None
    t0 = float(rng.uniform(0.0, ts[-1] - 4.0))
    idx = []
    for t in t0 + T:
        j = int(np.argmin(np.abs(ts - t)))
        if abs(ts[j] - t) > TOL_S:
            return None
        idx.append(j)
    mats = []
    for j in idx:
        w2c = np.eye(4)
        w2c[:3, :4] = rows[j, 7:19].reshape(3, 4)
        mats.append(ortho(np.linalg.inv(w2c) @ FLIP))
    mats = np.stack(mats)
    focal = rows[idx, 1] * 36.0  # fx normalised by width, 36 mm wide sensor
    return rel_of(mats), focal, idx[0], idx[-1]


def bbb_windows(jf):
    d = json.load(open(jf))
    fr = {x["f"]: x for x in d["frames"]}
    fps = d["fps"]
    offs = [int(round(t * fps)) for t in T]
    out, last_end = [], -10**9
    for f0, (dep, hits) in sorted((int(k), v) for k, v in d["depth"].items()):
        if f0 <= last_end:
            continue
        idx = [f0 + o for o in offs]
        if any(i not in fr for i in idx):
            continue
        ws = [fr[i] for i in idx]
        if len({w["cam"] for w in ws}) > 1 or ws[0].get("type", "PERSP") != "PERSP":
            continue
        if dep is None or hits < 34 or dep <= 0:
            continue
        rel = rel_of(np.stack([ortho(w["m"]) for w in ws]))
        rel[:, :3, 3] /= dep
        focal = np.array([w["lens"] * 36.0 / w["sensor"] for w in ws])
        out.append((rel, focal, f"bbb:{os.path.basename(jf)[:-5]}:{idx[0]}-{idx[-1]}@{fps:g}fps"))
        last_end = idx[-1]
    return out


def save(path, items):
    rel = np.stack([x["rel"] for x in items]).astype(np.float64)
    focal = np.stack([x["focal"] for x in items]).astype(np.float64)
    np.savez(path, rel=rel, focal=focal,
             kind=np.array([x["kind"] for x in items]), src=np.array([x["src"] for x in items]),
             video=np.array([x["video"] for x in items]))
    print(path, len(items), {k: sum(x["kind"] == k for x in items) for k in ("re10k", "bbb")}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re10k", required=True)
    ap.add_argument("--bbb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=9000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(20261125)
    files = {s: sorted(glob.glob(os.path.join(a.re10k, s, "*.txt"))) for s in ("train", "test")}
    vids = {s: {} for s in files}
    for s, fl in files.items():
        for f in fl:
            with open(f) as fh:
                url = fh.readline().strip()
            vid = url.split("v=")[-1].split("&")[0] if "v=" in url else url.rsplit("/", 1)[-1]
            vids[s].setdefault(vid, []).append(f)
    both = set(vids["train"]) & set(vids["test"])
    print("re10k clips", {s: len(v) for s, v in files.items()}, "videos", {s: len(v) for s, v in vids.items()},
          "videos in both lists (dropped from train):", len(both), flush=True)
    pools = {"train": [], "val": [], "test": []}
    stats = {"too_short_or_gap": 0}

    def take(split_name, vid_map, n_max, per_video=2):
        order = sorted(vid_map)
        rng.shuffle(order)
        cands = {}
        for vid in order:
            fs = list(vid_map[vid])
            rng.shuffle(fs)
            cands[vid] = fs
        for _round in range(per_video):
            for vid in order:
                while cands[vid] and len(pools[split_name]) < n_max:
                    f = cands[vid].pop()
                    _, rows = read_re10k(f)
                    w = re10k_window(rows, rng) if len(rows) >= 16 else None
                    if w is None:
                        stats["too_short_or_gap"] += 1
                        continue
                    rel, focal, i0, i1 = w
                    pools[split_name].append({"rel": rel, "focal": focal, "kind": "re10k", "video": vid,
                                              "src": f"re10k:{os.path.basename(f)[:-4]}:rows{i0}-{i1}"})
                    break

    tr = {v: fs for v, fs in vids["train"].items() if v not in both}
    tr_val = {v: fs for v, fs in tr.items() if int(hashlib.sha1(v.encode()).hexdigest(), 16) % 10 == 0}
    tr_train = {v: fs for v, fs in tr.items() if v not in tr_val}
    take("train", tr_train, a.n_train)
    take("val", tr_val, 400)
    take("test", vids["test"], N_TEST_RE10K)
    for jf in sorted(glob.glob(os.path.join(a.bbb, "*.json"))):
        scene = int(os.path.basename(jf)[:2])
        dest = "train" if 1 <= scene <= 7 else ("test" if 8 <= scene <= 13 else None)
        if dest is None:
            continue
        for rel, focal, src in bbb_windows(jf):
            pools[dest].append({"rel": rel, "focal": focal, "kind": "bbb", "video": "bbb_" + os.path.basename(jf)[:2], "src": src})
    print("skipped", stats, flush=True)
    for s, items in pools.items():
        save(os.path.join(a.out, f"H_{s}.npz"), items)


if __name__ == "__main__":
    main()
