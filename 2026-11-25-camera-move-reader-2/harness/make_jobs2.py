"""Write run two's render job lists. Fixed before rendering; needs pools/H_*.npz for sizes.

test_scenes.jsonl and test_combos.jsonl are run one's files, copied unchanged.
train_G reuses run one's train scene seeds and clip numbers, so D-G repeats run
one apart from the held-out-pair card filter.

usage: python make_jobs2.py <pools dir> <jobs dir>
"""

import json
import sys

import numpy as np

CPS = 4  # clips per scene
pools_dir, out = sys.argv[1], sys.argv[2]
size = {s: len(np.load(f"{pools_dir}/H_{s}.npz")["rel"]) for s in ("train", "val", "test")}
print("pool sizes", size)
mrng = np.random.default_rng(31)          # moving-props flags
hrng = np.random.default_rng(32)          # human window order
hperm = {s: list(hrng.permutation(size[s])) for s in ("train", "val")}
hpos = {"train": 0, "val": 0}


def hwin(pool):
    w = int(hperm[pool][hpos[pool] % size[pool]])
    hpos[pool] += 1
    return w


def write(name, pool, base, sources, n_clips, moving_p, filt, hpool=None):
    """sources: one letter for the whole split, or a function clip_number -> letter."""
    n_scenes = -(-n_clips // CPS)
    counts = {}
    with open(f"{out}/{name}.jsonl", "w") as f:
        k = 0
        for i in range(n_scenes):
            clips = []
            for _ in range(CPS):
                if k >= n_clips:
                    break
                src = sources(k) if callable(sources) else sources
                c = {"clip_id": f"{name}_{k:05d}", "combo": "normal", "source": src,
                     "moving": bool(mrng.random() < moving_p), "filter": filt}
                if src == "H":
                    c["hpool"] = hpool
                    c["hwin"] = (k if hpool == "test" else hwin(hpool))
                clips.append(c)
                counts[src] = counts.get(src, 0) + 1
                k += 1
            f.write(json.dumps({"scene_seed": base + i, "split": name, "pool": pool, "clips": clips}) + "\n")
    print(name, n_clips, counts)


N = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
write("pilot_F", "train", 900_000, "F", 52, 0.3, True)
write("pilot_H", "train", 910_000, "H", 52, 0.3, True, hpool="train")
write("train_G", "train", 100_000, "G", N, 0.3, True)
write("train_F", "train", 1_100_000, "F", N, 0.3, True)
write("train_H", "train", 1_200_000, "H", N, 0.3, True, hpool="train")
# val: 50 scenes per source, so the source is constant within a scene
write("val", "train", 200_000, lambda k: "GFH"[min(k // 200, 2)], 600, 0.3, True, hpool="val")
write("test_human", "test", 500_000, "H", size["test"], 0.0, False, hpool="test")
write("test_moving", "test", 600_000, lambda k: "GF"[k % 2], 500, 1.0, True)
