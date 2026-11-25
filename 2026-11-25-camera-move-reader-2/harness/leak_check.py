"""Red-team check: does anything from the test sets leak into train or val?

Checks, from the saved clip metadata (not from the generator's intent):
1. No model, HDRI or ground texture used in a train or val clip is in a test clip.
2. Every asset in a test clip is in the reserved test pool (sha1(id) % 100 < 15).
3. No train or val clip contains a held-out move pair; every test_combos clip does.
4. No two clips share frames (identical f00.png hashes across splits).
Run two adds:
3b. The held-out check also runs on the card of the placed path (labeler.py),
    since F and H paths are not built from primitives.
5. No RealEstate10K video id and no Big Buck Bunny scene file is used on both
   the training side (train_*, val) and the test side.

usage: venv/bin/python leak_check.py > results/leak_check.txt
"""

import glob
import hashlib
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labeler  # noqa: E402
import moves  # noqa: E402
import numpy as np  # noqa: E402

TRAIN_SPLITS = ("train_G", "train_F", "train_H", "val")
TEST_SPLITS = ("test_scenes", "test_combos", "test_human", "test_moving")
HELD = moves.HELDOUT_PAIRS


def card_held(m):
    lab = labeler.label(np.array(m["cameras"]), np.array(m["focal_mm"]), m["D_median"])
    ax, comps, skip = lab["axes"], set(), set()
    if lab["orbit"]:
        comps.add("orbit")
        skip |= {"truck", "pan"}
        if ax["dolly"]["dir"] > 0 and abs(ax["dolly"]["value"]) <= 0.6 * abs(ax["truck"]["value"]):
            skip.add("dolly")
    if lab["dollyzoom"]:
        comps.add("dollyzoom")
        skip |= {"dolly", "zoom"}
    comps |= {a for a in labeler.AXES if a not in skip and ax[a]["dir"] != 0}
    return any(a in comps and b in comps for a, b in HELD)

DATA = "/ephemeral/study/data"


def in_test_pool(i):
    return int(hashlib.sha1(i.encode()).hexdigest(), 16) % 100 < 15


def main():
    assets = defaultdict(set)
    bad_pairs = defaultdict(int)
    combos_missing = 0
    frame_hash = defaultdict(list)
    n = defaultdict(int)
    videos = {"train": set(), "test": set()}
    for split in TRAIN_SPLITS + TEST_SPLITS:
        for mp in glob.glob(os.path.join(DATA, split, "*", "meta.json")):
            m = json.load(open(mp))
            n[split] += 1
            sc = m["scene"]
            ids = {("model", p["model"]) for p in sc["props"]} | {("hdri", sc["hdri"]), ("ground", sc["ground"])}
            assets[split] |= ids
            held = moves.pair_is_heldout(m["move"]["primitives"])
            if split in TRAIN_SPLITS and (held or card_held(m)):
                bad_pairs[split] += 1
            if m["move"].get("video"):
                videos["train" if split in TRAIN_SPLITS else "test"].add(m["move"]["video"])
            if split == "test_combos" and not held:
                combos_missing += 1
            h = hashlib.sha1(open(os.path.join(os.path.dirname(mp), "f00.png"), "rb").read()).hexdigest()
            frame_hash[h].append(split)
    fails = []
    print("clips per split:", dict(n))
    test = set().union(*(assets[s] for s in TEST_SPLITS))
    for split in TRAIN_SPLITS:
        inter = assets[split] & test
        print(f"{split}: {len(assets[split])} distinct assets, shared with test: {len(inter)}")
        if inter:
            fails.append(f"{split} shares {sorted(inter)[:10]}")
        notrain = [i for _, i in assets[split] if in_test_pool(i)]
        if notrain:
            fails.append(f"{split} uses test-pool assets {notrain[:10]}")
    out_of_pool = [i for _, i in test if not in_test_pool(i)]
    print(f"test: {len(test)} distinct assets, outside the test pool: {len(out_of_pool)}")
    if out_of_pool:
        fails.append(f"test uses train-pool assets {out_of_pool[:10]}")
    print("held-out pairs in train/val:", dict(bad_pairs), "| test_combos clips without one:", combos_missing)
    if bad_pairs or combos_missing:
        fails.append("held-out pair leak")
    dup = {h: s for h, s in frame_hash.items() if len(s) > 1 and len(set(s)) > 1}
    print("first frames identical across splits:", len(dup))
    if dup:
        fails.append(f"{len(dup)} cross-split duplicate first frames")
    vshared = videos["train"] & videos["test"]
    print(f"human-path videos/files: train side {len(videos['train'])}, test side {len(videos['test'])}, on both: {len(vshared)}")
    if vshared:
        fails.append(f"human-path sources on both sides: {sorted(vshared)[:10]}")
    print("RESULT:", "PASS" if not fails else "FAIL " + "; ".join(fails))


if __name__ == "__main__":
    main()
