"""Red-team check: does anything from the test sets leak into train or val?

Checks, from the saved clip metadata (not from the generator's intent):
1. No model, HDRI or ground texture used in a train or val clip is in a test clip.
2. Every asset in a test clip is in the reserved test pool (sha1(id) % 100 < 15).
3. No train or val clip contains a held-out move pair; every test_combos clip does.
4. No two clips share frames (identical f00.png hashes across splits).

usage: venv/bin/python leak_check.py > results/leak_check.txt
"""

import glob
import hashlib
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moves  # noqa: E402

DATA = "/ephemeral/study/data"


def in_test_pool(i):
    return int(hashlib.sha1(i.encode()).hexdigest(), 16) % 100 < 15


def main():
    assets = defaultdict(set)
    bad_pairs = defaultdict(int)
    combos_missing = 0
    frame_hash = defaultdict(list)
    n = defaultdict(int)
    for split in ("train", "val", "test_scenes", "test_combos"):
        for mp in glob.glob(os.path.join(DATA, split, "*", "meta.json")):
            m = json.load(open(mp))
            n[split] += 1
            sc = m["scene"]
            ids = {("model", p["model"]) for p in sc["props"]} | {("hdri", sc["hdri"]), ("ground", sc["ground"])}
            assets[split] |= ids
            held = moves.pair_is_heldout(m["move"]["primitives"])
            if split in ("train", "val") and held:
                bad_pairs[split] += 1
            if split == "test_combos" and not held:
                combos_missing += 1
            h = hashlib.sha1(open(os.path.join(os.path.dirname(mp), "f00.png"), "rb").read()).hexdigest()
            frame_hash[h].append(split)
    fails = []
    print("clips per split:", dict(n))
    test = assets["test_scenes"] | assets["test_combos"]
    for split in ("train", "val"):
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
    print("RESULT:", "PASS" if not fails else "FAIL " + "; ".join(fails))


if __name__ == "__main__":
    main()
