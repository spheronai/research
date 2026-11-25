"""Pilot check 2: what the new sources produce. Cards, travel, rejects, render time, moving props.
usage: venv/bin/python pilot_stats.py data/pilot_F data/pilot_H"""
import collections
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labeler  # noqa: E402

for d in sys.argv[1:]:
    ms = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "*", "meta.json")))]
    cards, comps, t = collections.Counter(), collections.Counter(), []
    for m in ms:
        lab = labeler.label(np.array(m["cameras"]), np.array(m["focal_mm"]), m["D_median"])
        cards[lab["card"]] += 1
        for a in labeler.AXES:
            if lab["axes"][a]["dir"]:
                comps[a] += 1
        comps["orbit"] += bool(lab["orbit"])
        comps["handheld"] += lab["handheld"]
        comps["static"] += lab["static"]
        t.append(m["render_s"])
    print(f"== {d}: {len(ms)} clips, render {np.mean(t):.2f} s/clip (process time), heldout rejects {sum(m['heldout_rejects'] for m in ms)}, "
          f"with moving props {sum(bool(m['moving_props']) for m in ms)}")
    print("axis presence:", dict(comps))
    if ms and ms[0]["move"].get("kind"):
        print("kinds:", collections.Counter(m["move"]["kind"] for m in ms), "travel/D median",
              float(np.median([m["move"]["travel_frac_D"] for m in ms])))
    for c, k in cards.most_common(12):
        print(f"  {k:3d}  {c}")
