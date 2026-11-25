"""Blender script: rebuild each test scene from its saved spec and render every arm's re-shoot.

usage: blender -b --factory-startup -P reshoot_render.py -- --jobs jobs.jsonl --shard 0 --nshards 4 \
           --assets /ephemeral/study/assets --out /ephemeral/study/reshoot
Writes <out>/<arm>/<clip_id>/fXX.png.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_clips as rc  # noqa: E402


def main():
    a = rc.parse()
    manifest = json.load(open(os.path.join(a.assets, "manifest.json")))
    jobs = [json.loads(l) for l in open(a.jobs)]
    by_clip = {}
    for j in jobs:
        by_clip.setdefault(j["clip_id"], []).append(j)
    clip_ids = sorted(by_clip)[a.shard::a.nshards]
    cam, lib = rc.setup_render(a.device)
    t0, n = time.time(), 0
    for cid in clip_ids:
        todo = [j for j in by_clip[cid] if not os.path.exists(os.path.join(a.out, j["arm"], cid, "f15.png"))]
        if not todo:
            continue
        spec = json.loads(json.dumps(todo[0]["scene"]))
        for p in spec["props"]:
            p.pop("failed", None)
        rc.build_scene(spec, manifest, a.assets, lib)
        for j in todo:
            rc.render_clip(cam, np.array(j["mats"]), j["focals"], os.path.join(a.out, j["arm"], cid))
            n += 1
        print(f"PROGRESS shard {a.shard} renders {n} {time.time() - t0:.0f}s", flush=True)
    print(f"DONE shard {a.shard} renders {n} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
