"""Write the render job list. Scene seeds are disjoint per split; fixed before rendering."""
import json, sys

CLIPS_PER_SCENE = 4
SPLITS = [  # name, asset pool, scenes, combo mode, seed base
    ("pilot", "train", 12, "normal", 900_000),
    ("train", "train", 1500, "normal", 100_000),
    ("val", "train", 125, "normal", 200_000),
    ("test_scenes", "test", 250, "normal", 300_000),
    ("test_combos", "test", 125, "heldout", 400_000),
]
out = sys.argv[1] if len(sys.argv) > 1 else "jobs"
for name, pool, n, combo, base in SPLITS:
    with open(f"{out}/{name}.jsonl", "w") as f:
        k = 0
        for i in range(n):
            clips = []
            for _ in range(CLIPS_PER_SCENE):
                clips.append({"clip_id": f"{name}_{k:05d}", "combo": combo})
                k += 1
            f.write(json.dumps({"scene_seed": base + i, "split": name, "pool": pool, "clips": clips}) + "\n")
    print(name, n * CLIPS_PER_SCENE)
