"""Run two renderer: run one's scenes and renderer, three motion sources, moving props.

Run:
  blender -b --factory-startup -P render_clips2.py -- --jobs jobs/x.jsonl --shard 0 --nshards 4 \
      --assets /ephemeral/study/assets --out /ephemeral/study/data --pools /ephemeral/study/pools

Job line (run one's format, extra keys optional):
  {"scene_seed", "split", "pool", "clips": [{"clip_id", "combo",
    "source": "G"|"F"|"H" (default G), "moving": bool (default false),
    "filter": bool (default false), "hpool": "train"|"val"|"test", "hwin": int}]}

A clip with source G, moving false and filter false takes exactly run one's
code path and random stream, so run one's test splits re-render unchanged.

filter: label the path as placed (labeler.py, ground-truth depth) and reject it
if its card contains one of the five held-out pairs. Used for every training
and val clip of every source.

moving: one to three props move as rigid bodies (up to 0.5 scene depth of
travel and 90 degrees of turn, eased, inside the 4 s). Drawn from a separate
random stream so the camera path does not depend on it. Labels describe the
camera only.
"""

import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import labeler  # noqa: E402
import moves  # noqa: E402
import moves_free  # noqa: E402
import render_clips as rc  # noqa: E402

HELD = moves.HELDOUT_PAIRS


def card_components(lab):
    """Named parts of the card, with orbit and dolly zoom absorbing their axes (as card() does)."""
    ax = lab["axes"]
    comps, skip = set(), set()
    if lab["orbit"]:
        comps.add("orbit")
        skip |= {"truck", "pan"}
        if ax["dolly"]["dir"] > 0 and abs(ax["dolly"]["value"]) <= 0.6 * abs(ax["truck"]["value"]):
            skip.add("dolly")
    if lab["dollyzoom"]:
        comps.add("dollyzoom")
        skip |= {"dolly", "zoom"}
    for a in labeler.AXES:
        if a not in skip and ax[a]["dir"] != 0:
            comps.add(a)
    return comps


def card_has_heldout(lab):
    c = card_components(lab)
    return any(a in c and b in c for a, b in HELD)


# ---------------------------------------------------------------- placement

def place_generic(rng, boxes, traj_fn, tries=60):
    """run one's place_camera with the trajectory function passed in."""
    sc = bpy.context.scene
    for _ in range(tries):
        if boxes and rng.random() < 0.8:
            c, r, h = boxes[int(rng.integers(len(boxes)))]
            target = np.array([c[0], c[1], float(rng.uniform(0.2, 0.8)) * max(h, 0.3)])
        else:
            target = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0.2, 1.5)])
        dist = float(rng.uniform(3.0, 9.0))
        az = rng.uniform(0, 2 * math.pi)
        height = float(rng.uniform(0.4, 3.0))
        horiz = math.sqrt(max(dist ** 2 - (height - target[2]) ** 2, 1.0))
        pos = np.array([target[0] + horiz * math.cos(az), target[1] + horiz * math.sin(az), height])
        out = traj_fn(rng, pos, target)
        if out is None:
            continue
        mats, focals, D0 = out
        if rc.collides(mats[:, :3, 3], boxes):
            continue
        dg = bpy.context.evaluated_depsgraph_get()
        dvec = target - pos
        hit, *_ = sc.ray_cast(dg, Vector(pos), Vector(dvec / np.linalg.norm(dvec)), distance=float(np.linalg.norm(dvec)) - 0.6)
        if hit:
            continue
        return pos, target, mats, focals, D0
    return None


def human_traj(pool, w):
    rel0 = pool["rel"][w]
    focal = pool["focal"][w]
    kind = str(pool["kind"][w])
    raw = float(np.max(np.linalg.norm(rel0[:, :3, 3], axis=1)))

    def fn(rng, pos, target):
        R0 = moves.look_at(pos, target)
        d = rc.ray_depths(pos, R0, float(focal[0]))
        D = float(np.median(d)) if len(d) >= 34 else float(np.linalg.norm(target - pos))
        rel = rel0.copy()
        if kind == "re10k":
            travel = float(np.exp(rng.uniform(math.log(0.02), math.log(0.6))))
            rel[:, :3, 3] *= (travel * D / raw) if raw > 1e-12 else 0.0
            fn.info = {"travel_frac_D": travel, "raw_travel": raw}
        else:  # Big Buck Bunny: already in units of the film's scene depth
            rel[:, :3, 3] *= D
            fn.info = {"travel_frac_D": raw, "raw_travel": raw}
        M0 = np.eye(4)
        M0[:3, :3] = R0
        M0[:3, 3] = pos
        return np.einsum("ij,njk->nik", M0, rel), focal.copy(), float(np.linalg.norm(target - pos))
    fn.info = {}
    return fn


# ---------------------------------------------------------------- moving props

def scene_props():
    return [o for o in bpy.context.scene.collection.objects if o.type == "MESH" and o.name != "ground"]


def plan_movers(mrng, boxes, objs, mats, focals, D):
    """Pick 1-3 props (visible at frame 0 when possible) and give each a rigid eased move."""
    R0, p0 = mats[0, :3, :3], mats[0, :3, 3]
    fx = focals[0] / moves.SENSOR_W_MM * moves.RES_W
    vis = []
    for i, (c, r, h) in enumerate(boxes):
        q = R0.T @ (np.array([c[0], c[1], h / 2]) - p0)
        if q[2] < -0.3 and abs(q[0] / -q[2] * fx) < moves.RES_W / 2 and abs(q[1] / -q[2] * fx) < moves.RES_H / 2:
            vis.append(i)
    vis = [i for i in vis if objs[i] is not None]
    cand = vis if vis else [i for i in range(len(boxes)) if objs[i] is not None]
    if not cand:
        return []
    n = min(int(mrng.integers(1, 4)), len(cand))
    pick = list(mrng.choice(cand, size=n, replace=False))
    u = np.linspace(0.0, 1.0, moves.N_FRAMES)
    cams = mats[:, :3, 3]
    out = []
    for i in pick:
        c, r, h = boxes[int(i)]
        o = objs[int(i)]
        for _ in range(10):
            ang = mrng.uniform(0, 2 * math.pi)
            travel = float(mrng.uniform(0.05, 0.5)) * D
            turn = float(mrng.uniform(-90, 90))
            s = float(mrng.uniform(0.0, 0.5))
            e = float(mrng.uniform(min(s + 0.3, 1.0), 1.0))
            easing = str(mrng.choice(moves.EASINGS))
            w = moves.ease(np.clip((u - s) / max(e - s, 1e-6), 0, 1), easing)
            locs = np.array([[c[0] + math.cos(ang) * travel * x, c[1] + math.sin(ang) * travel * x, c[2]] for x in w])
            dmin = min(float(np.min(np.hypot(cams[:, 0] - L[0], cams[:, 1] - L[1]))) for L in locs) - r
            low = cams[:, 2] < h + 0.35
            close = any(np.any((np.hypot(cams[:, 0] - L[0], cams[:, 1] - L[1]) < r + 0.5) & low) for L in locs)
            if close:
                continue
            out.append({"obj": o, "prop_index": int(i), "locs": locs, "rotz0": float(o.rotation_euler[2]),
                        "rotz": float(o.rotation_euler[2]) + np.radians(turn) * w,
                        "spec": {"prop_index": int(i), "dir_deg": math.degrees(ang), "travel": travel,
                                 "travel_frac_D": travel / D, "turn_deg": turn, "start": s, "end": e,
                                 "easing": easing, "clearance": dmin}})
            break
    return out


def render_clip2(cam, mats, focals, out_dir, movers):
    sc = bpy.context.scene
    os.makedirs(out_dir, exist_ok=True)
    try:
        for i, (M, f) in enumerate(zip(mats, focals)):
            for m in movers:
                m["obj"].location = Vector(m["locs"][i].tolist())
                m["obj"].rotation_euler[2] = float(m["rotz"][i])
            cam.matrix_world = Matrix(M.tolist())
            cam.data.lens = float(f)
            sc.render.filepath = os.path.join(out_dir, f"f{i:02d}.png")
            bpy.ops.render.render(write_still=True)
    finally:  # restore, so the next clip in this scene starts from the scene spec
        for m in movers:
            m["obj"].location = Vector(m["locs"][0].tolist())
            m["obj"].rotation_euler[2] = m["rotz0"]
        bpy.context.view_layer.update()


# ---------------------------------------------------------------- main

def main():
    import argparse
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--assets", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pools", default="/ephemeral/study/pools")
    p.add_argument("--device", default="OPTIX")
    a = p.parse_args(argv)
    with open(os.path.join(a.assets, "manifest.json")) as f:
        manifest = json.load(f)
    jobs = [json.loads(l) for l in open(a.jobs)][a.shard::a.nshards]
    pools = {}
    cam, lib = rc.setup_render(a.device)
    t_start, n_done = time.time(), 0
    for job in jobs:
        rng = np.random.default_rng(job["scene_seed"])
        todo = [c for c in job["clips"] if not os.path.exists(os.path.join(a.out, job["split"], c["clip_id"], "meta.json"))]
        if not todo:
            continue
        scene_spec = rc.sample_scene(rng, manifest, job["pool"])
        boxes = rc.build_scene(scene_spec, manifest, a.assets, lib)
        objs = scene_props()
        # build_scene links props in box order; match by location to be safe
        byloc = {tuple(np.round(np.array(o.location), 5)): o for o in objs}
        objs = [byloc.get(tuple(np.round(c, 5))) for c, _, _ in boxes]
        for c in job["clips"]:
            idx = int(c["clip_id"].split("_")[-1])
            crng = np.random.default_rng([job["scene_seed"], idx])
            src = c.get("source", "G")
            filt = bool(c.get("filter", False))
            placed, mv, info, rejects = None, None, {}, 0
            for attempt in range(8 if not filt else 24):
                if src == "G":
                    mv = moves.sample_move(crng, allow_heldout=False, require_heldout=(c["combo"] == "heldout"))
                    mv = dict(mv, source="G")
                    placed = rc.place_camera(crng, boxes, mv)
                elif src == "F":
                    mv = moves_free.sample_free(crng)
                    placed = place_generic(crng, boxes, lambda r, p_, t_, mv=mv: moves_free.trajectory_free(mv, p_, t_))
                else:
                    if c["hpool"] not in pools:
                        z = np.load(os.path.join(a.pools, f"H_{c['hpool']}.npz"))
                        pools[c["hpool"]] = {k: z[k] for k in z.files}
                    P = pools[c["hpool"]]
                    w = (int(c["hwin"]) + attempt * 7919) % len(P["rel"])
                    fn = human_traj(P, w)
                    placed = place_generic(crng, boxes, fn)
                    mv = {"source": "H", "primitives": ["human"], "handheld": False, "window": w,
                          "kind": str(P["kind"][w]), "video": str(P["video"][w]), "src": str(P["src"][w]), **fn.info}
                if not placed:
                    continue
                if filt:
                    pos, target, mats, focals, D0 = placed
                    d = rc.ray_depths(pos, mats[0, :3, :3], focals[0])
                    D = float(np.median(d)) if len(d) >= 34 else D0
                    if card_has_heldout(labeler.label(mats, focals, D)):
                        rejects += 1
                        placed = None
                        continue
                break
            if not placed:
                print("SKIP no camera placement", c["clip_id"], src, "heldout_rejects", rejects, flush=True)
                continue
            out_dir = os.path.join(a.out, job["split"], c["clip_id"])
            if os.path.exists(os.path.join(out_dir, "meta.json")):
                continue
            pos, target, mats, focals, D0 = placed
            depths = rc.ray_depths(pos, mats[0, :3, :3], focals[0])
            D = float(np.median(depths)) if len(depths) >= 34 else D0
            movers = []
            if c.get("moving"):
                mrng = np.random.default_rng([job["scene_seed"], idx, 7])
                movers = plan_movers(mrng, boxes, objs, mats, focals, D)
            t0 = time.time()
            render_clip2(cam, mats, focals, out_dir, movers)
            meta = {
                "clip_id": c["clip_id"], "split": job["split"], "pool": job["pool"],
                "scene_seed": job["scene_seed"], "scene": scene_spec, "move": mv, "source": src,
                "heldout_filter": filt, "heldout_rejects": rejects,
                "moving_props": [m["spec"] for m in movers],
                "cam_pos": pos.tolist(), "target": target.tolist(), "D0": D0,
                "D_median": D, "depth_hits": int(len(depths)),
                "cameras": [M.tolist() for M in mats], "focal_mm": [float(x) for x in focals],
                "sensor_width_mm": moves.SENSOR_W_MM, "res": [moves.RES_W, moves.RES_H],
                "samples": rc.SAMPLES, "render_s": round(time.time() - t0, 2),
                "blender": bpy.app.version_string,
            }
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump(meta, f)
            n_done += 1
            if n_done % 10 == 0:
                el = time.time() - t_start
                print(f"PROGRESS shard {a.shard} clips {n_done} elapsed {el:.0f}s per_clip {el / n_done:.2f}s", flush=True)
    print(f"DONE shard {a.shard} clips {n_done} elapsed {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
