"""Blender script: build procedural scenes from CC0 assets and render camera-move clips.

Run:
  blender -b --factory-startup -P render_clips.py -- --jobs jobs.jsonl --shard 0 --nshards 4 \
      --assets /ephemeral/study/assets --out /ephemeral/study/data

Each job line: {"scene_seed": int, "split": str, "pool": "train"|"test",
                "clips": [{"clip_id": str, "combo": "normal"|"heldout"}]}

Every clip writes f00.png..f15.png and meta.json (scene spec, move spec,
per-frame camera-to-world matrices, focal lengths, median scene depth).
The scene spec is enough to rebuild the scene exactly (see build_scene),
which the re-shoot test uses.
"""

import argparse
import json
import math
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moves  # noqa: E402

SAMPLES = int(os.environ.get("CAM_SAMPLES", "24"))


def parse():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--assets", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="OPTIX")
    return p.parse_args(argv)


# ---------------------------------------------------------------- setup

def setup_render(device):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if device != "CPU":
        prefs.compute_device_type = device
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == device
        sc.cycles.device = "GPU"
    else:
        sc.cycles.device = "CPU"
    sc.cycles.samples = SAMPLES
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    if device != "CPU":
        sc.cycles.denoising_use_gpu = True
    # four Blender processes share 28 vCPUs; do not let each grab them all
    sc.render.threads_mode = "FIXED"
    sc.render.threads = int(os.environ.get("CAM_THREADS", "6"))
    sc.cycles.max_bounces = 4
    sc.cycles.diffuse_bounces = 2
    sc.cycles.glossy_bounces = 2
    sc.cycles.transmission_bounces = 2
    sc.cycles.transparent_max_bounces = 4
    sc.render.use_persistent_data = True
    sc.render.resolution_x = moves.RES_W
    sc.render.resolution_y = moves.RES_H
    sc.render.resolution_percentage = 100
    sc.render.use_motion_blur = False
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.compression = 15
    sc.view_settings.view_transform = "AgX"
    cam_data = bpy.data.cameras.new("cam")
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = moves.SENSOR_W_MM
    cam_data.clip_start = 0.05
    cam_data.clip_end = 1000
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    lib = bpy.data.collections.new("library")  # not linked to the scene: hidden
    return cam, lib


_model_cache = {}


def load_model(assets, model, lib):
    """Import a glTF once, join to one mesh, normalise so the largest side is 1."""
    mid = model["id"]
    if mid in _model_cache:
        return _model_cache[mid]
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.join(assets, model["file"]))
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    new_names = [o.name for o in new]
    if not meshes:
        raise RuntimeError("no mesh in asset")
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    # bake parent transforms into the meshes before joining
    for o in meshes:
        mw = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = mw
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    for name in new_names:
        o = bpy.data.objects.get(name)
        if o is not None and o != obj:
            bpy.data.objects.remove(o, do_unlink=True)
    pts = np.array([v.co[:] for v in obj.data.vertices])
    lo, hi = pts.min(0), pts.max(0)
    size = float((hi - lo).max())
    # recentre: footprint centre at origin, bottom at z=0, largest side 1
    off = Vector((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))
    obj.data.transform(Matrix.Translation(off))
    obj.data.transform(Matrix.Scale(1.0 / size, 4))
    obj.data.update()
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    lib.objects.link(obj)
    ext = (hi - lo) / size
    _model_cache[mid] = (obj, ext)
    return _model_cache[mid]


# ---------------------------------------------------------------- scene

def sample_scene(rng, manifest, pool):
    models = [m for m in manifest["models"] if m["split"] == pool]
    hdris = [h for h in manifest["hdris"] if h["split"] == pool]
    texs = [t for t in manifest["textures"] if t["split"] == pool]
    n_props = int(rng.integers(10, 29))
    props, placed = [], []
    for _ in range(n_props):
        m = models[int(rng.integers(len(models)))]
        scale = float(np.exp(rng.uniform(math.log(0.35), math.log(2.8))))
        for _try in range(40):
            r = 7.0 * math.sqrt(rng.random())
            a = rng.uniform(0, 2 * math.pi)
            x, y = r * math.cos(a), r * math.sin(a)
            rad = 0.6 * scale
            if all(math.hypot(x - px, y - py) > rad + pr for px, py, pr in placed):
                placed.append((x, y, rad))
                props.append({"model": m["id"], "loc": [x, y, 0.0], "rot_z": float(rng.uniform(0, 360)), "scale": scale})
                break
    return {
        "hdri": hdris[int(rng.integers(len(hdris)))]["id"],
        "hdri_rot_z": float(rng.uniform(0, 360)),
        "hdri_strength": float(rng.uniform(0.6, 1.6)),
        "ground": texs[int(rng.integers(len(texs)))]["id"],
        "ground_tile_m": float(rng.uniform(1.5, 4.0)),
        "props": props,
    }


def build_scene(spec, manifest, assets, lib):
    sc = bpy.context.scene
    # clear previous instances
    for o in list(sc.collection.objects):
        if o.type != "CAMERA":
            bpy.data.objects.remove(o, do_unlink=True)
    # world
    hd = {h["id"]: h for h in manifest["hdris"]}[spec["hdri"]]
    world = sc.world or bpy.data.worlds.new("world")
    sc.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Rotation"].default_value = (0, 0, math.radians(spec["hdri_rot_z"]))
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    key = os.path.join(assets, hd["file"])
    env.image = bpy.data.images.get(os.path.basename(key)) or bpy.data.images.load(key, check_existing=True)
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = spec["hdri_strength"]
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    # ground
    tx = {t["id"]: t for t in manifest["textures"]}[spec["ground"]]
    mat = bpy.data.materials.get("ground_" + tx["id"])
    if mat is None:
        mat = bpy.data.materials.new("ground_" + tx["id"])
        mat.use_nodes = True
        n = mat.node_tree
        bsdf = n.nodes["Principled BSDF"]
        tcn = n.nodes.new("ShaderNodeTexCoord")
        mpn = n.nodes.new("ShaderNodeMapping")
        mpn.name = "tile"
        n.links.new(tcn.outputs["Object"], mpn.inputs["Vector"])

        def img(path, noncolor):
            t = n.nodes.new("ShaderNodeTexImage")
            t.image = bpy.data.images.load(os.path.join(assets, path), check_existing=True)
            if noncolor:
                t.image.colorspace_settings.name = "Non-Color"
            n.links.new(mpn.outputs["Vector"], t.inputs["Vector"])
            return t
        d = img(tx["maps"]["diffuse"], False)
        n.links.new(d.outputs["Color"], bsdf.inputs["Base Color"])
        if "rough" in tx["maps"]:
            r = img(tx["maps"]["rough"], True)
            n.links.new(r.outputs["Color"], bsdf.inputs["Roughness"])
        if "normal" in tx["maps"]:
            nm = img(tx["maps"]["normal"], True)
            nmap = n.nodes.new("ShaderNodeNormalMap")
            n.links.new(nm.outputs["Color"], nmap.inputs["Color"])
            n.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    s = 1.0 / spec["ground_tile_m"]
    mat.node_tree.nodes["tile"].inputs["Scale"].default_value = (s, s, s)
    me = bpy.data.meshes.get("groundmesh")
    if me is None:
        me = bpy.data.meshes.new("groundmesh")
        h = 80.0
        me.from_pydata([(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)], [], [(0, 1, 2, 3)])
    ground = bpy.data.objects.new("ground", me)
    if not me.materials:
        me.materials.append(mat)
    else:
        me.materials[0] = mat
    sc.collection.objects.link(ground)
    # props
    boxes = []
    models = {m["id"]: m for m in manifest["models"]}
    for p in spec["props"]:
        try:
            src, ext = load_model(assets, models[p["model"]], lib)
        except Exception as e:  # a broken asset is skipped, and recorded
            print("WARN model failed", p["model"], repr(e)[:100], flush=True)
            p["failed"] = True
            continue
        o = src.copy()  # shares mesh data
        o.location = p["loc"]
        o.rotation_euler = (0, 0, math.radians(p["rot_z"]))
        o.scale = (p["scale"],) * 3
        sc.collection.objects.link(o)
        r_xy = 0.5 * math.hypot(ext[0], ext[1]) * p["scale"]
        boxes.append((np.array(p["loc"]), r_xy, ext[2] * p["scale"]))
    bpy.context.view_layer.update()
    return boxes


# ---------------------------------------------------------------- cameras

def collides(points, boxes, margin=0.35):
    for c, r, h in boxes:
        d = np.hypot(points[:, 0] - c[0], points[:, 1] - c[1])
        if np.any((d < r + margin) & (points[:, 2] < h + margin)):
            return True
    return bool(np.any(points[:, 2] < 0.15))


def ray_depths(cam_pos, R, focal, max_d=200.0):
    """Depth along the view axis on a 24x14 grid of pixels at frame 0."""
    sc = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    fx = focal / moves.SENSOR_W_MM * moves.RES_W
    fwd = -R[:, 2]
    out = []
    for v in np.linspace(-0.45, 0.45, 14) * moves.RES_H:
        for u in np.linspace(-0.45, 0.45, 24) * moves.RES_W:
            d = R @ np.array([u / fx, v / fx, -1.0])
            d /= np.linalg.norm(d)
            hit, loc, *_ = sc.ray_cast(dg, Vector(cam_pos), Vector(d), distance=max_d)
            if hit:
                out.append(float(np.dot(np.array(loc) - cam_pos, fwd)))
    return np.array(out)


def place_camera(rng, boxes, spec_move):
    for _ in range(60):
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
        mats, focals, D0 = moves.trajectory(spec_move, pos, target)
        if collides(mats[:, :3, 3], boxes):
            continue
        # the target must not be hidden behind a prop at frame 0
        sc = bpy.context.scene
        dg = bpy.context.evaluated_depsgraph_get()
        dvec = target - pos
        hit, loc, *_ = sc.ray_cast(dg, Vector(pos), Vector(dvec / np.linalg.norm(dvec)), distance=float(np.linalg.norm(dvec)) - 0.6)
        if hit:
            continue
        return pos, target, mats, focals, D0
    return None


def render_clip(cam, mats, focals, out_dir):
    sc = bpy.context.scene
    os.makedirs(out_dir, exist_ok=True)
    for i, (M, f) in enumerate(zip(mats, focals)):
        cam.matrix_world = Matrix(M.tolist())
        cam.data.lens = float(f)
        sc.render.filepath = os.path.join(out_dir, f"f{i:02d}.png")
        bpy.ops.render.render(write_still=True)


def main():
    a = parse()
    with open(os.path.join(a.assets, "manifest.json")) as f:
        manifest = json.load(f)
    jobs = [json.loads(l) for l in open(a.jobs)]
    jobs = jobs[a.shard::a.nshards]
    cam, lib = setup_render(a.device)
    t_start = time.time()
    n_done = 0
    for job in jobs:
        rng = np.random.default_rng(job["scene_seed"])
        todo = [c for c in job["clips"] if not os.path.exists(os.path.join(a.out, job["split"], c["clip_id"], "meta.json"))]
        if not todo:
            continue
        scene_spec = sample_scene(rng, manifest, job["pool"])
        boxes = build_scene(scene_spec, manifest, a.assets, lib)
        for c in job["clips"]:
            crng = np.random.default_rng([job["scene_seed"], int(c["clip_id"].split("_")[-1])])
            placed = None
            for _ in range(8):
                mv = moves.sample_move(crng, allow_heldout=False, require_heldout=(c["combo"] == "heldout"))
                placed = place_camera(crng, boxes, mv)
                if placed:
                    break
            if not placed:
                print("SKIP no camera placement", c["clip_id"], flush=True)
                continue
            out_dir = os.path.join(a.out, job["split"], c["clip_id"])
            if os.path.exists(os.path.join(out_dir, "meta.json")):
                continue
            pos, target, mats, focals, D0 = placed
            depths = ray_depths(pos, mats[0, :3, :3], focals[0])
            D = float(np.median(depths)) if len(depths) >= 34 else D0
            t0 = time.time()
            render_clip(cam, mats, focals, out_dir)
            meta = {
                "clip_id": c["clip_id"], "split": job["split"], "pool": job["pool"],
                "scene_seed": job["scene_seed"], "scene": scene_spec, "move": mv,
                "cam_pos": pos.tolist(), "target": target.tolist(), "D0": D0,
                "D_median": D, "depth_hits": int(len(depths)),
                "cameras": [M.tolist() for M in mats], "focal_mm": [float(x) for x in focals],
                "sensor_width_mm": moves.SENSOR_W_MM, "res": [moves.RES_W, moves.RES_H],
                "samples": SAMPLES, "render_s": round(time.time() - t0, 2),
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
