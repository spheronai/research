"""Blender: dump the active scene camera animation from a Big Buck Bunny scene file.

Per frame: camera name, camera-to-world matrix, lens (mm), sensor width and fit.
Also the median ray-hit depth from the camera on a 24x14 grid every 10 frames,
used to put the translation in scene-depth units.
"""
import bpy, json, sys, math
import numpy as np
from mathutils import Vector

out = sys.argv[sys.argv.index("--") + 1]
sc = bpy.context.scene
res = {"file": bpy.data.filepath, "scene": sc.name, "frame_start": sc.frame_start, "frame_end": sc.frame_end,
       "fps": sc.render.fps / sc.render.fps_base, "res": [sc.render.resolution_x, sc.render.resolution_y],
       "markers": [(m.frame, m.camera.name if m.camera else None) for m in sc.timeline_markers], "frames": [], "depth": {}}
W, H = sc.render.resolution_x, sc.render.resolution_y


def depth(cam, f):
    ds = _depth(cam, sc, bpy.context.evaluated_depsgraph_get())
    bs = sc.background_set
    while not ds[1] and bs is not None:  # geometry often lives in a linked background scene
        bs.frame_set(f)
        vl = bs.view_layers[0]
        ds = _depth(cam, bs, vl.depsgraph)
        bs = bs.background_set
    return ds


def _depth(cam, scene, dg):
    M = cam.matrix_world
    o = M.translation
    sw = cam.data.sensor_width
    f = cam.data.lens
    hw = sw / 2 / f
    hh = hw * H / W
    ds = []
    for i in range(24):
        for j in range(14):
            x = (-1 + 2 * (i + 0.5) / 24) * hw
            y = (-1 + 2 * (j + 0.5) / 14) * hh
            d = (M.to_3x3() @ Vector((x, y, -1.0))).normalized()
            hit, loc, *_ = scene.ray_cast(dg, o, d, distance=1e4)
            if hit:
                ds.append(((loc - o) @ (M.to_3x3() @ Vector((0, 0, -1.0)))))
    return (float(np.median(ds)), len(ds)) if ds else (None, 0)


if sc.camera:
    for f in range(sc.frame_start, sc.frame_end + 1):
        sc.frame_set(f)
        c = sc.camera
        res["frames"].append({"f": f, "cam": c.name, "m": [list(r) for r in c.matrix_world], "lens": c.data.lens,
                              "sensor": c.data.sensor_width, "fit": c.data.sensor_fit, "type": c.data.type})
        if (f - sc.frame_start) % 10 == 0:
            res["depth"][f] = depth(c, f)
json.dump(res, open(out, "w"))
hits = [v[1] for v in res["depth"].values()]
print("CAMDUMP", res["scene"], res["frame_start"], res["frame_end"], res["fps"], len(res["frames"]), "depth hits median", (sorted(hits)[len(hits)//2] if hits else 0))
