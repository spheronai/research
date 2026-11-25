"""Write a predicted camera path to the formats tools read.

Input: per-frame camera-to-world matrices in Blender convention (Z up, camera
looks down -Z, Y up in the image) and focal length in mm on a 36 mm sensor.
Translation is in whatever unit the caller passes; the model's raw output is
in units of the frame-0 median scene depth, so pass --scale <metres to the
subject> to get metres.

Formats:
  .txt   RealEstate10K / camera-control line format, one line per frame:
         `frame fx fy cx cy 0 0 <3x4 world-to-camera, row-major>`, intrinsics
         normalised by image width/height, OpenCV camera axes. This is what
         camera-controlled video models (CameraCtrl, Wan Fun Camera) read.
  .chan  Nuke camera channels, Y up: `frame tx ty tz rx ry rz hfov`. Import in
         Nuke with rotation order ZXY (its default for .chan).
  .usda  USD camera with time-sampled transform and focal length, Z up.
  .py    Blender script that creates a keyframed camera.

usage: python export_camera.py pred.json clip_id --out shot --fps 4 --scale 3.0
"""

import argparse
import json
import math

import numpy as np

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
ZUP_TO_YUP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)


def euler_zxy(R):
    """Angles (rx, ry, rz) in degrees with R = Ry @ Rx @ Rz (Nuke order ZXY)."""
    a = math.asin(max(-1.0, min(1.0, -R[1, 2])))
    b = math.atan2(R[0, 2], R[2, 2])
    c = math.atan2(R[1, 0], R[1, 1])
    return math.degrees(a), math.degrees(b), math.degrees(c)


def write_txt(path, c2w, focal, w, h):
    with open(path, "w") as f:
        for i, (M, fm) in enumerate(zip(c2w, focal)):
            w2c = np.linalg.inv(M @ FLIP)[:3]
            fx = fm / 36.0
            fy = fm / 36.0 * w / h
            vals = [i, fx, fy, 0.5, 0.5, 0, 0] + w2c.reshape(-1).tolist()
            f.write(" ".join(f"{v:.8g}" for v in vals) + "\n")


def write_chan(path, c2w, focal, frame0=1):
    with open(path, "w") as f:
        for i, (M, fm) in enumerate(zip(c2w, focal)):
            Y = ZUP_TO_YUP @ M
            rx, ry, rz = euler_zxy(Y[:3, :3])
            hfov = math.degrees(2 * math.atan(18.0 / fm))
            t = Y[:3, 3]
            f.write(f"{i + frame0} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {rx:.6f} {ry:.6f} {rz:.6f} {hfov:.6f}\n")


def write_usda(path, c2w, focal, fps, w, h):
    lines = ['#usda 1.0', '(', f'    framesPerSecond = {fps}', f'    timeCodesPerSecond = {fps}',
             '    startTimeCode = 0', f'    endTimeCode = {len(c2w) - 1}', '    upAxis = "Z"',
             '    metersPerUnit = 1', ')', '', 'def Camera "shot_camera"', '{',
             '    float horizontalAperture = 36', f'    float verticalAperture = {36.0 * h / w:.6f}',
             '    float focalLength.timeSamples = {']
    lines += [f'        {i}: {fm:.6f},' for i, fm in enumerate(focal)]
    lines += ['    }', '    matrix4d xformOp:transform.timeSamples = {']
    for i, M in enumerate(c2w):
        rows = ", ".join("(" + ", ".join(f"{v:.8g}" for v in r) + ")" for r in M.T)
        lines.append(f'        {i}: ({rows}),')
    lines += ['    }', '    uniform token[] xformOpOrder = ["xformOp:transform"]', '}', '']
    open(path, "w").write("\n".join(lines))


def write_blender(path, c2w, focal, fps, w, h):
    data = {"m": [M.tolist() for M in c2w], "f": [float(x) for x in focal]}
    open(path, "w").write(f'''import bpy
from mathutils import Matrix
D = {json.dumps(data)}
sc = bpy.context.scene
sc.render.fps = {fps}
sc.render.resolution_x, sc.render.resolution_y = {w}, {h}
sc.frame_start, sc.frame_end = 1, {len(c2w)}
cd = bpy.data.cameras.new("shot_camera")
cd.sensor_fit, cd.sensor_width = "HORIZONTAL", 36.0
cam = bpy.data.objects.new("shot_camera", cd)
sc.collection.objects.link(cam)
sc.camera = cam
for i, (m, f) in enumerate(zip(D["m"], D["f"])):
    cam.matrix_world = Matrix(m)
    cd.lens = f
    cam.keyframe_insert("location", frame=i + 1)
    cam.keyframe_insert("rotation_euler", frame=i + 1)
    cd.keyframe_insert("lens", frame=i + 1)
''')


def export(c2w, focal, out, fps=4, w=392, h=224, scale=1.0):
    c2w = np.array(c2w, float).copy()
    c2w[:, :3, 3] *= scale
    focal = np.asarray(focal, float)
    write_txt(out + ".txt", c2w, focal, w, h)
    write_chan(out + ".chan", c2w, focal)
    write_usda(out + ".usda", c2w, focal, fps, w, h)
    write_blender(out + "_blender.py", c2w, focal, fps, w, h)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pred")
    ap.add_argument("clip")
    ap.add_argument("--out", default="shot")
    ap.add_argument("--fps", type=float, default=4)
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    p = json.load(open(a.pred))[a.clip]
    export(p["c2w"], p["focal_mm"], a.out, a.fps, scale=a.scale)
