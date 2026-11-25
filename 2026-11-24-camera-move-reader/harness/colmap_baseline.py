"""Arm A: structure-from-motion with COLMAP (pycolmap), per-image focal length.

For each clip: SIFT features, exhaustive matching over 16 frames, incremental
mapping with one camera per image (so a zoom can be represented), up to 3
tries with different seeds. A clip counts as failed if fewer than 16 frames
register. Writes <clip>/colmap.npz (Blender-convention c2w, focal mm, D).

usage: venv/bin/python colmap_baseline.py data/test_scenes [--workers 20]
"""

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom  # noqa: E402


MODE = os.environ.get("COLMAP_MODE", "per_image")


def run_clip(clip):
    import pycolmap

    out = os.path.join(clip, "colmap.npz" if MODE == "per_image" else "colmap_single.npz")
    if os.path.exists(out):
        return clip, "cached"
    frames = sorted(glob.glob(os.path.join(clip, "f*.png")))
    names = [os.path.basename(f) for f in frames]
    best = None
    tries = []
    for seed in range(3):
        work = tempfile.mkdtemp(prefix="colmap_", dir="/ephemeral/study/tmp")
        try:
            db = os.path.join(work, "db.db")
            opts = pycolmap.FeatureExtractionOptions()
            opts.num_threads = 1
            opts.sift.max_num_features = 8192
            ropts = pycolmap.ImageReaderOptions()
            ropts.camera_model = "SIMPLE_RADIAL"
            pycolmap.extract_features(db, clip, image_names=names, camera_mode=(pycolmap.CameraMode.PER_IMAGE if MODE == "per_image" else pycolmap.CameraMode.SINGLE),
                                      reader_options=ropts, extraction_options=opts,
                                      device=pycolmap.Device.cpu)
            mopts = pycolmap.FeatureMatchingOptions()
            mopts.num_threads = 1
            pycolmap.match_exhaustive(db, matching_options=mopts, device=pycolmap.Device.cpu)
            popts = pycolmap.IncrementalPipelineOptions()
            popts.num_threads = 1
            popts.random_seed = seed
            popts.mapper.init_min_num_inliers = 50
            recs = pycolmap.incremental_mapping(db, clip, os.path.join(work, "sparse"), options=popts)
            if recs:
                rec = max(recs.values(), key=lambda r: r.num_reg_images())
                n = rec.num_reg_images()
                tries.append(n)
                if best is None or n > best[0]:
                    best = (n, rec)
            else:
                tries.append(0)
        except Exception as e:  # a crash is a failure, recorded
            tries.append(f"error: {repr(e)[:80]}")
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if best and best[0] == 16:
            break
    if not best or best[0] < 16:
        np.savez(out, failed=True, registered=best[0] if best else 0, tries=json.dumps(tries))
        return clip, "failed"
    rec = best[1]
    by_name = {img.name: img for img in rec.images.values()}
    c2w, focal = [], []
    for nm in names:
        img = by_name[nm]
        w2c = np.eye(4)
        w2c[:3, :] = img.cam_from_world().matrix()
        c2w.append(geom.cv_c2w_to_blender(np.linalg.inv(w2c)))
        cam = rec.cameras[img.camera_id]
        focal.append(cam.params[0] / cam.width * 36.0)
    # scale: median depth of frame-0 observed points
    img0 = by_name[names[0]]
    w2c0 = np.eye(4)
    w2c0[:3, :] = img0.cam_from_world().matrix()
    depths = []
    for p2d in img0.points2D:
        if p2d.has_point3D():
            X = rec.points3D[p2d.point3D_id].xyz
            depths.append((w2c0[:3, :3] @ X + w2c0[:3, 3])[2])
    D = float(np.median(depths)) if depths else 1.0
    np.savez(out, failed=False, c2w=np.array(c2w, np.float32), focal_mm=np.array(focal, np.float32),
             D=np.float32(D), registered=16, tries=json.dumps(tries))
    return clip, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("splits", nargs="+")
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    os.makedirs("/ephemeral/study/tmp", exist_ok=True)
    for sd in a.splits:
        clips = sorted(d for d in glob.glob(os.path.join(sd, "*")) if os.path.exists(os.path.join(d, "meta.json")))
        t0 = time.time()
        counts = {}
        with ProcessPoolExecutor(a.workers) as ex:
            for i, (_, st) in enumerate(ex.map(run_clip, clips, chunksize=2)):
                counts[st] = counts.get(st, 0) + 1
                if (i + 1) % 100 == 0:
                    print(f"PROGRESS {sd} {i + 1}/{len(clips)} {counts} {time.time() - t0:.0f}s", flush=True)
        print(f"DONE {sd} {counts} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
