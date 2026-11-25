"""Run Depth Anything 3 (DA3-Base, Apache-2.0) on every clip.

Writes <clip>/da3.npz with per-frame Blender-convention camera-to-world
matrices, focal length in mm (36 mm sensor), median frame-0 depth for
scale, and pooled backbone features for the trained head.

usage: venv-da3/bin/python da3_infer.py data/test_scenes [data/val ...]
"""

import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from depth_anything_3.api import DepthAnything3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom  # noqa: E402

MODEL = os.environ.get("DA3_MODEL", "depth-anything/DA3-BASE")
POOL_HW = (7, 12)  # 84 tokens per frame (the 21x36 grid pooled 3x3)


def main():
    model = DepthAnything3.from_pretrained(MODEL).to("cuda").eval()
    n_layers = None
    for split_dir in sys.argv[1:]:
        clips = sorted(d for d in glob.glob(os.path.join(split_dir, "*")) if os.path.exists(os.path.join(d, "meta.json")))
        sh, ns = int(os.environ.get("DA3_SHARD", "0")), int(os.environ.get("DA3_NSHARDS", "1"))
        clips = [c for c in clips if int(os.path.basename(c).split("_")[-1]) % ns == sh]
        t0, done = time.time(), 0
        for c in clips:
            out = os.path.join(c, "da3.npz")
            if os.path.exists(out):
                continue
            frames = sorted(glob.glob(os.path.join(c, "f*.png")))
            if len(frames) != 16:
                continue
            if n_layers is None:
                n_layers = 11  # last block of the ViT-B backbone
            with torch.no_grad():
                pred = model.inference(frames, ref_view_strategy="first", export_feat_layers=[n_layers])
            ext = np.asarray(pred.extrinsics)
            if ext.shape[-2] == 3:
                e4 = np.tile(np.eye(4), (len(ext), 1, 1))
                e4[:, :3, :] = ext
                ext = e4
            c2w_cv = np.linalg.inv(ext)
            c2w_bl = np.stack([geom.cv_c2w_to_blender(m) for m in c2w_cv])
            K = np.asarray(pred.intrinsics)
            H, W = pred.depth.shape[1:]
            focal = np.array([geom.focal_mm_from_K(k, W) for k in K])
            d0 = np.asarray(pred.depth[0])
            conf = pred.conf[0] if pred.conf is not None else None
            if conf is not None:
                d0 = d0[conf >= np.percentile(conf, 40)]
            D = float(np.median(d0))
            feats = None
            aux = pred.aux or {}
            key = next((k for k in aux if "feat" in k), None)
            if key is not None:
                f = torch.as_tensor(np.asarray(aux[key]))  # N, h, w, C
                f = f.permute(0, 3, 1, 2).float()
                f = F.adaptive_avg_pool2d(f, POOL_HW)  # N, C, 4, 7
                feats = f.flatten(2).transpose(1, 2).half().numpy()  # N, 28, C
            np.savez_compressed(out, c2w=c2w_bl.astype(np.float32), focal_mm=focal.astype(np.float32),
                                D=np.float32(D), K=K.astype(np.float32), hw=np.array([H, W]),
                                feats=feats if feats is not None else np.zeros(0, np.float16))
            done += 1
            if done % 100 == 0:
                print(f"PROGRESS {split_dir} {done}/{len(clips)} {(time.time() - t0) / done:.2f}s/clip", flush=True)
        print(f"DONE {split_dir} {done} clips {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
