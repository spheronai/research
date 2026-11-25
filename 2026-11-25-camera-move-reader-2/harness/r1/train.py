"""Arm D: a small head on frozen Depth Anything 3 (DA3-Base) outputs.

Inputs per frame: DA3's camera relative to frame 0 (rotation 6D, translation
in DA3 frame-0 median-depth units), DA3 focal (log ratio to frame 0 and log
absolute), and DA3 backbone features pooled to 7x12 tokens.
Outputs: per-frame camera (rotation 6D, translation in scene-depth units,
log focal ratio, log absolute focal) and the move card (per-axis direction
and size, orbit, dolly zoom, handheld, easing).

usage: venv/bin/python train.py --data /ephemeral/study/data --out runs/seed0 --seed 0
       venv/bin/python train.py --predict runs/seed0 --splits test_scenes test_combos
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom  # noqa: E402
import labeler  # noqa: E402

AX = labeler.AXES
NF, NT, C = 16, 84, 768


def rot6d(R):
    return np.concatenate([R[..., :, 0], R[..., :, 1]], -1)


def rot_from_6d(x):
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], -1)


def geo_input(c2w, focal, D):
    rel = geom.relative(c2w)
    R = rel[:, :3, :3]
    t = rel[:, :3, 3] / max(D, 1e-6)
    f = np.asarray(focal, float)
    return np.concatenate([rot6d(R), t, np.log(f / f[0])[:, None], np.log(f / 36.0)[:, None]], 1).astype(np.float32)


def load_split(data, split, need_gt=True):
    clips = sorted(d for d in glob.glob(os.path.join(data, split, "*"))
                   if os.path.exists(os.path.join(d, "da3.npz")) and os.path.exists(os.path.join(d, "meta.json")))
    ids, geo, feats, tgt_path, tgt_cls = [], [], [], [], []
    for c in clips:
        z = np.load(os.path.join(c, "da3.npz"))
        if z["feats"].size == 0:
            continue
        ids.append(os.path.basename(c))
        geo.append(geo_input(z["c2w"], z["focal_mm"], float(z["D"])))
        feats.append(z["feats"].astype(np.float16))
        if not need_gt:  # prediction never reads test ground truth
            continue
        m = json.load(open(os.path.join(c, "meta.json")))
        G = np.array(m["cameras"])
        gf = np.array(m["focal_mm"])
        tgt_path.append(geo_input(G, gf, m["D_median"]))
        lab = labeler.label(G, gf, m["D_median"])
        t = labeler.targets(lab)
        tgt_cls.append([t[f"{a}_dir"] for a in AX] + [t[f"{a}_size"] for a in AX]
                       + [t["orbit"], t["dollyzoom"], t["handheld"], t["easing"], t["has_motion"]])
    if not need_gt:
        return ids, torch.tensor(np.stack(geo)), torch.tensor(np.stack(feats)), None, None
    return (ids, torch.tensor(np.stack(geo)), torch.tensor(np.stack(feats)),
            torch.tensor(np.stack(tgt_path)), torch.tensor(np.array(tgt_cls)))


class Head(nn.Module):
    def __init__(self, d=256, layers=6, heads=8):
        super().__init__()
        self.feat_in = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, d))
        self.geo_in = nn.Sequential(nn.Linear(11, d), nn.GELU(), nn.Linear(d, d))
        self.frame_emb = nn.Parameter(torch.zeros(NF, d))
        self.tok_emb = nn.Parameter(torch.zeros(NT + 1, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.frame_emb, std=0.02)
        nn.init.normal_(self.tok_emb, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d)
        self.path_out = nn.Linear(d, 11)
        nn.init.zeros_(self.path_out.weight)
        nn.init.zeros_(self.path_out.bias)
        n_cls = 7 * 3 + 7 * 4 + 3 + 2 + 2 + 4
        self.cls_out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, n_cls))

    def forward(self, geo, feats):
        B = geo.shape[0]
        f = self.feat_in(feats.float())                    # B,16,84,d
        g = self.geo_in(geo).unsqueeze(2)                   # B,16,1,d
        x = torch.cat([g, f], 2) + self.tok_emb[None, None] + self.frame_emb[None, :, None]
        x = x.reshape(B, NF * (NT + 1), -1)
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1)
        x = self.norm(self.tf(x))
        cls = x[:, 0]
        gt = x[:, 1:].reshape(B, NF, NT + 1, -1)[:, :, 0]   # geometry token per frame
        path = geo + self.path_out(gt)                       # residual on DA3's estimate
        pooled = torch.cat([cls, gt.mean(1)], -1)
        return path, self.cls_out(pooled)


def split_logits(z):
    out, i = {}, 0
    for a in AX:
        out[f"{a}_dir"] = z[:, i:i + 3]; i += 3
    for a in AX:
        out[f"{a}_size"] = z[:, i:i + 4]; i += 4
    for k, n in (("orbit", 3), ("dollyzoom", 2), ("handheld", 2), ("easing", 4)):
        out[k] = z[:, i:i + n]; i += n
    return out


def losses(path, logits, tp, tc):
    Rp = rot_from_6d(path[..., :6])
    Rt = rot_from_6d(tp[..., :6])
    l_rot = (Rp - Rt).pow(2).sum((-1, -2)).mean()
    l_t = F.smooth_l1_loss(path[..., 6:9], tp[..., 6:9], beta=0.02)
    l_f = F.smooth_l1_loss(path[..., 9:11], tp[..., 9:11], beta=0.02)
    lg = split_logits(logits)
    l_c = 0.0
    for j, a in enumerate(AX):
        l_c = l_c + F.cross_entropy(lg[f"{a}_dir"], tc[:, j]) + 0.5 * F.cross_entropy(lg[f"{a}_size"], tc[:, 7 + j])
    l_c = l_c + F.cross_entropy(lg["orbit"], tc[:, 14]) + F.cross_entropy(lg["dollyzoom"], tc[:, 15]) \
        + F.cross_entropy(lg["handheld"], tc[:, 16])
    moving = tc[:, 18] == 1
    if moving.any():
        l_c = l_c + F.cross_entropy(lg["easing"][moving], tc[moving, 17])
    total = 10 * l_rot + 10 * l_t + 10 * l_f + l_c
    return total, {"rot": float(l_rot), "t": float(l_t), "f": float(l_f), "cls": float(l_c)}


def train(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    tr = load_split(a.data, a.train_split)
    va = load_split(a.data, a.val_split)
    print(f"loaded train {len(tr[0])} val {len(va[0])} in {time.time() - t0:.0f}s", flush=True)
    dev = "cuda"
    model = Head().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * math.ceil(len(tr[0]) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05)
    best, log = 1e9, []
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr[0]))
        for i in range(0, len(perm), a.bs):
            idx = perm[i:i + a.bs]
            geo, feats, tp, tc = (x[idx].to(dev, non_blocking=True) for x in tr[1:])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                path, logits = model(geo, feats)
            loss, parts = losses(path.float(), logits.float(), tp, tc)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        vl = evaluate_loss(model, va, dev)
        log.append({"epoch": ep, "train_last": parts, "val": vl})
        print(f"epoch {ep} val {vl['total']:.4f} rot {vl['rot']:.4f} t {vl['t']:.4f} f {vl['f']:.4f} cls {vl['cls']:.4f} {time.time() - t0:.0f}s", flush=True)
        if vl["total"] < best:
            best = vl["total"]
            torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
    json.dump({"args": vars(a), "log": log, "best_val": best, "seconds": time.time() - t0}, open(os.path.join(a.out, "train_log.json"), "w"), indent=1)


@torch.no_grad()
def evaluate_loss(model, data, dev):
    model.eval()
    tot, n, agg = 0.0, 0, {}
    for i in range(0, len(data[0]), 64):
        geo, feats, tp, tc = (x[i:i + 64].to(dev) for x in data[1:])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            path, logits = model(geo, feats)
        loss, parts = losses(path.float(), logits.float(), tp, tc)
        b = len(geo)
        tot += float(loss) * b
        n += b
        for k, v in parts.items():
            agg[k] = agg.get(k, 0) + v * b
    return {"total": tot / n, **{k: v / n for k, v in agg.items()}}


@torch.no_grad()
def predict(a):
    dev = "cuda"
    model = Head().to(dev)
    model.load_state_dict(torch.load(os.path.join(a.predict, "model.pt"), map_location=dev))
    model.eval()
    for split in a.splits:
        ids, geo, feats, _, _ = load_split(a.data, split, need_gt=False)
        preds = {}
        for i in range(0, len(ids), 64):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                path, logits = model(geo[i:i + 64].to(dev), feats[i:i + 64].to(dev))
            path = path.float()
            R = rot_from_6d(path[..., :6]).cpu().numpy()
            t = path[..., 6:9].cpu().numpy()
            lf = path[..., 10].cpu().numpy()
            lg = {k: v.float().softmax(-1).cpu().numpy() for k, v in split_logits(logits.float()).items()}
            for b in range(len(R)):
                c2w = np.tile(np.eye(4), (NF, 1, 1))
                c2w[:, :3, :3] = R[b]
                c2w[:, :3, 3] = t[b]
                focal = 36.0 * np.exp(lf[b])
                preds[ids[i + b]] = {
                    "c2w": c2w.tolist(), "focal_mm": focal.tolist(), "D": 1.0,
                    "names": {k: int(v[b].argmax()) for k, v in lg.items()},
                }
        out = os.path.join(a.predict, f"pred_{split}.json")
        json.dump(preds, open(out, "w"))
        print("wrote", out, len(preds), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/ephemeral/study/data")
    ap.add_argument("--out", default="runs/seed0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--predict", default=None)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--splits", nargs="*", default=["test_scenes", "test_combos"])
    a = ap.parse_args()
    if a.predict:
        predict(a)
    else:
        train(a)
