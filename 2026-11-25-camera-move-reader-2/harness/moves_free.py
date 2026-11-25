"""Run two, source F: free-form camera paths that are not built from named primitives.

Each of the 7 axes (yaw, pitch, roll, forward, right, up, log focal) switches
on independently with probability 0.35, with its own signed amplitude and
easing, running over the whole 4 s.

Pilot change (24 Nov 2026, before the plan freeze): the plan also gave each
channel its own start/stop time inside the 4 s. On 400 shake-free paths that
made 84.8% read as handheld, because the labeller's degree-5 fit cannot follow
a move that starts or stops mid-clip, against 0% for full-length channels and
0% for the grammar. Start and stop were dropped so the handheld label keeps
meaning shake.

With probability 0.3 the camera keeps looking at the frame-0 target ("look-at
mode"), which turns sideways travel into arcs and orbits; yaw and pitch then
come from the look-at, not from their own channels. Handheld noise (30%) is
run one's.

Held-out pairs are NOT excluded here. render_clips2.py labels the path as
placed and rejects any training clip whose card contains a held-out pair.

Pure numpy; runs inside Blender's Python and outside it.
"""

import math

import numpy as np

from moves import DURATION_S, EASINGS, N_FRAMES, _rot, _shake, ease, look_at

AXES = ["yaw", "pitch", "roll", "fwd", "right", "up", "logf"]
P_ON = 0.35
P_LOOKAT = 0.30
# amplitude ranges: degrees, fraction of the distance to the target, ln focal ratio
AMP = {
    "yaw": (3.0, 45.0),
    "pitch": (3.0, 30.0),
    "roll": (3.0, 20.0),
    "fwd": (0.03, 0.45),
    "right": (0.03, 0.45),
    "up": (0.03, 0.40),
    "logf": (0.05, 0.80),
}
FOCALS = [18, 24, 28, 35, 50, 65, 85]


def sample_free(rng):
    look = bool(rng.random() < P_LOOKAT)
    chans = {}
    for ax in AXES:
        if rng.random() >= P_ON:
            continue
        if look and ax in ("yaw", "pitch"):
            continue
        lo, hi = AMP[ax]
        s, e = 0.0, 1.0
        chans[ax] = {"amp": float(rng.uniform(lo, hi) * rng.choice([-1.0, 1.0])),
                     "easing": str(rng.choice(EASINGS)), "start": s, "end": e}
    spec = {
        "source": "F",
        "primitives": ["free"],
        "channels": chans,
        "look_at": look,
        "handheld": bool(rng.random() < 0.30),
        "focal_mm": float(rng.choice(FOCALS)),
        "shake_seed": int(rng.integers(0, 2**31 - 1)),
    }
    if spec["handheld"]:
        spec["shake_rot_deg"] = float(rng.uniform(0.4, 1.4))
        spec["shake_trans"] = float(rng.uniform(0.004, 0.012))
    return spec


def _chan(c, u):
    if c is None:
        return np.zeros_like(u)
    w = np.clip((u - c["start"]) / max(c["end"] - c["start"], 1e-6), 0.0, 1.0)
    return c["amp"] * ease(w, c["easing"])


def trajectory_free(spec, cam_pos, target):
    """Per-frame (4x4 camera-to-world, focal_mm), D0 for a free-form spec."""
    cam_pos = np.asarray(cam_pos, float)
    target = np.asarray(target, float)
    D0 = float(np.linalg.norm(target - cam_pos))
    R0 = look_at(cam_pos, target)
    u = np.linspace(0.0, 1.0, N_FRAMES)
    ch = spec["channels"]
    v = {ax: _chan(ch.get(ax), u) for ax in AXES}
    shake = _shake(spec, u * DURATION_S) if spec.get("handheld") else np.zeros((N_FRAMES, 6))
    mats, focals = [], []
    for i in range(N_FRAMES):
        pos = cam_pos + R0 @ (np.array([v["right"][i], v["up"][i], -v["fwd"][i]]) * D0)
        if spec["look_at"]:
            if np.linalg.norm(target - pos) < 0.2 * D0:  # never fly through the target
                return None
            R_base = look_at(pos, target)
            R_local = _rot("z", v["roll"][i] + shake[i, 2]) @ _rot("y", shake[i, 0]) @ _rot("x", shake[i, 1])
        else:
            R_base = R0
            R_local = (_rot("y", v["yaw"][i] + shake[i, 0]) @ _rot("x", v["pitch"][i] + shake[i, 1])
                       @ _rot("z", v["roll"][i] + shake[i, 2]))
        M = np.eye(4)
        M[:3, :3] = R_base @ R_local
        M[:3, 3] = pos + R0 @ (shake[i, 3:] * D0)
        mats.append(M)
        focals.append(spec["focal_mm"] * math.exp(v["logf"][i]))
    return np.array(mats), np.array(focals), D0
