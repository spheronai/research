"""Camera move grammar: sample a move spec and turn it into per-frame cameras.

Pure numpy so it runs both inside Blender's Python and outside it.

Conventions (Blender): world Z is up. A camera looks down its local -Z axis,
local +Y is image up, local +X is image right. A camera is stored as a 4x4
camera-to-world matrix plus a focal length in mm on a 36 mm wide sensor.
"""

import math

import numpy as np

N_FRAMES = 16          # frames per clip, sampled evenly from the move
DURATION_S = 4.0       # move length at 24 fps (96 frames, we keep 16)
SENSOR_W_MM = 36.0
RES_W, RES_H = 392, 224

PRIMITIVES = ["pan", "tilt", "roll", "dolly", "truck", "pedestal", "zoom", "orbit", "dollyzoom"]

# Move pairs that never appear together in training (held-out compositions).
HELDOUT_PAIRS = [("tilt", "truck"), ("pan", "pedestal"), ("zoom", "pedestal"), ("roll", "dolly"), ("orbit", "tilt")]

EASINGS = ["linear", "ease_in", "ease_out", "ease_in_out"]


def ease(u, kind):
    u = np.clip(u, 0.0, 1.0)
    if kind == "linear":
        return u
    if kind == "ease_in":
        return u ** 2
    if kind == "ease_out":
        return 1.0 - (1.0 - u) ** 2
    if kind == "ease_in_out":
        return 0.5 - 0.5 * np.cos(math.pi * u)
    raise ValueError(kind)


def _signed(rng, lo, hi):
    return float(rng.uniform(lo, hi) * rng.choice([-1.0, 1.0]))


# Amplitude ranges. Angles in degrees, distances as a fraction of the start
# distance to the look-at target, zoom as a log focal ratio.
AMP = {
    "pan": (5.0, 45.0),
    "tilt": (4.0, 30.0),
    "roll": (4.0, 20.0),
    "dolly": (0.06, 0.45),
    "truck": (0.06, 0.45),
    "pedestal": (0.06, 0.40),
    "zoom": (0.10, 0.80),
    "orbit": (10.0, 50.0),
    "dollyzoom": (0.10, 0.35),
}


def pair_is_heldout(prims):
    s = set(prims)
    return any(a in s and b in s for a, b in HELDOUT_PAIRS)


def sample_move(rng, allow_heldout=False, require_heldout=False):
    """Return a move spec dict. Labels are NOT taken from this; see labeler.py."""
    r = rng.random()
    if r < 0.07 and not require_heldout:
        prims = []
    else:
        k = 1 if r < 0.45 else (2 if r < 0.88 else 3)
        if require_heldout and k == 1:  # a held-out pair needs two primitives
            k = 2
        for _ in range(200):
            prims = list(rng.choice(PRIMITIVES, size=k, replace=False))
            # orbit and dollyzoom already contain pan/truck and dolly/zoom
            if "orbit" in prims and ({"pan", "truck"} & set(prims)):
                continue
            if "dollyzoom" in prims and ({"dolly", "zoom"} & set(prims)):
                continue
            if "orbit" in prims and "dollyzoom" in prims:
                continue
            held = pair_is_heldout(prims)
            if require_heldout and not held:
                continue
            if held and not (allow_heldout or require_heldout):
                continue
            break
        else:
            raise RuntimeError("could not sample a move")
    amps = {}
    for p in prims:
        lo, hi = AMP[p]
        amps[p] = _signed(rng, lo, hi)
    spec = {
        "primitives": prims,
        "amplitudes": amps,
        "easing": str(rng.choice(EASINGS)),
        "handheld": bool(rng.random() < 0.30),
        "focal_mm": float(rng.choice([18, 24, 28, 35, 50, 65, 85])),
        "shake_seed": int(rng.integers(0, 2**31 - 1)),
    }
    if spec["handheld"]:
        spec["shake_rot_deg"] = float(rng.uniform(0.4, 1.4))
        spec["shake_trans"] = float(rng.uniform(0.004, 0.012))
    return spec


def _rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _axis_rot(axis, deg):
    """Rodrigues rotation about a unit axis."""
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    a = math.radians(deg)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


def look_at(pos, target):
    """Camera-to-world rotation for a camera at pos looking at target, roll 0."""
    fwd = np.asarray(target, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd)
    up_w = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up_w)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    # columns: local x (right), local y (up), local z (backward)
    return np.stack([right, up, -fwd], axis=1)


def _shake(spec, t):
    """Smooth handheld noise: sum of three sines per channel, 1 to 4 Hz."""
    rng = np.random.default_rng(spec["shake_seed"])
    out = np.zeros((len(t), 6))
    for ch in range(6):
        for _ in range(3):
            f = rng.uniform(1.0, 4.0)
            ph = rng.uniform(0, 2 * math.pi)
            out[:, ch] += np.sin(2 * math.pi * f * t + ph) / 3.0
    out[:, :3] *= spec["shake_rot_deg"]
    out[:, 3:] *= spec["shake_trans"]
    return out


def trajectory(spec, cam_pos, target):
    """Per-frame (4x4 camera-to-world, focal_mm) for a move spec.

    Movements are expressed in the frame-0 camera frame so that "dolly" means
    along the starting view axis, as an operator would describe it.
    """
    cam_pos = np.asarray(cam_pos, float)
    target = np.asarray(target, float)
    D0 = float(np.linalg.norm(target - cam_pos))
    R0 = look_at(cam_pos, target)
    right0, up0, back0 = R0[:, 0], R0[:, 1], R0[:, 2]
    fwd0 = -back0
    u = np.linspace(0.0, 1.0, N_FRAMES)
    e = ease(u, spec["easing"])
    t_sec = u * DURATION_S
    a = spec["amplitudes"]
    f0 = spec["focal_mm"]
    shake = _shake(spec, t_sec) if spec.get("handheld") else np.zeros((N_FRAMES, 6))

    mats, focals = [], []
    for i in range(N_FRAMES):
        pos = cam_pos.copy()
        yaw = pitch = roll = 0.0
        focal = f0
        if "dolly" in a:
            pos += fwd0 * a["dolly"] * D0 * e[i]
        if "truck" in a:
            pos += right0 * a["truck"] * D0 * e[i]
        if "pedestal" in a:
            pos += up0 * a["pedestal"] * D0 * e[i]
        if "pan" in a:
            yaw += a["pan"] * e[i]
        if "tilt" in a:
            pitch += a["tilt"] * e[i]
        if "roll" in a:
            roll += a["roll"] * e[i]
        if "zoom" in a:
            focal = f0 * math.exp(a["zoom"] * e[i])
        R_extra = np.eye(3)
        if "orbit" in a:
            # rotate the camera position around the target about world up and
            # keep looking at the target
            # about the start camera's up axis through the target, so the
            # target stays put in frame and no roll is introduced
            ang = a["orbit"] * e[i]
            Ra = _axis_rot(up0, ang)
            pos = target + Ra @ (pos - target)
            R_base = Ra @ R0
        elif "dollyzoom" in a:
            # push in (positive) or pull out (negative) and change focal length
            # so the target keeps its size in frame
            d = a["dollyzoom"] * D0 * e[i]
            pos = pos + fwd0 * d
            focal = focal * (D0 - d) / D0
            R_base = R0
        else:
            R_base = R0
        # operator moves in the camera's own axes: pan about up, tilt about right
        R_local = _rot("y", yaw + shake[i, 0]) @ _rot("x", pitch + shake[i, 1]) @ _rot("z", roll + shake[i, 2])
        R = R_base @ R_extra @ R_local
        pos = pos + R0 @ (shake[i, 3:] * D0)
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = pos
        mats.append(M)
        focals.append(focal)
    return np.array(mats), np.array(focals), D0


def path_points(spec, cam_pos, target):
    mats, _, _ = trajectory(spec, cam_pos, target)
    return mats[:, :3, 3]
