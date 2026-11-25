"""Rule labeller: turn a camera path into a cinematographer's move card.

The same function labels ground-truth paths (training targets) and predicted
paths (baselines), so every method with a path gets a naming score.
Thresholds are fixed here before any rendering; changes after the plan
freeze are deviations.

Input: mats (N,4,4) camera-to-world in Blender convention (camera looks down
local -Z, +Y up, +X right), focals (N,) in mm, D = median scene depth at frame
0 in the same units as the translation (any unit; only ratios are used).
"""

import math

import numpy as np

AXES = ["pan", "tilt", "roll", "dolly", "truck", "pedestal", "zoom"]
SIZES = ["none", "small", "medium", "large"]
EASINGS = ["linear", "ease_in", "ease_out", "ease_in_out"]

# size bin edges: (none below, small below, medium below, large above)
ROT_BINS = (3.0, 10.0, 25.0)        # degrees
TRANS_BINS = (0.03, 0.12, 0.30)     # fraction of median scene depth
ZOOM_BINS = (0.05, 0.20, 0.50)      # |ln(f_end / f_0)|
HANDHELD_ROT_RMS = 0.05             # degrees, residual after a degree-5 fit
HANDHELD_TRANS_RMS = 0.0008         # fraction of D
POLY_DEG = 5

NAMES = {
    "pan": ("pan right", "pan left"),           # (negative, positive)
    "tilt": ("tilt down", "tilt up"),
    "roll": ("roll right", "roll left"),
    "dolly": ("pull out", "push in"),
    "truck": ("truck left", "truck right"),
    "pedestal": ("pedestal down", "pedestal up"),
    "zoom": ("zoom out", "zoom in"),
}


def yxz_angles(R):
    """R = Ry(yaw) Rx(pitch) Rz(roll), degrees."""
    pitch = math.asin(max(-1.0, min(1.0, -R[1, 2])))
    yaw = math.atan2(R[0, 2], R[2, 2])
    roll = math.atan2(R[1, 0], R[1, 1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def relative_motion(mats, focals, D):
    """Per-frame [yaw, pitch, roll (deg), fwd, right, up (/D), ln focal ratio]."""
    mats = np.asarray(mats, float)
    R0 = mats[0, :3, :3]
    p0 = mats[0, :3, 3]
    out = np.zeros((len(mats), 7))
    for i, M in enumerate(mats):
        Rr = R0.T @ M[:3, :3]
        y, p, r = yxz_angles(Rr)
        t = R0.T @ (M[:3, 3] - p0) / D
        out[i] = [y, p, r, -t[2], t[0], t[1], math.log(focals[i] / focals[0])]
    return out


def _bin(v, edges):
    a = abs(v)
    if a < edges[0]:
        return 0
    if a < edges[1]:
        return 1
    if a < edges[2]:
        return 2
    return 3


def _poly_smooth(y):
    x = np.linspace(-1, 1, len(y))
    c = np.polynomial.polynomial.polyfit(x, y, POLY_DEG)
    return np.polynomial.polynomial.polyval(x, c)


def label(mats, focals, D, hh_rot=None, hh_trans=None):
    """hh_rot / hh_trans override the handheld thresholds (per-method calibration on val)."""
    m = relative_motion(mats, focals, D)
    smooth = np.stack([_poly_smooth(m[:, k]) for k in range(7)], axis=1)
    resid = m - smooth
    rot_rms = float(np.sqrt(np.mean(resid[:, :3] ** 2)))
    trans_rms = float(np.sqrt(np.mean(resid[:, 3:6] ** 2)))
    hr = HANDHELD_ROT_RMS if hh_rot is None else hh_rot
    ht = HANDHELD_TRANS_RMS if hh_trans is None else hh_trans
    handheld = rot_rms > hr or trans_rms > ht

    end = smooth[-1] - smooth[0]
    vals = {"pan": end[0], "tilt": end[1], "roll": end[2], "dolly": end[3],
            "truck": end[4], "pedestal": end[5], "zoom": end[6]}
    axes = {}
    for ax in AXES:
        edges = ROT_BINS if ax in ("pan", "tilt", "roll") else (ZOOM_BINS if ax == "zoom" else TRANS_BINS)
        size = _bin(vals[ax], edges)
        direction = 0 if size == 0 else (1 if vals[ax] > 0 else -1)
        axes[ax] = {"dir": direction, "size": size, "value": float(vals[ax])}

    # orbit: lateral travel while turning back toward the subject
    orbit = 0
    if axes["truck"]["dir"] != 0 and axes["pan"]["dir"] != 0:
        # truck right (+) with pan left (+ yaw) keeps a subject in frame
        if axes["truck"]["dir"] == axes["pan"]["dir"]:
            orbit = axes["truck"]["dir"]  # +1 orbit right, -1 orbit left
    # forward travel that an arc explains is not a separate dolly
    dolly_in_orbit = bool(orbit and axes["dolly"]["dir"] > 0
                          and abs(axes["dolly"]["value"]) <= 0.6 * abs(axes["truck"]["value"]))
    dollyzoom = (axes["dolly"]["dir"] != 0 and axes["zoom"]["dir"] != 0
                 and axes["dolly"]["dir"] != axes["zoom"]["dir"] and not dolly_in_orbit)
    static = all(axes[a]["dir"] == 0 for a in AXES)

    easing = None
    if not static:
        scale = np.array([25.0, 25.0, 25.0, 0.3, 0.3, 0.3, 0.5])
        disp = (smooth - smooth[0]) / scale
        s = np.linalg.norm(disp, axis=1)
        if s[-1] > 1e-6:
            p = s / s[-1]
            u = np.linspace(0, 1, len(p))
            templates = {
                "linear": u,
                "ease_in": u ** 2,
                "ease_out": 1 - (1 - u) ** 2,
                "ease_in_out": 0.5 - 0.5 * np.cos(math.pi * u),
            }
            easing = min(templates, key=lambda k: float(np.sum((p - templates[k]) ** 2)))

    return {
        "axes": axes,
        "orbit": orbit,
        "dollyzoom": bool(dollyzoom),
        "handheld": bool(handheld),
        "static": bool(static),
        "easing": easing,
        "rot_resid_rms": rot_rms,
        "trans_resid_rms": trans_rms,
        "card": card({"axes": axes, "orbit": orbit, "dollyzoom": dollyzoom,
                      "handheld": handheld, "static": static, "easing": easing}),
    }


def card(lab):
    """Short human move card, e.g. 'push in (medium), tilt up (small), ease in-out, handheld'."""
    ax = lab["axes"]
    parts = []
    skip = set()
    if lab["orbit"]:
        size = max(ax["truck"]["size"], ax["pan"]["size"])
        parts.append(f"orbit {'right' if lab['orbit'] > 0 else 'left'} ({SIZES[size]})")
        skip |= {"truck", "pan"}
        # an arc also moves the camera forward by r(1 - cos a); that is part
        # of the orbit, not a separate push-in, while it stays under 0.6x the
        # lateral travel (tan(a/2) for arcs up to 60 degrees)
        if ax["dolly"]["dir"] > 0 and abs(ax["dolly"]["value"]) <= 0.6 * abs(ax["truck"]["value"]):
            skip.add("dolly")
    if lab["dollyzoom"]:
        d = "push in, zoom out" if ax["dolly"]["dir"] > 0 else "pull out, zoom in"
        parts.append(f"dolly zoom ({d})")
        skip |= {"dolly", "zoom"}
    for a in AXES:
        if a in skip or ax[a]["dir"] == 0:
            continue
        name = NAMES[a][1 if ax[a]["dir"] > 0 else 0]
        if a == "pedestal" and ax[a]["size"] == 3:
            name = name.replace("pedestal", "crane")
        parts.append(f"{name} ({SIZES[ax[a]['size']]})")
    if lab["static"]:
        parts = ["static"]
    elif lab["easing"]:
        parts.append(lab["easing"].replace("_", " ").replace("in out", "in-out"))
    if lab["handheld"]:
        parts.append("handheld")
    return ", ".join(parts)


def targets(lab):
    """Integer targets for training: per axis dir class (0 neg, 1 none, 2 pos) and size."""
    t = {}
    for a in AXES:
        t[f"{a}_dir"] = lab["axes"][a]["dir"] + 1
        t[f"{a}_size"] = lab["axes"][a]["size"]
    t["orbit"] = lab["orbit"] + 1
    t["dollyzoom"] = int(lab["dollyzoom"])
    t["handheld"] = int(lab["handheld"])
    t["easing"] = EASINGS.index(lab["easing"]) if lab["easing"] else 0
    t["has_motion"] = int(not lab["static"])
    return t
