"""Shared geometry helpers: conventions, metrics, alignment."""

import math

import numpy as np

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV camera <-> Blender camera


def cv_c2w_to_blender(c2w_cv):
    return np.asarray(c2w_cv) @ FLIP


def blender_c2w_to_cv(c2w_bl):
    return np.asarray(c2w_bl) @ FLIP


def focal_mm_from_K(K, width_px, sensor_w=36.0):
    return float(K[0, 0]) / float(width_px) * sensor_w


def relative(mats):
    """Express cameras in the frame-0 camera frame (gauge-free)."""
    mats = np.asarray(mats, float)
    inv0 = np.linalg.inv(mats[0])
    return np.einsum("ij,njk->nik", inv0, mats)


def geodesic_deg(Ra, Rb):
    R = Ra.T @ Rb
    c = (np.trace(R) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def umeyama_sim3(src, dst):
    """Similarity transform (s, R, t) minimising |dst - (s R src + t)|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    var_s = (xs ** 2).sum() / len(src)
    if var_s < 1e-12:
        return 1.0, np.eye(3), mu_d - mu_s
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


def path_metrics(pred_mats, pred_focal, pred_D, gt_mats, gt_focal, gt_D):
    """Rotation error (deg), ATE (in GT median-depth units), focal errors."""
    P = relative(pred_mats)
    G = relative(gt_mats)
    rot = float(np.mean([geodesic_deg(P[i, :3, :3], G[i, :3, :3]) for i in range(1, len(G))]))
    ps = P[:, :3, 3] / pred_D
    gs = G[:, :3, 3] / gt_D
    extent = float(np.linalg.norm(gs - gs[0], axis=1).max())
    if extent > 0.03:
        s, R, t = umeyama_sim3(ps, gs)
        aligned = (s * (R @ ps.T)).T + t
        ate = float(np.sqrt(np.mean(np.sum((aligned - gs) ** 2, axis=1))))
    else:
        ate = None  # no camera travel: ATE is not defined after scale alignment
    # travel the method reports when the true camera did not move
    drift = float(np.linalg.norm(ps - ps[0], axis=1).max()) if extent <= 0.03 else None
    pf, gf = np.asarray(pred_focal, float), np.asarray(gt_focal, float)
    zoom_err = float(np.mean(np.abs(np.log(pf / pf[0]) - np.log(gf / gf[0]))))
    focal_err = float(np.mean(np.abs(np.log(pf) - np.log(gf))))
    return {"rot_err_deg": rot, "ate": ate, "static_drift": drift, "zoom_err": zoom_err,
            "focal_abs_err": focal_err, "gt_extent": extent}
