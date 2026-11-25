import json, numpy as np, glob, os, sys
sys.path.insert(0, "/ephemeral/study/harness")
import labeler
AX = labeler.AXES
W = {"pan": ("right", "left"), "tilt": ("down", "up"), "roll": ("right", "left"), "dolly": ("out", "in"),
     "truck": ("left", "right"), "pedestal": ("down", "up"), "zoom": ("out", "in")}
def fmt(n):
    s = []
    for a in AX:
        v = n.get(a + "_dir", 1)
        if v != 1:
            s.append(a + " " + (W[a][0] if v == 0 else W[a][1]))
    if n.get("orbit", 1) != 1:
        s.append("orbit " + ("left" if n["orbit"] == 0 else "right"))
    if n.get("handheld"):
        s.append("handheld")
    return ", ".join(s) or "static"
R = "/ephemeral/study"
hh = json.load(open(R + "/results/results_test_scenes.json"))["arms"]["B_da3_base_rules"]["handheld_threshold_deg"]
vlm = json.load(open(R + "/runs/vlm/pred_real_tos.json"))
d0 = json.load(open(R + "/runs/seed0/pred_real_tos.json"))
out = {}
for c in sorted(glob.glob(R + "/data/real_tos/*")):
    k = os.path.basename(c)
    z = np.load(c + "/da3.npz")
    b = labeler.targets(labeler.label(z["c2w"], z["focal_mm"], float(z["D"]), hh_rot=hh, hh_trans=1e9))
    out[k] = {"start_s": json.load(open(c + "/meta.json"))["start_s"], "B_da3_rules": fmt(b), "C_qwen3_vl": fmt(vlm[k]["names"] or {}),
              "D_ours": fmt(d0[k]["names"]), "da3_focal_first_last": [round(float(z["focal_mm"][0]), 1), round(float(z["focal_mm"][-1]), 1)],
              "ours_focal_first_last": [round(d0[k]["focal_mm"][0], 1), round(d0[k]["focal_mm"][-1], 1)]}
    print(k, out[k])
json.dump(out, open(R + "/results/real_tos_cards.json", "w"), indent=1)
