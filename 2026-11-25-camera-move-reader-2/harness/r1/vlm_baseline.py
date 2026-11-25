"""Arm C: an open vision-language model names the move, zero-shot.

Model: Qwen/Qwen3-VL-8B-Instruct (Apache-2.0). The 16 frames are passed as a
video at 4 fps (the clip spans 4 seconds). Temperature 0. One fixed prompt.
Writes <out>/pred_<split>.json with per-axis directions and flags.

usage: venv/bin/python vlm_baseline.py --splits test_scenes test_combos --out runs/vlm
"""

import argparse
import glob
import json
import os
import re
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

PROMPT = """These 16 frames are a 4-second video shot, in order. Describe only how the CAMERA moves, not how objects move.

Answer with one JSON object and nothing else, using exactly these keys and allowed values:
{"pan": "left" | "right" | "none",
 "tilt": "up" | "down" | "none",
 "roll": "left" | "right" | "none",
 "dolly": "in" | "out" | "none",
 "truck": "left" | "right" | "none",
 "pedestal": "up" | "down" | "none",
 "zoom": "in" | "out" | "none",
 "orbit": "left" | "right" | "none",
 "handheld": true | false}

Definitions, from the camera operator's point of view:
- pan: the camera turns left or right on the spot.
- tilt: the camera points up or down on the spot.
- roll: the camera rotates around its lens axis; "left" means the top of the frame swings to the left.
- dolly: the camera physically moves forward ("in") or backward ("out").
- truck: the camera physically moves sideways, left or right.
- pedestal: the camera physically moves straight up or down.
- zoom: the focal length changes ("in" makes things bigger without the camera moving), so there is no parallax.
- orbit: the camera travels around a subject while turning to keep it in frame; "right" means the camera travels to the right.
- handheld: small, irregular shake.
Several can be true at once. Use "none" when that motion is absent or too small to notice."""

MAP = {
    "pan": {"left": 2, "right": 0}, "tilt": {"up": 2, "down": 0}, "roll": {"left": 2, "right": 0},
    "dolly": {"in": 2, "out": 0}, "truck": {"right": 2, "left": 0}, "pedestal": {"up": 2, "down": 0},
    "zoom": {"in": 2, "out": 0}, "orbit": {"right": 2, "left": 0},
}


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        j = json.loads(m.group(0))
    except Exception:
        return None
    out = {}
    for k, mp in MAP.items():
        v = str(j.get(k, "none")).strip().lower()
        out[f"{k}_dir" if k != "orbit" else "orbit"] = mp.get(v, 1)
    hv = j.get("handheld", False)
    out["handheld"] = int(hv is True or str(hv).strip().lower() == "true")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/ephemeral/study/data")
    ap.add_argument("--splits", nargs="+", default=["test_scenes", "test_combos"])
    ap.add_argument("--out", default="runs/vlm")
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    proc = AutoProcessor.from_pretrained(MODEL)
    proc.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    for split in a.splits:
        outp = os.path.join(a.out, f"pred_{split}.json")
        preds = json.load(open(outp)) if os.path.exists(outp) else {}
        clips = sorted(d for d in glob.glob(os.path.join(a.data, split, "*")) if os.path.exists(os.path.join(d, "meta.json")))
        clips = [c for c in clips if os.path.basename(c) not in preds]
        t0 = time.time()
        for i in range(0, len(clips), a.bs):
            batch = clips[i:i + a.bs]
            msgs, vids = [], []
            for c in batch:
                frames = [Image.open(f).convert("RGB") for f in sorted(glob.glob(os.path.join(c, "f*.png")))]
                vids.append(frames)
                msgs.append([{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": PROMPT}]}])
            texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            meta = [{"fps": 4.0, "total_num_frames": 16, "duration": 4.0, "frames_indices": list(range(16))} for _ in vids]
            inputs = proc(text=texts, videos=vids, video_metadata=meta, return_tensors="pt", padding=True,
                          do_sample_frames=False).to("cuda")
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=160, do_sample=False)
            gen = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            for c, g in zip(batch, gen):
                p = parse(g)
                preds[os.path.basename(c)] = {"raw": g, "names": p, "parse_ok": p is not None}
            if (i // a.bs) % 10 == 0:
                json.dump(preds, open(outp, "w"))
                print(f"PROGRESS {split} {i + len(batch)}/{len(clips)} {time.time() - t0:.0f}s", flush=True)
        json.dump(preds, open(outp, "w"))
        print(f"DONE {split} {len(preds)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
