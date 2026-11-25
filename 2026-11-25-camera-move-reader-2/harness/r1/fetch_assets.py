"""Download CC0 assets from Poly Haven and split them into train and test pools.

Poly Haven assets are CC0. We keep the asset ids, URLs and the split in
assets/manifest.json so the dataset can be rebuilt.

Split rule (fixed before rendering): an asset goes to the test pool when
int(sha1(id), 16) % 100 < 15, so 15% of each kind is never seen in training.
"""

import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

API = "https://api.polyhaven.com"
UA = {"User-Agent": "camera-move-study/0.1 (research dataset build)"}
OUT = sys.argv[1] if len(sys.argv) > 1 else "/ephemeral/study/assets"
N = {"models": 170, "hdris": 90, "textures": 60}
SEED = 20261124


def is_test(asset_id):
    return int(hashlib.sha1(asset_id.encode()).hexdigest(), 16) % 100 < 15


def get(url, **kw):
    for i in range(5):
        try:
            r = requests.get(url, headers=UA, timeout=60, **kw)
            r.raise_for_status()
            return r
        except Exception:
            if i == 4:
                raise
            time.sleep(2 * (i + 1))


def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    r = get(url, stream=True)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    os.replace(tmp, path)


def fetch_model(aid):
    files = get(f"{API}/files/{aid}").json()
    g = files["gltf"]["1k"]["gltf"]
    base = os.path.join(OUT, "models", aid)
    download(g["url"], os.path.join(base, os.path.basename(g["url"])))
    for rel, inc in g.get("include", {}).items():
        download(inc["url"], os.path.join(base, rel))
    return {"id": aid, "file": os.path.join("models", aid, os.path.basename(g["url"])), "url": g["url"]}


def fetch_hdri(aid):
    files = get(f"{API}/files/{aid}").json()
    h = files["hdri"]["1k"]["hdr"]
    p = os.path.join(OUT, "hdris", f"{aid}.hdr")
    download(h["url"], p)
    return {"id": aid, "file": os.path.relpath(p, OUT), "url": h["url"]}


def fetch_texture(aid):
    files = get(f"{API}/files/{aid}").json()
    out = {"id": aid, "maps": {}}
    for key, want in (("Diffuse", "diffuse"), ("nor_gl", "normal"), ("Rough", "rough")):
        if key in files and "1k" in files[key]:
            fmt = "jpg" if "jpg" in files[key]["1k"] else next(iter(files[key]["1k"]))
            u = files[key]["1k"][fmt]["url"]
            p = os.path.join(OUT, "textures", aid, f"{want}.{fmt}")
            download(u, p)
            out["maps"][want] = os.path.relpath(p, OUT)
    if "diffuse" not in out["maps"]:
        return None
    return out


def main():
    rnd = random.Random(SEED)
    manifest = {"source": "https://polyhaven.com (CC0)", "seed": SEED, "split_rule": "sha1(id) % 100 < 15 -> test"}
    for kind, fn, t in (("models", fetch_model, "models"), ("hdris", fetch_hdri, "hdris"), ("textures", fetch_texture, "textures")):
        listing = get(f"{API}/assets", params={"t": t}).json()
        ids = sorted(listing)
        if kind == "textures":
            # ground-like surfaces only
            ids = [i for i in ids if set(listing[i].get("categories", [])) & {"floor", "terrain", "ground", "natural", "outdoor", "brick", "concrete", "rock", "sand"}]
        rnd.shuffle(ids)
        pick = sorted(ids[: N[kind]])
        with ThreadPoolExecutor(8) as ex:
            got = list(ex.map(lambda a: _safe(fn, a), pick))
        items = [g for g in got if g]
        for it in items:
            it["split"] = "test" if is_test(it["id"]) else "train"
        manifest[kind] = items
        print(kind, len(items), "test:", sum(i["split"] == "test" for i in items), flush=True)
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)


def _safe(fn, a):
    try:
        return fn(a)
    except Exception as e:
        print("skip", a, repr(e)[:120], flush=True)
        return None


if __name__ == "__main__":
    main()
