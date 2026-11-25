"""Re-download run one's exact Poly Haven assets from its manifest.json (CC0).

Run one's fetch_assets.py sampled ids at random; re-running it could pick
different ids if Poly Haven's catalogue changed. This fetches the listed ids
only, keeps run one's manifest as the manifest, and checks every listed file.

usage: venv/bin/python fetch_from_manifest.py /ephemeral/study/assets   (manifest.json already copied there)
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

OUT = sys.argv[1]
sys.argv = [sys.argv[0], OUT]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fetch_assets as fa  # noqa: E402

m = json.load(open(os.path.join(OUT, "manifest.json")))
jobs = [(fa.fetch_model, x["id"]) for x in m["models"]] + [(fa.fetch_hdri, x["id"]) for x in m["hdris"]] + \
       [(fa.fetch_texture, x["id"]) for x in m["textures"]]
with ThreadPoolExecutor(16) as ex:
    res = list(ex.map(lambda j: (j[1], j[0](j[1])), jobs))
missing = []
for x in m["models"] + m["hdris"]:
    if not os.path.exists(os.path.join(OUT, x["file"])):
        missing.append(x["file"])
for x in m["textures"]:
    missing += [p for p in x["maps"].values() if not os.path.exists(os.path.join(OUT, p))]
print("fetched", len(res), "missing", len(missing), missing[:10])
print("ASSETS_OK" if not missing else "ASSETS_MISSING")
