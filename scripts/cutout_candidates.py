"""Fetch high-res CC0/CC-BY photos of a mallard and a mute swan (iNaturalist) for the web page cutouts."""
import glob
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import requests
from PIL import Image, ImageDraw

out = sys.argv[1]
UA = {"User-Agent": "duck-swan-classifier-student-project/1.0"}
jobs, meta = [], {}
for name, q in [("duck", "Anas platyrhynchos"), ("swan", "Cygnus olor")]:
    r = requests.get("https://api.inaturalist.org/v1/observations", headers=UA, timeout=60, params={
        "taxon_name": q, "quality_grade": "research", "photos": "true",
        "photo_license": "cc0,cc-by", "per_page": 40, "order_by": "votes"}).json()
    for i, o in enumerate(r["results"][:40]):
        p = o["photos"][0]
        fn = f"{out}/{name}_{i:02d}.jpg"
        jobs.append((p["url"].replace("square", "large"), fn))
        meta[fn] = {"license": p["license_code"], "attribution": p["attribution"], "obs": o["uri"]}


def fetch(j):
    try:
        Image.open(BytesIO(requests.get(j[0], headers=UA, timeout=60).content)).convert("RGB").save(j[1])
        return 1
    except Exception:
        return 0


with ThreadPoolExecutor(32) as ex:
    print(sum(ex.map(fetch, jobs)), "downloaded")
json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)

for name in ["duck", "swan"]:
    fs = sorted(glob.glob(f"{out}/{name}_*.jpg"))
    c, cols = 200, 8
    sheet = Image.new("RGB", (cols * c, ((len(fs) + cols - 1) // cols) * c), "white")
    d = ImageDraw.Draw(sheet)
    for i, fp in enumerate(fs):
        im = Image.open(fp)
        im.thumbnail((c, c))
        sheet.paste(im, ((i % cols) * c, (i // cols) * c))
        d.rectangle([(i % cols) * c, (i // cols) * c, (i % cols) * c + 22, (i // cols) * c + 14], fill="black")
        d.text(((i % cols) * c + 3, (i // cols) * c + 2), fp[-6:-4], fill="yellow")
    sheet.save(f"{out}/sheet_{name}.jpg")
