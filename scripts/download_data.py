"""Build the Duck vs Swan dataset from two public sources.

Usage (run from the project root):
    python -I scripts/download_data.py openimages   # Open Images V7 (bbox crops)
    python -I scripts/download_data.py inat         # iNaturalist research-grade photos
    python -I scripts/download_data.py build        # dedupe, balance, split 70/15/15

Sources
  * Open Images V7 (Google, CC-BY 4.0 annotations): classes "Duck" and "Swan",
    cropped to the largest human-verified bounding box (+10% padding).
  * iNaturalist (research-grade = species ID verified by the community),
    photos licensed CC0 / CC-BY / CC-BY-NC only.
"""
import json
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
CROPS = RAW / "crops"
OUT = ROOT / "data" / "processed"
CLASSES = ["duck", "swan"]
TARGET_PER_CLASS = 5000
PINNED_TEST = {"inat_765276220.jpg", "inat_763186588.jpg", "inat_760381227.jpg", "inat_761492642.jpg"}
SEED = 42
INCLUDE_PREFIX = "inat_"  # see build(): Open Images crops were excluded after spot check
UA = {"User-Agent": "duck-swan-classifier-student-project/1.0"}


# --------------------------------------------------------------------------- #
# Source 1: Open Images V7
# --------------------------------------------------------------------------- #
def openimages(max_samples=2400):
    import fiftyone.zoo as foz
    from PIL import Image

    oi_names = {"duck": "Duck", "swan": "Swan"}
    for cls, oi_cls in oi_names.items():
        out_dir = CROPS / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for split in ["train", "validation", "test"]:
            ds = foz.load_zoo_dataset(
                "open-images-v7",
                split=split,
                label_types=["detections"],
                classes=[oi_cls],
                max_samples=max_samples,
                dataset_name=f"oi-{cls}-{split}",
                only_matching=False,
            )
            n = 0
            for s in ds:
                dets = s.ground_truth.detections if s.ground_truth else []
                labels = {d.label for d in dets}
                if {"Duck", "Swan"} <= labels:  # ambiguous image, skip
                    continue
                boxes = [d for d in dets if d.label == oi_cls]
                if not boxes:
                    continue
                d = max(boxes, key=lambda b: b.bounding_box[2] * b.bounding_box[3])
                try:
                    img = Image.open(s.filepath).convert("RGB")
                except Exception:
                    continue
                W, H = img.size
                x, y, w, h = d.bounding_box
                pad_w, pad_h = 0.1 * w, 0.1 * h
                box = (
                    max(0, int((x - pad_w) * W)),
                    max(0, int((y - pad_h) * H)),
                    min(W, int((x + w + pad_w) * W)),
                    min(H, int((y + h + pad_h) * H)),
                )
                crop = img.crop(box)
                if min(crop.size) < 64:
                    continue
                crop.thumbnail((512, 512))
                crop.save(out_dir / f"oi_{Path(s.filepath).stem}.jpg", quality=92)
                n += 1
            print(f"[openimages] {cls}/{split}: {n} crops")
            ds.delete()


# --------------------------------------------------------------------------- #
# Source 2: iNaturalist
# --------------------------------------------------------------------------- #
INAT_TAXA = {
    # 9 duck genera: dabbling, diving, perching, Muscovy/domestic, goldeneyes, mergansers
    "duck": ["Anas", "Aythya", "Aix", "Mareca", "Spatula", "Cairina", "Bucephala", "Mergus", "Netta"],
    # 6 swan species with an equal quota, so black swans (C. atratus) are well represented
    "swan": ["Cygnus olor", "Cygnus cygnus", "Cygnus columbianus", "Cygnus buccinator",
             "Cygnus atratus", "Cygnus melancoryphus"],
}


def _taxon_id(requests, name):
    rank = "species" if " " in name else "genus"
    r = requests.get(
        "https://api.inaturalist.org/v1/taxa",
        params={"q": name, "rank": rank, "per_page": 5},
        headers=UA,
        timeout=30,
    )
    r.raise_for_status()
    for t in r.json()["results"]:
        if t["name"] == name:
            return t["id"]
    raise RuntimeError(f"taxon not found: {name}")


def inat(per_class=5600):
    import requests as _requests
    from PIL import Image
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    # retry transient SSL / 5xx / rate-limit errors with back-off
    requests = _requests.Session()
    requests.mount("https://", HTTPAdapter(
        max_retries=Retry(total=8, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504]),
        pool_maxsize=32))

    def fetch(job):
        url, path = job
        if path.exists():
            return True
        try:
            r = requests.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            if min(img.size) < 64:
                return False
            img.save(path, quality=92)
            return True
        except Exception:
            return False

    manifest_path = RAW / "inat_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for cls, genera in INAT_TAXA.items():
        out_dir = CROPS / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        ids = [_taxon_id(requests, g) for g in genera]
        quota = per_class // len(genera) + 1
        jobs = []
        for g, tid in zip(genera, ids):
            n_before = len(jobs)
            got, id_below = 0, None
            while got < quota:
                params = {
                    "taxon_id": tid,
                    "quality_grade": "research",
                    "photos": "true",
                    "photo_license": "cc0,cc-by,cc-by-nc",
                    "per_page": 200,
                    "order_by": "id",
                    "order": "desc",
                }
                if id_below:
                    params["id_below"] = id_below
                r = requests.get(
                    "https://api.inaturalist.org/v1/observations",
                    params=params, headers=UA, timeout=60,
                )
                r.raise_for_status()
                res = r.json()["results"]
                if not res:
                    break
                for obs in res:
                    id_below = obs["id"]
                    p = obs["photos"][0]  # first photo = main photo of the bird
                    url = p["url"].replace("square", "medium")
                    jobs.append((url, out_dir / f"inat_{p['id']}.jpg"))
                    got += 1
                    if got >= quota:
                        break
                time.sleep(1.0)  # respect iNat API rate limit
            for _, path in jobs[n_before:]:
                manifest[path.name] = g
            print(f"[inat] {cls}/{g}: {got} observations queued")
        with ThreadPoolExecutor(32) as ex:  # S3 latency ~5 s/photo, so go wide
            ok = sum(ex.map(fetch, jobs))
        print(f"[inat] {cls}: {ok}/{len(jobs)} photos downloaded")
        manifest_path.write_text(json.dumps(manifest))


# --------------------------------------------------------------------------- #
# Build: dedupe, balance, split, stats, contact sheets
# --------------------------------------------------------------------------- #
def build():
    import imagehash
    from PIL import Image

    import numpy as np

    rng = random.Random(SEED)
    hashes = []  # 64-bit perceptual hashes of kept images (vectorised Hamming check)
    kept = {c: [] for c in CLASSES}
    dropped = {"corrupt_or_small": 0, "duplicate": 0, "in_both_classes": 0}
    # a photo attached to both a duck and a swan observation has an ambiguous label
    names = {c: {p.name for p in (CROPS / c).glob(f"{INCLUDE_PREFIX}*.jpg")} for c in CLASSES}
    ambiguous = names["duck"] & names["swan"]
    dropped["in_both_classes"] = len(ambiguous)
    for cls in CLASSES:
        # Only research-grade iNaturalist photos are used. A visual spot check of the Open
        # Images "Duck" class showed label noise (rubber ducks, Canada geese, grebes,
        # statues), and mixing two sources unevenly per class risks a source-style shortcut.
        for p in sorted((CROPS / cls).glob(f"{INCLUDE_PREFIX}*.jpg")):
            if p.name in ambiguous:
                continue
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                dropped["corrupt_or_small"] += 1
                continue
            if min(img.size) < 64:
                dropped["corrupt_or_small"] += 1
                continue
            h = np.uint64(int(str(imagehash.phash(img)), 16))
            if hashes:
                x = np.bitwise_xor(np.array(hashes, dtype=np.uint64), h)
                dist = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
                if dist.min() <= 4:
                    dropped["duplicate"] += 1
                    continue
            hashes.append(h)
            kept[cls].append(p)

    n = min(TARGET_PER_CLASS, *(len(v) for v in kept.values()))
    if OUT.exists():
        shutil.rmtree(OUT)
    manifest_path = RAW / "inat_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    stats = {"per_class": n, "dropped": dropped, "splits": {}, "sources": {}, "taxa": {}}
    for cls in CLASSES:
        files = kept[cls]
        rng.shuffle(files)
        # the 4 images shown in the report's website screenshots always stay in the test set
        pinned = [f for f in files if f.name in PINNED_TEST]
        others = [f for f in files if f.name not in PINNED_TEST]
        files = others[:n - len(pinned)] + pinned
        stats["sources"][cls] = {
            "open_images": sum(f.name.startswith("oi_") for f in files),
            "inaturalist": sum(f.name.startswith("inat_") for f in files),
        }
        taxa = {}
        for f in files:
            # first download (genus-level query) did not record species
            t = manifest.get(f.name, "Cygnus spp." if cls == "swan" else "Anatidae spp.")
            taxa[t] = taxa.get(t, 0) + 1
        stats["taxa"][cls] = dict(sorted(taxa.items(), key=lambda kv: -kv[1]))
        n_tr, n_va = int(0.70 * n), int(0.15 * n)
        parts = {
            "train": files[:n_tr],
            "val": files[n_tr:n_tr + n_va],
            "test": files[n_tr + n_va:],
        }
        for split, fs in parts.items():
            d = OUT / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for f in fs:
                shutil.copy2(f, d / f.name)
            stats["splits"].setdefault(split, {})[cls] = len(fs)
        contact_sheet(files[:36], ROOT / "data" / f"contact_{cls}.jpg")
    (ROOT / "data" / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def contact_sheet(files, out, cell=128, cols=6):
    from PIL import Image

    rows = (len(files) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), "white")
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB")
        im.thumbnail((cell, cell))
        sheet.paste(im, ((i % cols) * cell, (i // cols) * cell))
    sheet.save(out, quality=90)


if __name__ == "__main__":
    {"openimages": openimages, "inat": inat, "build": build}[sys.argv[1]]()
