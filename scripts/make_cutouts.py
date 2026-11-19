"""Remove the background of chosen photos (rembg / U^2-Net) and save trimmed transparent PNGs.

Usage: python -I scripts/make_cutouts.py <input.jpg> <output.png> [<input> <output> ...]
"""
import sys

from PIL import Image
from rembg import new_session, remove

session = new_session("isnet-general-use")
args = sys.argv[1:]
for src, dst in zip(args[::2], args[1::2]):
    img = Image.open(src).convert("RGB")
    cut = remove(img, session=session, alpha_matting=True)
    cut = cut.crop(cut.getbbox())
    cut.thumbnail((900, 900))
    cut.save(dst, optimize=True)
    # preview on grey to check edges
    bg = Image.new("RGBA", cut.size, (200, 200, 205, 255))
    bg.alpha_composite(cut)
    bg.convert("RGB").save(dst.replace(".png", "_preview.jpg"))
    print(dst, cut.size)
