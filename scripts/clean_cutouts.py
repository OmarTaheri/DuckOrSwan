import sys
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage
for src, dst, flip in [("data/raw/cutouts/duck13.png","app/static/img/duck.png",False),("data/raw/cutouts/swan28.png","app/static/img/swan.png",True)]:
    im = Image.open(src).convert("RGBA"); a = np.array(im)
    lab, n = ndimage.label(a[...,3] > 20)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n+1))
    keep = lab == (np.argmax(sizes) + 1)
    keep = ndimage.binary_dilation(keep, iterations=2)
    a[...,3] = np.where(keep, a[...,3], 0)
    im = Image.fromarray(a); im = im.crop(im.getbbox())
    if flip: im = ImageOps.mirror(im)
    im.save(dst, optimize=True); print(dst, im.size, n, "components")
