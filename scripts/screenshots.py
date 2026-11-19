"""Screenshot the running website for the report, with the URL shown.

Playwright renders the page off-screen (nothing else on the desktop can leak into the
image). A Chrome-style address bar showing the page's real URL (page.url) is drawn
above each screenshot, as the assignment asks for the URL to be visible.

Usage: python -I scripts/screenshots.py http://localhost:8000 img1 img2 img3 img4
"""
import json
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "report" / "figures"
URL = sys.argv[1]
IMAGES = [Path(p) for p in sys.argv[2:]]
W, H, SCALE = 1440, 900, 1.5


def font(size, bold=False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def icon_font(size):
    try:
        return ImageFont.truetype("seguisym.ttf", size)
    except OSError:
        return ImageFont.load_default()


def with_browser_frame(png: bytes, url: str, out: Path):
    page = Image.open(BytesIO(png)).convert("RGB")
    s = page.width / W
    tab_h, bar_h = int(40 * s), int(48 * s)
    img = Image.new("RGB", (page.width, page.height + tab_h + bar_h), "#dee1e6")
    d = ImageDraw.Draw(img)
    # tab strip with one active tab
    d.rounded_rectangle([int(8 * s), int(8 * s), int(260 * s), tab_h + 10], radius=int(8 * s), fill="white")
    d.text((int(22 * s), int(14 * s)), "Duck vs Swan Classification", fill="#202124", font=font(int(13 * s)))
    # toolbar + omnibox
    d.rectangle([0, tab_h, page.width, tab_h + bar_h], fill="white")
    for i, ch in enumerate(["←", "→", "↻"]):
        d.text((int((16 + 34 * i) * s), tab_h + int(10 * s)), ch, fill="#5f6368", font=icon_font(int(20 * s)))
    x0 = int(120 * s)
    d.rounded_rectangle([x0, tab_h + int(7 * s), page.width - int(60 * s), tab_h + bar_h - int(7 * s)],
                        radius=int(17 * s), fill="#f1f3f4")
    d.text((x0 + int(18 * s), tab_h + int(13 * s)), "ⓘ", fill="#5f6368", font=icon_font(int(16 * s)))
    d.text((x0 + int(44 * s), tab_h + int(12 * s)), url, fill="#202124", font=font(int(16 * s)))
    d.line([0, tab_h + bar_h - 1, page.width, tab_h + bar_h - 1], fill="#c4c7c5", width=max(1, int(s)))
    img.paste(page, (0, tab_h + bar_h))
    img.save(out, optimize=True)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome")
    page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=SCALE)
    page.goto(URL)
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(800)
    with_browser_frame(page.screenshot(), page.url, FIG / "web_idle.png")

    results = []
    for i, img in enumerate(IMAGES, 1):
        if page.is_visible("#again"):
            page.click("#again")
        page.set_input_files("#file", str(img))
        page.wait_for_selector(".side.win", timeout=30000)
        page.wait_for_timeout(1300)  # let the green transition finish
        out = f"web_test{i}.png"
        with_browser_frame(page.screenshot(), page.url, FIG / out)
        label = page.inner_text(".side.win .side-label").capitalize()
        badge = page.inner_text(".side.win .badge")
        conf = round(float(badge.replace("✓", "").replace("%", "").replace(">", "").strip()) / 100, 4)
        conf_text = badge.replace("✓", "").strip()
        results.append({"file": out, "image": img.name, "true": img.parent.name.capitalize(),
                        "label": label, "confidence": conf, "confidence_text": conf_text})
        print(results[-1])
    (FIG / "web_tests.json").write_text(json.dumps(results, indent=2))
    browser.close()
