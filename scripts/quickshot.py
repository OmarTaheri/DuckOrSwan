import sys
from playwright.sync_api import sync_playwright
url, out = sys.argv[1], sys.argv[2]
img = sys.argv[3] if len(sys.argv) > 3 else None
w, h = (int(v) for v in (sys.argv[4] if len(sys.argv) > 4 else "1440x900").split("x"))
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    pg = b.new_page(viewport={"width": w, "height": h})
    pg.goto(url); pg.wait_for_load_state("networkidle")
    if img:
        pg.set_input_files("#file", img); pg.wait_for_selector(".side.win", timeout=30000); pg.wait_for_timeout(1200)
    pg.screenshot(path=out)
    b.close()
