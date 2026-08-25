"""Acceptance test for the no-scroll requirement.

Walks every main tab and every sub-tab, in both colour modes, at the target
screen size and reports whether the page overflows. Fails if any view scrolls.

Height is measured as the layout container's own box — not scrollHeight, which
clamps at the viewport and hides how much slack is left, and not the union of
every descendant, which over-counts rows inside a dataframe or a collapsed
expander. Those scroll internally and are not page scrolling.
"""

import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent / "shots"
OUT.mkdir(exist_ok=True)
URL = "http://localhost:8501"

# 1920x1080 screen, less browser chrome (tab strip, URL bar, bookmarks bar).
VIEW_W, VIEW_H = 1920, 980

MAIN = ["Overview", "Rate vs Volume", "Priority Schedule", "Burst Risk",
        "Loss Composition", "Plant Profile"]
ERRORS = ["Traceback (most recent call last)", "KeyError:", "ValueError:",
          "TypeError:", "AttributeError:", "NameError:", "IndentationError:"]

problems, rows = [], []


def height(pg):
    return pg.evaluate("""() => {
        const c = document.querySelector('.stMain .block-container')
               || document.querySelector('.block-container');
        const r = c.getBoundingClientRect();
        return Math.round(r.bottom + window.scrollY);
    }""")


with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
    for scheme in ["light", "dark"]:
        pg = b.new_page(viewport={"width": VIEW_W, "height": VIEW_H},
                        device_scale_factor=1, color_scheme=scheme)
        pg.goto(URL, wait_until="networkidle", timeout=120_000)
        pg.wait_for_selector('[data-testid="stTabs"]', timeout=120_000)
        time.sleep(10)

        def visible(names_in):
            """Streamlit keeps inactive tab panels in the DOM but hidden, so
            tabs must be filtered to the ones actually on screen."""
            return [t for t in pg.query_selector_all('[role="tab"]')
                    if (t.inner_text().strip() in MAIN) == names_in
                    and t.is_visible()]

        for i, name in enumerate(MAIN):
            visible(True)[i].click()
            time.sleep(4)
            # Overview has no sub-tabs now, so a single "—" pass is expected.
            labels = [t.inner_text().strip() for t in visible(False)] or ["—"]
            for j, lab in enumerate(labels):
                subs = visible(False)
                if subs:
                    subs[j].click()
                    time.sleep(3)
                h = height(pg)
                fits = h <= VIEW_H
                rows.append((scheme, name, lab, h, fits))
                if not fits:
                    problems.append(f"[{scheme}] {name} / {lab}: "
                                    f"{h}px > {VIEW_H}px")
                body = pg.inner_text("body")
                for pat in ERRORS:
                    if pat in body:
                        problems.append(f"[{scheme}] {name}/{lab}: {pat}")
                if scheme == "light":
                    slug = re.sub(r"[^a-z0-9]+", "_",
                                  f"{i+1:02d} {name} {lab}".lower()).strip("_")
                    pg.screenshot(path=str(OUT / f"{slug}.png"))
        pg.close()
    b.close()

print(f"{'mode':7s}{'tab':20s}{'sub-tab':24s}{'height':>8s}{'slack':>7s}{'fits':>6s}")
for scheme, name, lab, h, fits in rows:
    print(f"{scheme:7s}{name:20s}{lab[:23]:24s}{h:>8d}{VIEW_H - h:>7d}"
          f"{'yes' if fits else 'NO':>6s}")

worst = max(rows, key=lambda r: r[3])
print(f"\n{len(rows)} views checked · tallest {worst[3]}px "
      f"({worst[1]} / {worst[2]}) · budget {VIEW_H}px")
if problems:
    print("\n--- PROBLEMS ---")
    for p in problems:
        print(" *", p)
    sys.exit(1)
print("\nEvery view fits without scrolling.")
