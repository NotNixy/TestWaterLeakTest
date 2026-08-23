"""Drive the running Streamlit app headlessly and screenshot every tab in both
colour modes. Fails loudly if Streamlit rendered a Python exception anywhere.

Dark mode is exercised by emulating the OS setting (`prefers-color-scheme`), not
by clicking the override — that is the path real users take, so it is the one
worth testing.
"""
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent / "shots"
OUT.mkdir(exist_ok=True)
URL = "http://localhost:8501"

TABS = ["Overview", "Rate vs Volume", "Priority Schedule", "Burst Risk",
        "Loss Composition", "Plant Profile"]

ERROR_PATTERNS = ["Traceback (most recent call last)", "KeyError:", "ValueError:",
                  "TypeError:", "AttributeError:", "NameError:", "IndexError:"]

problems = []


def capture(pg, scheme, only=None):
    pg.goto(URL, wait_until="networkidle", timeout=120_000)
    pg.wait_for_selector('[data-testid="stTabs"]', timeout=120_000)
    time.sleep(10)

    for i, name in enumerate(TABS):
        if only is not None and i not in only:
            continue
        tabs = pg.query_selector_all('[role="tab"]')
        if i >= len(tabs):
            problems.append(f"[{scheme}] tab '{name}' missing "
                            f"(found {len(tabs)} tabs)")
            continue
        tabs[i].click()
        time.sleep(6)
        pg.mouse.wheel(0, 250)
        time.sleep(1)
        pg.mouse.wheel(0, -250)
        time.sleep(2)

        for ex in pg.query_selector_all('[data-testid="stException"], .stException'):
            problems.append(f"[{scheme}/{name}] EXCEPTION: {ex.inner_text()[:400]}")
        body = pg.inner_text("body")
        for pat in ERROR_PATTERNS:
            if pat in body:
                problems.append(f"[{scheme}/{name}] page contains {pat}")

        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        suffix = "" if scheme == "light" else "_dark"
        pg.screenshot(path=str(OUT / f"{i+1:02d}_{slug}{suffix}.png"),
                      full_page=True)
        print(f"  captured {i+1:02d}_{slug}{suffix}.png")


with sync_playwright() as pw:
    b = pw.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])

    print("LIGHT (prefers-color-scheme: light)")
    pg = b.new_page(viewport={"width": 1680, "height": 1150},
                    device_scale_factor=2, color_scheme="light")
    capture(pg, "light")
    pg.close()

    # Dark: every tab, to catch any hard-coded light colour anywhere.
    print("DARK (prefers-color-scheme: dark)")
    pg = b.new_page(viewport={"width": 1680, "height": 1150},
                    device_scale_factor=2, color_scheme="dark")
    capture(pg, "dark")
    pg.close()

    b.close()

if problems:
    print("\n--- PROBLEMS ---")
    for p in problems:
        print(" *", p)
    sys.exit(1)
print("\nAll tabs rendered without exceptions in both colour modes.")
