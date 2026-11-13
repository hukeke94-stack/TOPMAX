#!/usr/bin/env python3
"""Screenshots of the live TOPMAX platform for Fig. 6 (built-in maize data): prediction setup and results, selection setup and results.

The interface text is enlarged before capture so that it stays legible at journal column width. Each panel is cropped
from the content pane between two interface elements, so the crops follow the layout instead of fixed pixel positions.
"""
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "figures/fig6_shots"
OUT.mkdir(parents=True, exist_ok=True)
URL = "https://topxbio.com/#demo"
SCALE = 3
CSS = ("html{font-size:22px !important}"
       ".text-\\[10px\\]{font-size:0.68rem !important;letter-spacing:0.03em !important}"
       ".text-\\[13px\\]{font-size:0.82rem !important}"
       ".text-xs{font-size:0.82rem !important;line-height:1.35 !important}"
       "svg text{font-size:15px !important}")


def box(loc):
    b = loc.bounding_box()
    return b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"]


def shot(page, name, top, bottom, pad=14):
    """Crop the content pane from the top of `top` to the bottom of `bottom` (boxes are measured in the same viewport frame)."""
    from io import BytesIO
    from PIL import Image
    pane = page.locator("div[class*='min-h-[520px]']").first
    pane.scroll_into_view_if_needed()
    png = pane.screenshot()
    px0, py0, _, _ = box(pane)
    y0 = box(top)[1] - pad - py0
    y1 = box(bottom)[3] + pad - py0
    im = Image.open(BytesIO(png))
    im.crop((0, max(0, round(y0 * SCALE)), im.width, min(im.height, round(y1 * SCALE)))).save(OUT / name)


with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    page = b.new_page(viewport={"width": 1024, "height": 1400}, device_scale_factor=SCALE)
    page.goto(URL, wait_until="load", timeout=90000)
    page.wait_for_timeout(4000)
    page.add_style_tag(content=CSS)
    page.locator("#demo").scroll_into_view_if_needed()

    # a: prediction setup
    page.get_by_role("button", name="Genomic Prediction", exact=True).click()
    page.wait_for_timeout(800)
    page.get_by_label("Select trait").select_option("CW")
    page.wait_for_timeout(300)
    shot(page, "a_prediction_setup.png", page.get_by_role("button", name="Built-in maize NCII data").first,
         page.get_by_role("button", name="Submit job"))

    # b: prediction results; the small fold-metrics chart is left out and the scatter plot takes the full width
    page.get_by_role("button", name="Submit job").click()
    page.get_by_text("Succeeded", exact=True).first.wait_for(timeout=600000)
    page.wait_for_timeout(2500)
    page.evaluate("""() => {
        const h = [...document.querySelectorAll('h4,p,span,div')].find(e => e.textContent.trim() === 'Fold Metrics' && e.children.length === 0);
        if (!h) return;
        let card = h; while (card && !(card.parentElement && getComputedStyle(card.parentElement).display === 'grid')) card = card.parentElement;
        if (card) { card.style.display = 'none'; card.parentElement.style.gridTemplateColumns = '1fr'; }
    }""")
    page.wait_for_timeout(1500)
    top_b = page.get_by_text("Training CV PCC", exact=False).first.locator("xpath=ancestor::div[contains(@class,'rounded')][1]")
    scatter = page.get_by_text("Observed vs Predicted", exact=False).first.locator("xpath=ancestor::div[contains(@class,'rounded')][1]")
    shot(page, "b_prediction_results.png", top_b, scatter)

    # c: selection setup
    page.get_by_role("button", name="Multi-trait Selection", exact=True).click()
    page.wait_for_timeout(1500)
    shot(page, "c_selection_setup.png", page.get_by_role("button", name="Built-in maize NCII data").first,
         page.get_by_role("button", name="Run selection"))

    # d: selection results, first ten candidates without the scroll box (the similarity column does not fit at this text size)
    page.get_by_role("button", name="Run selection").click()
    page.get_by_text("Candidates ranked").first.wait_for(timeout=600000)
    page.wait_for_timeout(2500)
    page.add_style_tag(content=".max-h-80{max-height:none !important;overflow:visible !important}"
                               ".max-h-80 tbody tr:nth-child(n+11){display:none}"
                               ".max-h-80 th:nth-child(3),.max-h-80 td:nth-child(3){display:none}"
                               # keep the candidate names inside the white table box
                               ".max-h-80 table{width:100% !important;min-width:0 !important}"
                               ".max-h-80 th,.max-h-80 td{padding-left:0.55rem !important;padding-right:0.3rem !important}"
                               ".max-h-80 td:nth-child(2){font-size:0.72rem !important;white-space:nowrap}")
    page.wait_for_timeout(800)
    weights = page.get_by_text("Learned trait weights").first.locator("xpath=ancestor::div[contains(@class,'rounded')][1]")
    top_d = page.get_by_text("Weight learning", exact=True).first.locator("xpath=ancestor::div[contains(@class,'rounded')][1]")
    shot(page, "d_selection_results.png", top_d, weights)
    b.close()
print("saved", sorted(x.name for x in OUT.iterdir()))
