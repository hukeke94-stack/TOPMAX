#!/usr/bin/env python3
"""Fig. 6 from the live-site screenshots (fig6_screenshots.py): prediction in the left column (a, b), selection in the right (c, d).
170 mm wide at 600 dpi."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SHOTS = HERE.parent / "figures/fig6_shots"
OUT = HERE.parent / "figures/final/Fig6_platform.png"
DPI = 600
W = round(170 / 25.4 * DPI)
GUT, LAB, PAD = 90, 110, 24          # gutter between panels, room for panel letters, inner padding
BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
INK = (46, 60, 77)
RULE = (221, 226, 232)

# Each column, top to bottom: (letter, file, top crop in screenshot pixels)
COLUMNS = [[("a", "a_prediction_setup.png", 0), ("b", "b_prediction_results.png", 0)],
           [("c", "c_selection_setup.png", 0), ("d", "d_selection_results.png", 0)]]

col_w = (W - GUT) // 2
inner = col_w - 2 * PAD
cols = []
for col in COLUMNS:
    items = []
    for tag, name, top in col:
        im = Image.open(SHOTS / name).convert("RGB")
        im = im.crop((0, top, im.width, im.height))
        items.append((tag, im.resize((inner, round(im.height * inner / im.width)), Image.LANCZOS)))
    cols.append(items)
heights = [sum(LAB + im.height + 2 * PAD for _, im in c) + GUT * (len(c) - 1) for c in cols]
H = max(heights)
fig = Image.new("RGB", (W, H), "white")
dr = ImageDraw.Draw(fig)
font = ImageFont.truetype(BOLD, 80)
for k, c in enumerate(cols):
    x, y = k * (col_w + GUT), 0
    extra = (H - heights[k]) // len(c)          # spread any height difference over the column's panels
    for tag, im in c:
        h = im.height + 2 * PAD + extra
        dr.text((x, y + LAB - 22), tag, font=font, fill=INK, anchor="ls")
        dr.rounded_rectangle([x, y + LAB, x + col_w - 1, y + LAB + h - 1], radius=36, outline=RULE, width=5)
        fig.paste(im, (x + PAD, y + LAB + PAD))
        y += LAB + h + GUT
fig.save(OUT, dpi=(DPI, DPI))
print("saved", OUT, fig.size, f"{fig.height / DPI * 25.4:.0f} mm tall", "column heights", heights)
