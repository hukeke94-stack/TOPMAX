#!/usr/bin/env python3
"""Build Fig. 1 from the untouched original: only the listed rectangles are repainted, everything else stays as in pic1_final.

Usage: fig1_compose.py <logo.png or -> <output.png>
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
SRC = next(p for p in (HERE.parent / "figures" / "fig1_gpt" / "input_pic1_final.png", HERE / "fig1_inputs" / "input_pic1_final.png") if p.is_file())
SCALE = 2          # work on a 2x canvas so new text and logo are not blurred twice
NARROW = "/System/Library/Fonts/Supplemental/Arial Narrow.ttf"
NARROW_B = "/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf"
INK = (22, 24, 30)
NAVY = (18, 32, 72)
RED = (226, 30, 38)

# Rectangles in original pixels: (x0, y0, x1, y1). Nothing outside them may change.
LOGO = (680, 40, 962, 170)
LEGEND = (756, 316, 845, 342)
HEAT_LABEL = (31, 760, 112, 781)
SENTENCE = (88, 828, 664, 864)
CARD_NUM = {"speed": (700, 698, 866, 764), "memory": (920, 698, 1088, 764), "selection": (1380, 698, 1596, 764)}
CARD_TXT = {"speed": (684, 772, 880, 826), "memory": (912, 772, 1096, 826), "selection": (1374, 772, 1602, 826)}
FORMULA = (388, 166, 534, 204)  # n < p equation
HEAT = (18, 638, 590, 786)      # heatmap labels, cells and the TOPMAX box, without the colour bar
HEAT_ROWS = {"GBLUP": (638, 663), "LASSO": (663, 687), "SVR": (687, 709), "LightGBM": (709, 733), "DNNGP": (733, 754), "TOPMAX": (754, 786)}
HEAT_LABEL_X = 103              # labels lie left of this, cells to the right
LEGEND_TOP = (718, 255, 852, 312)  # GBLUP legend entry moves up, HIBLUP entry goes below it
COST_PLOT = (704, 360, 912, 512)   # HIBLUP curve, its OOM cross and label (drawn over, nothing cleared)
TEAL = (0, 163, 180)               # HIBLUP colour of Figs 3 and 5
OOM_RED = (246, 0, 0)
REGIONS = [LOGO, LEGEND, LEGEND_TOP, HEAT_LABEL, SENTENCE, FORMULA, *CARD_NUM.values(), *CARD_TXT.values()]  # cleared, then redrawn
CHANGED = REGIONS + [HEAT, COST_PLOT]                                                                        # may differ from the original


def clear(img, box):
    """Repaint a rectangle with the colour of its own border, so old text leaves no trace."""
    b = [v * SCALE for v in box]
    a = np.asarray(img.crop(b)).reshape(b[3] - b[1], b[2] - b[0], 3)
    edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    img.paste(tuple(int(v) for v in np.median(edge, axis=0)), b)


def text(img, xy, runs, size, bold=False, anchor="l"):
    """Draw coloured text runs, supersampled and softened to sit with the original raster. xy = (x, baseline)."""
    ss = 4
    font = ImageFont.truetype(NARROW_B if bold else NARROW, int(size * SCALE * ss))
    total = sum(font.getlength(t) for t, _ in runs)
    x0 = xy[0] * SCALE * ss - (total / 2 if anchor == "c" else 0)
    y0 = xy[1] * SCALE * ss
    W, H = int(total + 80), int(font.size * 1.6)
    x, top = int(x0) - 40, int(y0 - font.size * 1.2)
    rgb = Image.new("RGB", (W, H), (255, 255, 255))
    mask = Image.new("L", (W, H), 0)
    dr, dm = ImageDraw.Draw(rgb), ImageDraw.Draw(mask)
    pen = x0 - x
    for t, col in runs:
        dr.rectangle([pen, 0, pen + font.getlength(t), H], fill=col)
        # Stroke matches the heavier weight of the original raster text
        dm.text((pen, y0 - top), t, font=font, fill=255, anchor="ls", stroke_width=0 if bold else int(0.32 * SCALE * ss), stroke_fill=255)
        pen += font.getlength(t)
    size2 = (W // ss, H // ss)
    mask = mask.resize(size2, Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.55))
    img.paste(rgb.resize(size2, Image.BOX), (x // ss, top // ss), mask)


def paste_logo(img, logo_path):
    logo = Image.open(logo_path).convert("RGB")
    bbox = ImageChops.difference(logo, Image.new("RGB", logo.size, (255, 255, 255))).point(lambda v: 255 if v > 24 else 0).getbbox()
    logo = logo.crop(bbox)
    x0, y0, x1, y1 = [v * SCALE for v in LOGO]
    pad = 6 * SCALE
    r = min((x1 - x0 - 2 * pad) / logo.width, (y1 - y0 - 2 * pad) / logo.height)
    logo = logo.resize((round(logo.width * r), round(logo.height * r)), Image.LANCZOS)
    px, py = x0 + (x1 - x0 - logo.width) // 2, y0 + (y1 - y0 - logo.height) // 2
    box = (px, py, px + logo.width, py + logo.height)
    # White logo background becomes the panel colour
    img.paste(ImageChops.multiply(img.crop(box), logo), box)


def fix_dual_formula(img, base):
    """(X^T X^T + aI)z = y -> (X X^T + aI)z = y, using the original glyphs: drop the first superscript T and close the gap."""
    x0, y0, x1, y1 = [v * SCALE for v in FORMULA]
    clear(img, FORMULA)
    bg = img.getpixel((x0 + 2, y0 + 2))
    right = base.crop((412 * SCALE, y0, 532 * SCALE, y1))
    right.paste(bg, (0, 0, (423 - 412) * SCALE, (183 - 166) * SCALE))   # the first superscript T
    left = base.crop((390 * SCALE, y0, 412 * SCALE, y1))
    for part, px in ((left, 390 + 5), (right, 412 - 4)):                # shift both parts so the line stays centred
        box = (px * SCALE, y0, px * SCALE + part.width, y1)
        img.paste(ImageChops.darker(img.crop(box), part), box)


def drop_svr_row(img):
    """Remove the SVR row (not in Fig. 2) and stretch the cells of the other five rows to fill the same height.
    Row labels keep their size; the TOPMAX row, with its box, is scaled whole and its label redrawn."""
    S = SCALE
    x0, y0, x1, y1 = [v * S for v in HEAT]
    src = img.crop((x0, y0, x1, y1))
    a = np.asarray(src).reshape(src.height, src.width, 3)
    bg = tuple(int(v) for v in np.median(np.concatenate([a[0], a[:, 0]]), axis=0))
    keep = ["GBLUP", "LASSO", "LightGBM", "DNNGP", "TOPMAX"]
    f = (y1 - y0) / sum((HEAT_ROWS[k][1] - HEAT_ROWS[k][0]) * S for k in keep)
    out = Image.new("RGB", src.size, bg)
    lx, y, top_max = HEAT_LABEL_X * S, 0, None
    for i, k in enumerate(keep):
        ra, rb = [v * S - y0 for v in HEAT_ROWS[k]]
        h = (y1 - y0) - y if i == len(keep) - 1 else round((rb - ra) * f)
        strip = src.crop((0, ra, src.width, rb))
        if k == "TOPMAX":
            out.paste(strip.resize((src.width, h), Image.LANCZOS), (0, y))
            top_max = y
        else:
            out.paste(strip.crop((lx, 0, src.width, rb - ra)).resize((src.width - lx, h), Image.LANCZOS), (lx, y))
            out.paste(strip.crop((0, 0, lx, rb - ra)), (0, y + (h - (rb - ra)) // 2))
        y += h
    img.paste(out, (x0, y0))
    # TOPMAX label inside the stretched box
    ty = y0 + top_max
    inner = (HEAT_LABEL[0] * S, ty + round(6 * S * f), HEAT_LABEL[2] * S, ty + round((786 - 754 - 6) * S * f))
    img.paste(img.getpixel((inner[0] + 2, inner[1] + 2)), inner)
    text(img, (33, (ty / S) + (777 - 754) * f), [("TOPMAX", INK)], 19.5)


def shapes(img, box, paint, ss=4):
    """Antialiased vector drawing inside `box` (original pixels): paint(draw, P) gets P(x, y) mapping original pixels to the layer."""
    x0, y0, x1, y1 = [v * SCALE for v in box]
    k = SCALE * ss
    layer = Image.new("RGBA", ((x1 - x0) * ss, (y1 - y0) * ss), (0, 0, 0, 0))
    paint(ImageDraw.Draw(layer), lambda x, y: ((x - box[0]) * k, (y - box[1]) * k), k)
    layer = layer.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    region = img.crop((x0, y0, x1, y1)).convert("RGBA")
    img.paste(Image.alpha_composite(region, layer).convert("RGB"), (x0, y0))


def dashed(draw, pts, col, width, on, off):
    """Dashed polyline; lengths in layer pixels."""
    import math
    pos, draw_on, left = 0.0, True, on
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        seg = math.hypot(xb - xa, yb - ya)
        t = 0.0
        while t < seg:
            step = min(left, seg - t)
            if draw_on:
                f0, f1 = t / seg, (t + step) / seg
                draw.line([(xa + (xb - xa) * f0, ya + (yb - ya) * f0), (xa + (xb - xa) * f1, ya + (yb - ya) * f1)], fill=col, width=width)
            t += step
            left -= step
            if left <= 1e-9:
                draw_on = not draw_on
                left = on if draw_on else off


def add_hiblup(img, base):
    """Schematic cost panel: add HIBLUP (teal dashes, out of memory at about 90K) between GBLUP and TOPMAX."""
    import math
    S = SCALE
    # GBLUP legend entry, original pixels, moved up 23 px; HIBLUP entry takes the freed row
    src = base.crop((720 * S, 280 * S, 852 * S, 302 * S))
    img.paste(src, (720 * S, 257 * S))
    shapes(img, LEGEND_TOP, lambda d, P, k: dashed(d, [P(724, 299), P(752, 299)], TEAL + (255,), int(3.2 * k), 7 * k, 4 * k))
    text(img, (759, 307), [("HIBLUP", INK)], 21)

    # curve from the origin to the OOM point just left of the 100K tick (ticks: 10K 710, 40K 789, 100K 864)
    xa, xb, ya, yb, g = 712, 855, 507, 384, 3.4
    curve = [(xa + (xb - xa) * t, ya - (ya - yb) * (math.exp(g * t) - 1) / (math.exp(g) - 1)) for t in [i / 80 for i in range(81)]]

    def paint(d, P, k):
        dashed(d, [P(x, y) for x, y in curve], TEAL + (255,), int(3.0 * k), 8 * k, 4 * k)
        cx, cy, r = 856, 375, 7.5   # red cross like the GBLUP one
        for a_, b_ in (((cx - r, cy - r), (cx + r, cy + r)), ((cx - r, cy + r), (cx + r, cy - r))):
            d.line([P(*a_), P(*b_)], fill=OOM_RED + (255,), width=int(3.4 * k))
    shapes(img, COST_PLOT, paint)
    text(img, (863, 412), [("OOM", OOM_RED)], 21.5)


def build(logo_path):
    orig = Image.open(SRC).convert("RGB")
    img = orig.resize((orig.width * SCALE, orig.height * SCALE), Image.LANCZOS)
    base = img.copy()
    for box in REGIONS:
        clear(img, box)
    if logo_path != "-":
        paste_logo(img, logo_path)
    fix_dual_formula(img, base)
    text(img, (759, 336), [("TOPMAX", INK)], 21)
    text(img, (33, 777), [("TOPMAX", INK)], 19.5)
    text(img, (95, 853), [("TOPMAX", RED), (" matches GBLUP-level accuracy across species & traits", INK)], 23.6)
    text(img, (781, 752), [("33x", NAVY)], 66, bold=True, anchor="c")
    text(img, (1004, 752), [("6.4x", NAVY)], 66, bold=True, anchor="c")
    text(img, (1488, 752), [("377x", NAVY)], 66, bold=True, anchor="c")
    for cx, first, second in [(781, "faster than", "HIBLUP at 80K"), (1004, "less memory than", "HIBLUP at 80K"), (1488, "faster than", "TOP at 30K")]:
        text(img, (cx, 793), [(first, INK)], 23.5, anchor="c")
        text(img, (cx, 818), [(second, INK)], 23.5, anchor="c")
    drop_svr_row(img)
    add_hiblup(img, base)
    return base, img


def verify(base, img):
    """Every pixel outside the declared rectangles must equal the upscaled original."""
    diff = np.asarray(ImageChops.difference(base, img)).sum(axis=2)
    for x0, y0, x1, y1 in CHANGED:
        diff[y0 * SCALE:y1 * SCALE, x0 * SCALE:x1 * SCALE] = 0
    changed = int((diff != 0).sum())
    if changed:
        raise SystemExit(f"{changed} pixels changed outside the declared regions")
    print("Outside the declared rectangles: 0 pixels differ from the original")


if __name__ == "__main__":
    base, img = build(sys.argv[1])
    verify(base, img)
    # Same crop as the manuscript figure: drop the empty margin around the panels
    ink = ImageChops.difference(base, Image.new("RGB", base.size, (255, 255, 255))).convert("L").point(lambda v: 255 if v > 40 else 0).getbbox()
    pad = 8 * SCALE
    crop = (max(0, ink[0] - pad), max(0, ink[1] - pad), min(base.width, ink[2] + pad), min(base.height, ink[3] + pad))
    img.crop(crop).save(sys.argv[2])
    print("Saved", sys.argv[2], img.crop(crop).size)
