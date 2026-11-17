#!/usr/bin/env python3
"""Fig. 1 as vector graphics: the composed raster (fig1_compose.py output) is traced to filled paths with vtracer, and the
two small gradient elements that tracing renders poorly (the genotype-matrix icon and the heatmap colour bar) are redrawn
as vector rectangles with colours sampled from the raster. Writes Fig1_overview.pdf (vector) and Fig1_overview.png.

Usage: fig1_vectorize.py <composed raster png> <out dir>
Needs vtracer, cairosvg and opencv-python-headless (pip; environment .venv_trace).
"""
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

src, out = Path(sys.argv[1]), Path(sys.argv[2])
UP = 4                                    # trace at four times the raster size; a median filter there removes anti-aliasing noise
img = Image.open(src).convert("RGB")
a = np.asarray(img).astype(int)
work = out / "fig1_up.png"
import cv2  # noqa: E402
up = cv2.resize(np.asarray(img), None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
Image.fromarray(cv2.medianBlur(up, 5)).save(work)

import vtracer  # noqa: E402
svg_path = out / "fig1_trace.svg"
vtracer.convert_image_to_svg_py(str(work), str(svg_path), colormode="color", hierarchical="stacked", mode="spline",
                                filter_speckle=10, color_precision=6, layer_difference=24, corner_threshold=100,
                                length_threshold=8.0, max_iterations=10, splice_threshold=60, path_precision=3)
svg = svg_path.read_text()


def hexc(rgb):
    return "#%02x%02x%02x" % tuple(int(v) for v in rgb)


def med(x0, x1, y0, y1):
    return np.median(a[y0:y1, x0:x1].reshape(-1, 3), axis=0)


over = []
# genotype-matrix icon (raster px): border box, 5 x 6 cells between the white gap lines
over.append(f'<rect x="{103*UP}" y="{445*UP}" width="{212*UP}" height="{247*UP}" rx="{7*UP}" fill="#ffffff" '
            f'stroke="{hexc(med(103, 106, 540, 600))}" stroke-width="{3*UP}"/>')
cols = [(111, 147), (151, 187), (191, 227), (231, 266), (271, 306)]
rows = [(454, 490), (494, 529), (533, 567), (571, 605), (609, 644), (648, 683)]
for y0, y1 in rows:
    for x0, x1 in cols:
        c = med(x0 + 8, x1 - 8, y0 + 8, y1 - 8)
        over.append(f'<rect x="{x0*UP}" y="{y0*UP}" width="{(x1-x0+1)*UP}" height="{(y1-y0+1)*UP}" fill="{hexc(c)}"/>')
# heatmap colour bar: one rectangle per raster row, then the outline
for y in range(1237, 1515):
    over.append(f'<rect x="{1179*UP}" y="{y*UP}" width="{31*UP}" height="{UP + 0.6}" fill="{hexc(med(1186, 1203, y, y + 1))}"/>')
over.append(f'<rect x="{1177*UP}" y="{1235*UP}" width="{35*UP}" height="{281*UP}" fill="none" stroke="{hexc(med(1176, 1178, 1300, 1400))}" stroke-width="{3*UP}"/>')
# logo: replaced by the vector logo traced from the high-resolution design (logo_v5_traced.svg); raster bbox x 1355-1894, y 92-223
logo = (Path(__file__).resolve().parent.parent / "figures/fig1_gpt/logo_v5_traced.svg").read_text()
lw, lh = (float(v) for v in re.search(r'width="([\d.]+)" height="([\d.]+)"', logo).groups())
paths = re.findall(r"<path [^>]+/>", logo)[1:]            # the first path is the white background
x0, x1, y0, y1 = 1355, 1894, 92, 223
k = (x1 - x0) / lw * UP
panel_bg = hexc(med(1400, 1440, 240, 260))
def _recolor(pth):   # near-white logo layers (background and counters) take the panel colour
    m = re.search(r'fill="#([0-9A-Fa-f]{6})"', pth)
    rgb = [int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4)]
    return pth.replace(m.group(0), f'fill="{panel_bg}"') if min(rgb) > 235 else pth
paths = [_recolor(pth) for pth in paths]
over.append(f'<rect x="{1340*UP}" y="{80*UP}" width="{570*UP}" height="{158*UP}" fill="{panel_bg}"/>')
over.append(f'<g transform="translate({x0*UP},{(y0 + y1) / 2 * UP - lh * k / 2}) scale({k})">' + "".join(paths) + "</g>")
svg = svg.replace("</svg>", "\n".join(over) + "\n</svg>")
svg_path.write_text(svg)

import cairosvg  # noqa: E402
W_PX = 170 / 25.4 * 96                    # 170 mm at cairosvg's 96 px per inch
cairosvg.svg2pdf(url=str(svg_path), write_to=str(out / "Fig1_overview.pdf"), output_width=W_PX, output_height=W_PX * img.height / img.width)
cairosvg.svg2png(url=str(svg_path), write_to=str(out / "Fig1_overview.png"), output_width=img.width * 3 // 2)
print("paths", svg.count("<path"), "overlays", len(over))
