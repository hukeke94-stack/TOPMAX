"""Shared figure style: Genome Biology double-column width, Arial, fixed colour per method across all figures."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, NullFormatter

OUT = Path(__file__).resolve().parent.parent
FIG = OUT / "figures/final"
FIG.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4
WIDTH = 170 * MM  # double column

INK = "#2E3C4D"
MUTED = "#6F7C8A"
RULE = "#E4E9EF"
COL = {
    "TOPMAX": "#C55A11",
    "GBLUP": "#2E75B6",
    "HIBLUP": "#00A3B4",
    "TOP": "#5BA3A0",
    "LASSO": "#ADB1B9",
    "SVR": "#A6A8B0",
    "LightGBM": "#7B7D88",
    "DNNGP": "#55565F",
}
EDGE = {"TOPMAX": "#8C3D06", "GBLUP": "#1A4E8A", "HIBLUP": "#007380", "TOP": "#3D7A77", "LASSO": "#8A8E97", "SVR": "#83858D",
        "LightGBM": "#5C5E68", "DNNGP": "#3D3E46"}
MARK = {"TOPMAX": "o", "GBLUP": "s", "HIBLUP": "p", "TOP": "s", "LASSO": "^", "SVR": "D", "LightGBM": "v", "DNNGP": "h"}
CROP = {"Maize": "#E69F00", "Rice": "#2B6CB0", "Wheat": "#009E73"}
POPCOL = {"Millet": "#8C6BB1", "Maize CUBIC": "#F0B429", "Soybean": "#5BB5A2", "Chickpea": "#CC6677",
          "Maize NCII": "#E69F00", "Wheat": "#009E73", "Rice": "#2B6CB0"}
POPS = [("Millet_2A", "Millet"), ("CUBIC", "Maize CUBIC"), ("Soybean", "Soybean"), ("Chickpea_2A", "Chickpea"),
        ("NCII", "Maize NCII"), ("wheat", "Wheat"), ("rice", "Rice")]
HEAT = LinearSegmentedColormap.from_list("topmax", ["#FBF9FD", "#EEE8F6", "#DACEEB", "#C0AEDC", "#A18BCB"])  # light purples, apart from the blue/teal/orange method colours; dark text stays readable

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.titlepad": 4,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "grid.color": RULE,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.columnspacing": 1.0,
    "legend.handletextpad": 0.4,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.25,
    "lines.markersize": 3.8,
    "errorbar.capsize": 1.2,
    "savefig.dpi": 600,
    "figure.dpi": 150,
})


def line(ax, x, y, m, **kw):
    """Line with the marker and colour of a method, as in every figure."""
    return ax.plot(x, y, marker=MARK[m], color=COL[m], mec="white", mew=0.45, zorder=3, label=m, **kw)


def bold_legend(leg):
    for t in leg.get_texts():
        t.set_fontweight("bold")
    return leg


def plain_log(axis):
    """Log axis with plain numbers: 0.1, 1, 10, 1,000."""
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))


def f1_axis(ax):
    ax.set_xticks([2000, 10000, 30000, 100000, 500000], ["2K", "10K", "30K", "100K", "500K"])
    ax.xaxis.set_minor_formatter(NullFormatter())


def panel(ax, tag, x=-0.13, y=1.06):
    ax.text(x, y, tag, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom", ha="left", color=INK)


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
