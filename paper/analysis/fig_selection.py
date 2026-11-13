#!/usr/bin/env python3
"""Fig. 4: selection module on real crops. Trait weights, candidate ranks, coincidence, memory and runtime.

Trait weights are recovered from the candidate scores behind the coincidence analysis: log(score) is linear in the
standardized absolute distance to the target, so least squares returns the weights exactly (R2 is printed as a check).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullFormatter
from scipy import stats

from fig_style import COL, EDGE, MARK, CROP, bold_legend, INK, MM, MUTED, RULE, WIDTH, panel, plain_log, save

OUT = Path(__file__).resolve().parent.parent
SRC = OUT / "analysis/source_data"       # candidate scores of TOP and TOPMAX, and the benchmark table
DEMO = OUT / "selection_demodata"        # predicted traits, target and observed traits of each crop
CROPS = [("Maize", "selection_scores_maize.csv", "maize", "Training_observations.csv"),
         ("Rice", "selection_scores_rice.csv", "rice", "Training_observations.csv"),
         ("Wheat", "selection_scores_wheat.csv", "wheat", "Training_observations.csv")]
PCTS = [1, 5, 10, 15, 20]
BINS = [0, 1, 5, 10, 15, 20, 30, 100]
LABELS = ["0-1", "1-5", "5-10", "10-15", "15-20", "20-30", ">30"]

bench = pd.read_csv(SRC / "selection_benchmarks.csv").rename(columns={"runtime_min": "min", "peak_memory_gb": "gb", "epochs": "epoch"})
sim = bench[bench.dataset == "simulated_F1"].drop(columns="dataset")
crops = [bench[bench.dataset == c].drop(columns="dataset") for c in ("maize", "rice", "wheat")]
sim.to_csv(OUT / "analysis/selection_scale_table_b.csv", index=False)

fig = plt.figure(figsize=(WIDTH, 168 * MM))
gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.36, height_ratios=[1, 1, 0.82], left=0.065, right=0.99, top=0.965, bottom=0.05)
weights, rank_rows, flows, weight_stats = [], [], [], []
for j, (sp, f, demo, obs_file) in enumerate(CROPS):
    d = pd.read_csv(SRC / f).rename(columns={"TOPX": "TOPMAX"}).set_index("ID")
    n = len(d)
    test = pd.read_csv(DEMO / demo / "Testing_predictions.csv").set_index("ID").loc[d.index]
    check = pd.read_csv(DEMO / demo / "Check_predictions.csv").set_index("ID").iloc[0]
    sd = pd.read_csv(DEMO / demo / obs_file).set_index("ID").std(ddof=0)
    X = (test - check).abs() / sd
    A = np.column_stack([np.ones(n), X.to_numpy()])
    w = {}
    for m in ("TOP", "TOPMAX"):
        y = np.log(d[m].to_numpy())
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        w[m] = -beta[1:]
        r2 = 1 - (y - A @ beta).var() / y.var()
        print(sp, m, "log-linear R2", round(r2, 10))
    w = pd.DataFrame(w, index=test.columns)
    w.insert(0, "crop", sp)
    weights.append(w.rename_axis("trait").reset_index())
    reg = stats.linregress(w.TOP, w.TOPMAX)
    weight_stats.append({"crop": sp, "n_traits": len(w), "pearson": reg.rvalue, "slope": reg.slope, "intercept": reg.intercept,
                         "spearman": stats.spearmanr(w.TOP, w.TOPMAX).statistic})
    ax = fig.add_subplot(gs[0, j])
    lo, hi = min(w.TOP.min(), w.TOPMAX.min()), max(w.TOP.max(), w.TOPMAX.max())
    pad = 0.08 * (hi - lo)
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, c=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.plot(lim, [reg.intercept + reg.slope * v for v in lim], c=INK, lw=0.9, zorder=2)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.scatter(w.TOP, w.TOPMAX, s=20, color=CROP[sp], zorder=3, edgecolors="white", linewidths=0.4)
    ax.set(xlabel="TOP trait weight", ylabel="TOPMAX trait weight", xlim=lim, ylim=lim)
    ax.set_title(f"{sp} · {len(w)} traits", color=INK)
    ax.text(0.05, 0.95, f"r = {reg.rvalue:.3f}\nslope = {reg.slope:.2f}", transform=ax.transAxes, va="top", fontsize=7)
    ax.set_aspect("equal", adjustable="box")
    panel(ax, "abc"[j], x=-0.22)

    r1 = d.TOP.rank(ascending=False, method="first").to_numpy()
    r2_ = d.TOPMAX.rank(ascending=False, method="first").to_numpy()
    p1, p2 = r1 / n * 100, r2_ / n * 100
    reg = stats.linregress(p1, p2)
    rho = stats.spearmanr(d.TOP, d.TOPMAX).statistic
    for pct in PCTS:
        k = int(n * pct / 100)
        a, b = r1 <= k, r2_ <= k
        miss = a & ~b
        rank_rows.append({"species": sp, "n": n, "intensity_pct": pct, "k": k, "overlap": int((a & b).sum()),
                          "coincidence_pct": (a & b).sum() / k * 100, "n_top_only": int(miss.sum()),
                          "n_topmax_only": int((b & ~a).sum()), "worst_topmax_rank_pct_of_top_only": p2[miss].max() if miss.any() else np.nan,
                          "median_topmax_rank_pct_of_top_only": np.median(p2[miss]) if miss.any() else np.nan,
                          "pearson_score": d.TOP.corr(d.TOPMAX), "spearman_score": rho, "rank_slope": reg.slope, "rank_r2": reg.rvalue ** 2})
    ct = pd.crosstab(pd.cut(p1, BINS, labels=LABELS), pd.cut(p2, BINS, labels=LABELS))
    ct.insert(0, "species", sp)
    flows.append(ct.reset_index().rename(columns={"row_0": "TOP_rank_bin"}))
    ax = fig.add_subplot(gs[1, j])
    ax.add_patch(Rectangle((0, 20), 20, 10, fc="#F2F5F9", lw=0, zorder=0))
    ax.add_patch(Rectangle((20, 0), 10, 20, fc="#F2F5F9", lw=0, zorder=0))
    mask = (p1 <= 30) | (p2 <= 30)
    ax.scatter(p1[mask], p2[mask], s=1.3, color=CROP[sp], alpha=0.55, linewidths=0, rasterized=True, zorder=2)
    ax.plot([0, 30], [reg.intercept, reg.intercept + reg.slope * 30], c=INK, lw=0.8, zorder=3)
    ax.axhline(20, c=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.axvline(20, c=MUTED, lw=0.5, ls=(0, (3, 2)), zorder=1)
    k = int(n * 0.2)
    a, b = r1 <= k, r2_ <= k
    # upper left: within the top 20% under TOP but not under TOPMAX; lower right: the reverse
    ax.text(10, 25, f"TOP only\n{(a & ~b).sum()}", ha="center", va="center", fontsize=6.8)
    ax.text(25, 10, f"TOPMAX only\n{(b & ~a).sum()}", ha="center", va="center", fontsize=6.8)
    ax.text(15, 4.5, f"Both\n{(a & b).sum():,}", ha="center", va="center", fontsize=6.8)
    ax.set(xlim=(0, 30), ylim=(0, 30), xlabel="TOP rank (top %)", ylabel="TOPMAX rank (top %)", xticks=range(0, 31, 10), yticks=range(0, 31, 10))
    ax.set_title(f"{sp} · {n:,} candidates", color=INK)
    ax.text(0.97, 0.05, f"ρ = {rho:.4f}", transform=ax.transAxes, ha="right", fontsize=7)
    ax.set_aspect("equal", adjustable="box")
    panel(ax, "def"[j], x=-0.22)

pd.concat(weights).round(5).to_csv(OUT / "analysis/selection_weights.csv", index=False)
pd.DataFrame(weight_stats).round(4).to_csv(OUT / "analysis/selection_weight_stats.csv", index=False)
rs = pd.DataFrame(rank_rows)
rs.round(4).to_csv(OUT / "analysis/rank_stats.csv", index=False)
pd.concat(flows).to_csv(OUT / "analysis/rank_bin_flow.csv", index=False)

ax = fig.add_subplot(gs[2, 0])
for sp, _, _, _ in CROPS:
    g = rs[rs.species == sp]
    ax.plot(g.intensity_pct, g.coincidence_pct, "-o", color=CROP[sp], mec="white", mew=0.45, lw=1.4, ms=4.2)
    ax.text(20.9, g.coincidence_pct.iloc[-1] + {"Maize": 0.55, "Rice": 0, "Wheat": -0.6}[sp], sp, color=CROP[sp], va="center", fontsize=7, fontweight="bold", clip_on=False)
ax.set(xlabel="Selection intensity (top %)", ylabel="Selection coincidence (%)", ylim=(85, 100), xticks=PCTS, xlim=(0, 26))
ax.spines["bottom"].set_bounds(1, 20)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
panel(ax, "g", x=-0.22)
names = ["Maize", "Rice", "Wheat"]
for ax, col, xlab, tag in ((fig.add_subplot(gs[2, 1]), "gb", "Peak memory (GB)", "h"),
                           (fig.add_subplot(gs[2, 2]), "min", "Runtime (min)", "i")):
    vals = {m: [b[b.method == m].sort_values("n").iloc[-1][col] for b in crops] for m in ("TOP", "TOPMAX")}
    for k in range(3):
        lo, hi = vals["TOPMAX"][k], vals["TOP"][k]
        ax.plot([lo, hi], [k, k], color="#C9D1DA", lw=2.2, solid_capstyle="round", zorder=1)
        ax.text((lo * hi) ** 0.5, k - 0.12, f"{hi / lo:.0f}×", ha="center", va="bottom", fontsize=7, fontweight="bold", color=INK)
    for m in ("TOP", "TOPMAX"):
        ax.plot(vals[m], range(3), MARK[m], color=COL[m], mec="white", mew=0.6, ms=6.5, ls="", label=m, zorder=3)
        for k, vv in enumerate(vals[m]):
            lab = f"{vv:,.1f}" if vv >= 10 else f"{vv:.2f}"
            ax.text(vv, k + 0.2, lab, ha="center", va="top", fontsize=6.3, color=EDGE[m])
    ax.set_ylim(2.65, -0.55)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, which="major")
    ax.set_axisbelow(True)
    ax.set_yticks(range(3), names)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 100 if col == "gb" else 10000)
    ax.set_xlabel(xlab)
    ax.set_xticks([0.1, 1, 10, 100] if col == "gb" else [0.1, 1, 10, 100, 1000, 10000])
    plain_log(ax.xaxis)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", length=0)
    panel(ax, tag, x=-0.26)
    if tag == "h":
        bold_legend(ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.98), ncol=2, handlelength=1, borderaxespad=0))
save(fig, "Fig4_selection")
print(pd.DataFrame(weight_stats).round(4).to_string())
for b, s in zip(crops, names):
    t = b[b.method == "TOP"].sort_values("n").iloc[-1]
    x = b[b.method == "TOPMAX"].sort_values("n").iloc[-1]
    print(s, int(t.n), "speedup", round(t["min"] / x["min"], 1), "mem", round(t.gb / x.gb, 1))
print(sim.to_string())
