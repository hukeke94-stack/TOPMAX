#!/usr/bin/env python3
"""Fig. 2 (accuracy), Fig. 3 (resources) and Fig. 5 (scale). Run fig_selection.py first; it writes the selection table."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from fig_style import COL, EDGE, MARK, HEAT, bold_legend, line, INK, MM, MUTED, OUT, POPCOL, POPS, RULE, WIDTH, f1_axis, panel, plain_log, save

from matplotlib.colors import to_rgb


def tint(c, k=0.35):
    """Method colour lightened toward white, used for bar fills; outlines keep the full colour."""
    r = to_rgb(c)
    return tuple(1 - (1 - v) * (1 - k) for v in r)


raw = pd.read_csv(OUT / "analysis/results_summary.csv")
raw["m"] = raw.id.str.extract(r"^([A-Za-z]+(?:_sub1000)?)")
LAB = dict(POPS)
d = raw[raw.dataset.notna() & raw.variant.isna() & raw.rsvd_k.isna() & ~raw.id.str.contains("density")].copy()
d.loc[(d.status == "OK") & d.pcc.isna(), "pcc"] = 0.0  # constant prediction: no selected marker
d = d[d.status == "OK"]

# SVR uses scikit-learn default hyperparameters (analysis/svr_default); the earlier grid-searched runs stay in results/
_svr = pd.read_csv(OUT / "analysis/svr_default/svr_default_results.csv").set_index(["dataset", "fold"]).pcc_default
_is = (d.m == "SVR") & d.dataset.isin(["Millet_2A", "CUBIC", "Soybean", "Chickpea_2A"])
d.loc[_is, "pcc"] = [_svr[(a_, int(f_))] for a_, f_ in zip(d.loc[_is, "dataset"], d.loc[_is, "fold"])]
ns = d.groupby("dataset").n_train.first() + d.groupby("dataset").n_test.first()
ORDER = [p for p, _ in POPS]

# Fig. 2: accuracy. a six methods in the four smallest populations; b, c HIBLUP and TOPMAX against GBLUP in the fold run with all three
fig = plt.figure(figsize=(WIDTH, 112 * MM))
gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1], hspace=0.42, wspace=0.3, left=0.075, right=0.985, top=0.93, bottom=0.085)
a = fig.add_subplot(gs[0, :])
methods = ["GBLUP", "HIBLUP", "TOPMAX", "LASSO", "LightGBM", "DNNGP"]
small = ORDER[:4]
w = 0.135
for i, m in enumerate(methods):
    for x, p in zip(np.arange(4) + (i - 2.5) * w, small):
        g = d[(d.m == m) & (d.dataset == p)].pcc
        a.bar(x, g.mean(), w, color=tint(COL[m]), edgecolor=COL[m], lw=0.7, zorder=2)
        a.errorbar(x, g.mean(), yerr=g.sem(), fmt="none", ecolor=EDGE[m], elinewidth=0.7, capsize=1.4, capthick=0.6, zorder=3)
a.set_xticks(range(4), [LAB[p] for p in small], fontweight="bold")
for i, p in enumerate(small):
    a.annotate(f"n = {int(ns[p]):,}", (i, 0), xycoords=a.get_xaxis_transform(), xytext=(0, -12.5), textcoords="offset points", ha="center", va="top", fontsize=6.3, color=MUTED)
a.set_xlim(-0.55, 3.55)
a.set_ylim(0, 0.72)
a.set_ylabel("Prediction accuracy (PCC)")
a.yaxis.grid(True)
a.set_axisbelow(True)
a.tick_params(axis="x", length=0)
bold_legend(a.legend(handles=[Rectangle((0, 0), 1, 1, fc=tint(COL[m]), ec=COL[m], lw=0.8) for m in methods], labels=methods, ncol=6, loc="upper center",
                     bbox_to_anchor=(0.5, 1.16), handlelength=1, handleheight=0.8, columnspacing=1.0))
panel(a, "a", x=-0.06)

# b, c: the three multi-trait populations (not shown in a), in the fold run with all three methods (fold 1)
f1 = d[d.m.isin(["GBLUP", "HIBLUP", "TOPMAX"]) & (d.fold == 1)].pivot_table(index=["dataset", "trait"], columns="m", values="pcc", aggfunc="first").dropna()
f1 = f1.loc[[p for p, _ in POPS[4:]]]
for col, m, tag in ((0, "HIBLUP", "b"), (1, "TOPMAX", "c")):
    b = fig.add_subplot(gs[1, col])
    b.plot([0.3, 1], [0.3, 1], c=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    b.grid(True)
    b.set_axisbelow(True)
    for p, lab in POPS[4:]:
        s = f1.loc[p]
        b.scatter(s.GBLUP, s[m], s=16, color=POPCOL[lab], edgecolors="white", linewidths=0.35, zorder=3, label=lab)
    b.set(xlabel="GBLUP PCC", ylabel=f"{m} PCC", xlim=(0.3, 1), ylim=(0.3, 1), xticks=np.arange(0.4, 1.01, 0.2), yticks=np.arange(0.4, 1.01, 0.2))
    b.set_aspect("equal", adjustable="box")
    dm = (f1[m] - f1.GBLUP).abs().max()
    txt = f"max |ΔPCC| = {dm:.3f}" if dm >= 5e-4 else "max |ΔPCC| < 0.0001"
    b.text(0.04, 0.97, f"{len(f1)} traits\n{txt}", transform=b.transAxes, va="top", fontsize=7)
    if col == 1:
        b.legend(loc="lower right", fontsize=6, handletextpad=0.1, labelspacing=0.18, borderpad=0, markerscale=0.9)
    panel(b, tag, x=-0.2)
save(fig, "Fig2_accuracy")
long = d[d.m.isin(["GBLUP", "HIBLUP", "TOPMAX"])].pivot_table(index=["dataset", "trait"], columns="m", values="pcc", aggfunc="mean")

# Fig. 3: resources in the three populations with more than 5,000 individuals (the summary table keeps all seven)
r = d[d.m.isin(["GBLUP", "HIBLUP", "TOPMAX"]) & (d.fold == 1)].copy()
r["algo_s"] = r.core_s.fillna(r.fit_s + r.predict_s)
r["mem"] = r.peak_tree_anon_gb.fillna(r.max_single_rss_gb)
tab = []
for col in ("algo_s", "mem"):
    for m in ["GBLUP", "HIBLUP", "TOPMAX"]:
        tab.append(r[r.m == m].groupby("dataset")[col].agg(["count", "mean", "sem"]).assign(method=m, metric=col))
BIG = ORDER[4:]
fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 64 * MM), gridspec_kw={"wspace": 0.28, "left": 0.075, "right": 0.99, "top": 0.92, "bottom": 0.17})
w = 0.27
for col, a, lab, tag, fmt in (("algo_s", ax[0], "Runtime per fold (s)", "a", "{:.0f}"), ("mem", ax[1], "Peak memory (GB)", "b", "{:.1f}")):
    top = r[r.dataset.isin(BIG)].groupby(["m", "dataset"])[col].mean().max()
    for i, m in enumerate(["GBLUP", "HIBLUP", "TOPMAX"]):
        g = r[(r.m == m) & r.dataset.isin(BIG)].groupby("dataset")[col]
        mean, sem = g.mean().reindex(BIG), g.sem().reindex(BIG).fillna(0)
        x = np.arange(3) + (i - 1) * w
        a.bar(x, mean, w, color=tint(COL[m]), edgecolor=COL[m], lw=0.7, zorder=2, label=m)
        a.errorbar(x, mean, yerr=sem, fmt="none", ecolor=EDGE[m], elinewidth=0.6, capsize=1.4, capthick=0.5, zorder=3)
        for xx, v, e in zip(x, mean, sem):
            a.text(xx, v + e + top * 0.015, fmt.format(v), ha="center", va="bottom", fontsize=6.3, color=EDGE[m])
    a.set_xticks(range(3), [LAB[p] for p in BIG], fontweight="bold")
    for k, p in enumerate(BIG):
        a.annotate(f"n = {int(ns[p]):,}", (k, 0), xycoords=a.get_xaxis_transform(), xytext=(0, -12.5), textcoords="offset points", ha="center", va="top", fontsize=6.3, color=MUTED)
    a.set_xlim(-0.55, 2.55)
    a.set_ylim(0, top * 1.12)
    a.set_ylabel(lab)
    a.yaxis.grid(True)
    a.set_axisbelow(True)
    a.tick_params(axis="x", length=0)
    panel(a, tag, x=-0.13, y=1.04)
bold_legend(ax[0].legend(handles=[Rectangle((0, 0), 1, 1, fc=tint(COL[m]), ec=COL[m], lw=0.8) for m in ["GBLUP", "HIBLUP", "TOPMAX"]],
                         labels=["GBLUP", "HIBLUP", "TOPMAX"], loc="upper left", handlelength=1, handleheight=0.8))
save(fig, "Fig3_resources")
pd.concat(tab).round(4).to_csv(OUT / "analysis/fig3_population_summary.csv")

# Fig. 5: scale. Prediction (upper) and selection (lower) with one x axis per column
MEMRUN = raw.id.str.contains("_mem2026", na=False)  # 2026-11-12 reruns that recorded anonymous memory; their runtimes are from another host and are not used
s = raw[raw.dataset.isna() & raw.variant.isna() & raw.p.isna() & raw.rsvd_seed.isna() & raw.rsvd_k.isna() & raw.cg_tol.isna()
        & raw.method.isin(["GBLUP", "HIBLUP", "TOPMAX"]) & (raw.trait == "EW") & (raw.fold == 1) & ~MEMRUN].copy()
# GBLUP 30K only completed with the copy-free wrapper; other sizes use the standard run
lean = s.id.str.contains("lean")
s = s[(lean & (s.n >= 30000)) | (~lean & ~((s.n == 30000) & (s.method == "GBLUP")))]
ok = s[s.status == "OK"].copy()
# Memory: anonymous memory of the process tree, excluding file pages mapped from the genotype matrix (same measure as Fig. 3)
_anon = raw[MEMRUN & (raw.status == "OK")].set_index(["method", "n"]).peak_tree_anon_gb
# Two early GBLUP runs (2K, 10K) predate the anonymous-memory log; use another fold of the same size (memory does not depend on the fold)
_same = raw[raw.dataset.isna() & raw.variant.isna() & raw.p.isna() & (raw.status == "OK") & ~MEMRUN].groupby(["method", "n"]).peak_tree_anon_gb.median()
ok["mem"] = [_anon.get((m_, n_), a_ if pd.notna(a_) else _same.get((m_, n_))) for m_, n_, a_ in zip(ok.method, ok.n, ok.peak_tree_anon_gb)]
assert ok["mem"].notna().all(), "missing anonymous memory"
sel = pd.read_csv(OUT / "analysis/selection_scale_table_b.csv")  # written by fig_selection.py
fig = plt.figure(figsize=(WIDTH, 172 * MM))
outer = fig.add_gridspec(2, 1, height_ratios=[0.2, 1], hspace=0.17, left=0.085, right=0.985, top=0.985, bottom=0.05)
a = fig.add_subplot(outer[0])
piv = ok.pivot_table(index="method", columns="n", values="pcc").reindex(["GBLUP", "HIBLUP", "TOPMAX"])
A = np.ma.masked_invalid(piv.to_numpy())
HEAT.set_bad("#F3F4F6")
im = a.imshow(A, aspect="auto", cmap=HEAT, vmin=0.58, vmax=0.81)
a.set_xticks(range(piv.shape[1]), [f"{int(c / 1000)}K" for c in piv.columns], fontsize=6.6)
a.set_yticks(range(3), piv.index)
a.set_xticks(np.arange(-0.5, piv.shape[1]), minor=True)
a.set_yticks(np.arange(-0.5, 3), minor=True)
a.grid(which="minor", color="white", lw=0.8)
a.tick_params(which="both", length=0)
for sp in a.spines.values():
    sp.set_visible(False)
for i in range(3):
    for j in range(piv.shape[1]):
        v = piv.iloc[i, j]
        if not np.isnan(v):
            a.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=5.7, color=INK)
cols = list(piv.columns)
a.add_patch(Rectangle((-0.5, 1.5), piv.shape[1], 1, fc="none", ec=COL["TOPMAX"], lw=1.0, zorder=5))
for m, n_ in (("GBLUP", 40000), ("HIBLUP", 90000)):
    a.text(cols.index(n_), list(piv.index).index(m), "OOM", ha="center", va="center", fontsize=5.7, color="#B23A3A", fontweight="bold")
cb = fig.colorbar(im, ax=a, fraction=0.012, pad=0.008, ticks=[0.6, 0.7, 0.8])
cb.set_label("PCC", fontsize=7)
cb.ax.tick_params(labelsize=6.3, length=1.5, width=0.4)
cb.outline.set_linewidth(0.4)
panel(a, "a", x=-0.075, y=1.02)
inner = outer[1].subgridspec(2, 2, hspace=0.13, wspace=0.2)


def gap(ax_, x, lo, hi, text, log=True):
    ax_.annotate("", (x, lo), (x, hi), arrowprops={"arrowstyle": "<|-|>,head_width=0.13,head_length=0.28", "lw": 0.7, "color": INK, "shrinkA": 1.5, "shrinkB": 1.5})
    y = (lo * hi) ** 0.5 if log else (lo + hi) / 2
    ax_.text(x * 1.1, y, text, va="center", fontsize=8, fontweight="bold", color=INK)


rt, me = fig.add_subplot(inner[0, 0]), fig.add_subplot(inner[0, 1])
for ax_, col, lab, tag in ((rt, "wall_s", "Runtime per fold (min)", "b"), (me, "mem", "Peak memory (GB)", "c")):
    for m in ["GBLUP", "HIBLUP", "TOPMAX"]:
        g = ok[ok.method == m].sort_values("n")
        line(ax_, g.n, g[col] / (60 if col == "wall_s" else 1), m)
    ax_.set_xscale("log")
    ax_.set_ylabel(lab)
    ax_.tick_params(labelbottom=False)
    ax_.yaxis.grid(True, which="major")
    ax_.set_axisbelow(True)
    panel(ax_, tag, x=-0.15, y=1.03)
rt.set_yscale("log")
bold_legend(rt.legend(loc="lower right", borderaxespad=0.2))
rt.set_ylim(top=900)
for n_, m_, ha, dx, lab_ in ((35000, "GBLUP", "right", 0.9, "40K"), (88000, "HIBLUP", "left", 1.1, "90K")):
    rt.axvline(n_, color=EDGE[m_], lw=0.7, ls=(0, (3, 2)), zorder=1)
    rt.text(n_ * dx, 480, f"{m_}: OOM at {lab_}", ha=ha, va="center", fontsize=6.2, fontweight="bold", color=EDGE[m_],
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": EDGE[m_], "lw": 0.5}, zorder=5)
rt.set_title("Prediction module", loc="left", color=INK, pad=4)
me.set_title("8 threads, 90 GB limit", loc="left", color=MUTED, fontweight="normal", pad=4)
plain_log(rt.yaxis)
me.axhline(90, c="#A23B3B", ls=(0, (4, 2)), lw=0.7, zorder=2)
me.text(2100, 91.5, "90 GB limit", fontsize=6.6, color="#A23B3B", va="bottom", fontweight="bold")
hi80, tx80 = ok[(ok.method == "HIBLUP") & (ok.n == 80000)].iloc[0], ok[(ok.method == "TOPMAX") & (ok.n == 80000)].iloc[0]
gap(rt, 80000, tx80.wall_s / 60, hi80.wall_s / 60, f"{hi80.wall_s / tx80.wall_s:.0f}×")
gap(me, 80000, tx80["mem"], hi80["mem"], f"{hi80['mem'] / tx80['mem']:.1f}×", log=False)
srt = fig.add_subplot(inner[1, 0], sharex=rt)
sme = fig.add_subplot(inner[1, 1], sharex=me)
for ax_, col, lab, tag in ((srt, "min", "Runtime (min)", "d"), (sme, "gb", "Peak memory (GB)", "e")):
    for m in ["TOP", "TOPMAX"]:
        g = sel[sel.method == m].sort_values("n")
        line(ax_, g.n, g[col], m)
    ax_.set_xscale("log")
    ax_.set_yscale("log")
    ax_.set(xlabel="Simulated F1 sample size", ylabel=lab)
    ax_.set_xlim(1600, 620000)
    ax_.yaxis.grid(True, which="major")
    ax_.set_axisbelow(True)
    panel(ax_, tag, x=-0.15, y=1.03)
bold_legend(srt.legend(loc="upper left", borderaxespad=0.1))
srt.set_title("Selection module", loc="left", color=INK, pad=4)
sme.set_title("Compute cluster; TOPMAX 2 cores, TOP 10–20 cores", loc="left", color=MUTED, fontweight="normal", pad=4)
plain_log(srt.yaxis)
plain_log(sme.yaxis)
for ax_ in (srt, sme):
    f1_axis(ax_)
    ax_.set_xlim(1600, 620000)
t30, x30 = sel[(sel.method == "TOP") & (sel.n == 30000)].iloc[0], sel[(sel.method == "TOPMAX") & (sel.n == 30000)].iloc[0]
gap(srt, 30000, x30["min"], t30["min"], f"{t30['min'] / x30['min']:.0f}×")
gap(sme, 30000, x30.gb, t30.gb, f"{t30.gb / x30.gb:.0f}×")
save(fig, "Fig5_scale")

# Overall speedup of predicting 18 traits and then learning weights, without cross-validation repeats
ewf = ok.set_index(["method", "n"]).wall_s
rows = []
for n in (10000, 20000, 30000):
    tp, tx = sel[(sel.method == "TOP") & (sel.n == n)].iloc[0]["min"], sel[(sel.method == "TOPMAX") & (sel.n == n)].iloc[0]["min"]
    x = ewf["TOPMAX", n] * 18 / 60 + tx
    rows.append({"n": n, "TOPMAX_min": x, "HIBLUP_plus_TOP_min": ewf["HIBLUP", n] * 18 / 60 + tp, "GBLUP_plus_TOP_min": ewf["GBLUP", n] * 18 / 60 + tp})
ov = pd.DataFrame(rows)
ov["speedup_vs_HIBLUP_plus_TOP"] = ov.HIBLUP_plus_TOP_min / ov.TOPMAX_min
ov["speedup_vs_GBLUP_plus_TOP"] = ov.GBLUP_plus_TOP_min / ov.TOPMAX_min
ov.round(1).to_csv(OUT / "analysis/overall_speedup.csv", index=False)
print(ov.round(1).to_string())
print("done", len(long), piv.shape)
