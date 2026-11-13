#!/usr/bin/env python3
"""Additional file 1 figures: Fig. S1 TOPMAX scaling fit, Fig. S2 ablation, Fig. S3 marker density."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats

from fig_style import COL, EDGE, MARK, line, bold_legend, INK, MM, MUTED, OUT, RULE, WIDTH, f1_axis, panel, plain_log, save

raw = pd.read_csv(OUT / "analysis/results_summary.csv")
d = raw[raw.status == "OK"]

# Fig. S2: ablation of the prediction module (EW, fold 1)
idx = raw.set_index("id")
VAR = [("", "Full TOPMAX"), ("_variantcold_cg", "No warm start"), ("_variantrsvd_only", "RSVD only, no CG"),
       ("_variantexplicit_grm", "Explicit GRM solve"), ("_variantexact_gcv", "Full eigendecomposition")]
STG = [("stage_rsvd", "RSVD", "#C55A11"), ("stage_cg", "CG", "#EBA877"), ("stage_grm_build", "GRM build", "#2E75B6"),
       ("stage_eigh", "Eigendecomposition", "#123A6B"), ("stage_cholesky_solve", "Cholesky solve", "#00A3B4")]
rows = []
for n in (10000, 30000, 100000):
    for suf, lab in VAR:
        i = f"TOPMAX_n{n}_EW_f1_s20260611{suf}"
        if i not in idx.index:
            continue
        r = idx.loc[i]
        row = {"n": n, "variant": lab, "pcc": r.pcc, "wall_s": r.wall_s, "fit_s": r.fit_s, "anon_gb": r.peak_tree_anon_gb,
               "rss_gb": r.max_single_rss_gb, "alpha": r.best_alpha, "cg_iter": r.cg_iterations}
        for s, _, _ in STG:
            row[s] = r.get(s, 0)
        row["other_s"] = r.fit_s - sum(0 if pd.isna(row[s]) else row[s] for s, _, _ in STG)
        rows.append(row)
t = pd.DataFrame(rows).fillna({s: 0 for s, _, _ in STG})
t.round(4).to_csv(OUT / "analysis/ablation_summary.csv", index=False)
ns = sorted(t.n.unique())
fig, ax = plt.subplots(1, len(ns), figsize=(WIDTH, 62 * MM), gridspec_kw={"wspace": 0.12, "left": 0.13, "right": 0.97, "top": 0.92, "bottom": 0.27})
for k, (a, n) in enumerate(zip(ax, ns)):
    labels = [lab for _, lab in VAR]
    s = t[t.n == n].set_index("variant").reindex(labels).reset_index()  # same rows in every panel; 100K ran three variants
    s[[c for c, _, _ in STG] + ["other_s"]] = s[[c for c, _, _ in STG] + ["other_s"]].fillna(0)
    left = pd.Series(0.0, index=s.index)
    for c, lab, col in STG:
        a.barh(s.index, s[c], left=left, height=0.68, color=col, lw=0, label=lab)
        left += s[c]
    a.barh(s.index, s.other_s.clip(lower=0), left=left, height=0.68, color="#D8DCE2", lw=0, label="Data loading and scaling")
    left += s.other_s.clip(lower=0)
    for i, r in s.iterrows():
        if not np.isnan(r.pcc):
            a.text(left[i] + left.max() * 0.02, i, f"{r.pcc:.3f}", va="center", fontsize=6.5)
    a.set_yticks(s.index, s.variant if k == 0 else [""] * len(s))
    a.invert_yaxis()
    a.tick_params(axis="y", length=0)
    a.set(xlabel="Fit time (s)", xlim=(0, left.max() * 1.22))
    a.set_title(f"n = {n:,}", color=INK)
    panel(a, "abc"[k], x=-0.42 if k == 0 else -0.06, y=1.02)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.55, 0.0), handlelength=1, handleheight=0.8)
save(fig, "FigS2_ablation")

# Fig. S3: marker density
fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 66 * MM), gridspec_kw={"wspace": 0.27, "left": 0.07, "right": 0.99, "top": 0.92, "bottom": 0.14})
n_ = d[d.id.str.contains("density")].copy()
n_["time"] = n_.core_s.fillna(n_.stage_rsvd + n_.stage_cg)
for m in ("TOPMAX",):
    g = n_[n_.method == m].groupby("p").time
    ax[0].errorbar(g.mean().index / 1000, g.mean(), yerr=g.sem(), fmt="-" + MARK[m], color=COL[m], mec="white", mew=0.45, ecolor=EDGE[m], elinewidth=0.6, capsize=1.3, label=m)
ax[0].set(xlabel="Markers (thousands)", ylabel="TOPMAX algorithm time per fold (s)")
ax[0].set_title("Maize NCII · 5,820 individuals", loc="left", color=INK)
f = d[d.dataset.isna() & (d.n == 20000) & (d.trait == "EW") & (d.fold == 1) & d.variant.isna() & d.rsvd_k.isna() & ~d.id.str.contains("_mem2026", na=False)].copy()
f["time"] = f.core_s.fillna(f.stage_rsvd + f.stage_cg)
f["p2"] = f.p.fillna(50000)
for m in ("TOPMAX",):
    g = f[f.method == m].sort_values("p2")
    line(ax[1], g.p2 / 1000, g.time, m)
ax[1].set(xlabel="Markers (thousands)", ylabel="TOPMAX algorithm time per fold (s)")
ax[1].set_title("Simulated F1 · 20,000 individuals", loc="left", color=INK)
for a, tag in zip(ax, "ab"):
    a.set_ylim(bottom=0)
    a.yaxis.grid(True)
    a.set_axisbelow(True)
    panel(a, tag, x=-0.13, y=1.03)
save(fig, "FigS3_marker_density")
print(n_.groupby(["p", "method"]).time.mean().round(1).to_string())

# Fig. S1: TOPMAX scaling fit
MEMRUN = d.id.str.contains("_mem2026", na=False)  # reruns that recorded anonymous memory; runtime comes from the original host
s = d[(d.method == "TOPMAX") & d.dataset.isna() & d.variant.isna() & d.p.isna() & d.rsvd_seed.isna() & d.rsvd_k.isna() & d.cg_tol.isna()
      & (d.trait == "EW") & (d.fold == 1) & ~MEMRUN].sort_values("n")
anon = d[MEMRUN & (d.method == "TOPMAX")].set_index("n").peak_tree_anon_gb
n = s.n.to_numpy(float)
tm, m = s.wall_s.to_numpy(), anon.reindex(s.n).to_numpy()
fit = n >= 10000
pl_t = stats.linregress(np.log10(n[fit]), np.log10(tm[fit]))
lin_t = stats.linregress(n[fit], tm[fit])
mm = fit
lin_m = stats.linregress(n[mm], m[mm])
pd.Series({"n_points_time": int(fit.sum()), "time_power": pl_t.slope, "time_power_r2": pl_t.rvalue ** 2, "time_s_per_1000_samples": lin_t.slope * 1000,
           "time_intercept_s": lin_t.intercept, "time_linear_r2": lin_t.rvalue ** 2, "n_points_mem": int(mm.sum()), "mem_gb_per_10000": lin_m.slope * 1e4,
           "mem_intercept_gb": lin_m.intercept, "mem_linear_r2": lin_m.rvalue ** 2, "mem_linear_pred_500k": lin_m.slope * 5e5 + lin_m.intercept,
           "mem_observed_500k": float(m[n == 500000][0])}).to_csv(OUT / "analysis/topmax_scaling_fit.csv", header=["value"])
fig, ax = plt.subplots(1, 2, figsize=(WIDTH, 66 * MM), gridspec_kw={"wspace": 0.25, "left": 0.07, "right": 0.99, "top": 0.93, "bottom": 0.14})
xs = np.logspace(4, np.log10(5e5), 50)
ax[0].plot(xs, 10 ** (pl_t.intercept + pl_t.slope * np.log10(xs)) / 60, c=INK, lw=0.8, zorder=2)
ax[0].plot(n, tm / 60, "o", color=COL["TOPMAX"], mec="white", mew=0.35, ms=4, zorder=3)
ax[0].set(xscale="log", yscale="log", xlabel="Simulated F1 sample size", ylabel="Runtime per fold (min)")
f1_axis(ax[0])
plain_log(ax[0].yaxis)
ax[0].text(0.05, 0.93, f"runtime $\\propto$ n$^{{{pl_t.slope:.2f}}}$, R² = {pl_t.rvalue ** 2:.3f}", transform=ax[0].transAxes, va="top")
xl = np.array([1e4, 5e5])
ax[1].plot(xl / 1000, lin_m.slope * xl + lin_m.intercept, c=INK, lw=0.8, ls=(0, (3, 2)), zorder=2)
ax[1].plot(n / 1000, m, "o", color=COL["TOPMAX"], mec="white", mew=0.35, ms=4, zorder=3)
ax[1].axhline(90, c="#A23B3B", lw=0.7, ls=(0, (4, 2)))
ax[1].text(10, 91.5, "90 GB limit", fontsize=6.6, color="#A23B3B", va="bottom")
ax[1].set(xlabel="Simulated F1 sample size (thousands)", ylabel="Peak memory (GB)", ylim=(0, 100))
ax[1].text(0.05, 0.78, f"{lin_m.slope * 1e4:.2f} GB per 10,000 individuals\nR² = {lin_m.rvalue ** 2:.3f}", transform=ax[1].transAxes, va="top")
for a, tag in zip(ax, "ab"):
    a.yaxis.grid(True)
    a.set_axisbelow(True)
    panel(a, tag, x=-0.13, y=1.02)
save(fig, "FigS1_topmax_scaling")
print("slope", round(pl_t.slope, 3), "mem", round(lin_m.slope * 1e4, 2))
