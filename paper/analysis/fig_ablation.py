#!/usr/bin/env python3
"""Fig. S2: stepwise ablation of the prediction module. Rows: simulated maize F1 (100,000 individuals, as in Fig. 6) and the
18K rice panel (14,727 training lines). Versions are cumulative: RSVD only -> + CG (explicit kernel) -> + EC (matrix-free
products, no n x n kernel) -> + warm start (= TOPMAX). Columns: accuracy (PCC), fit time split by stage, peak memory.
Means over traits (fold 1), error bars = standard errors. Inputs: ablation_f1_100k_20261117/*.json, ablation_rice_20261116/ablation_rice.csv."""
import glob
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fig_style import COL, EDGE, INK, MM, MUTED, OUT, RULE, WIDTH, panel, save

A = OUT / "analysis"
VER = [("rsvd_only", "RSVD only"), ("rsvd_cg_explicit", "+ CG (explicit kernel)"), ("cold_cg", "+ Matrix-free products"), ("full", "+ warm start (TOPMAX)")]
STG = [("stage_rsvd", "RSVD + GCV", "#C55A11"), ("stage_grm_build", "Kernel build", "#2E75B6"), ("stage_cg", "CG", "#EBA877"),
       ("other", "Scaling and other", "#D8DCE2")]


def load_f1():
    rows = []
    for f in sorted(glob.glob(str(A / "ablation_f1_100k_20261117/*__*.json"))):
        d = json.load(open(f))
        r = {k: d.get(k) for k in ("variant", "trait", "pcc", "fit_s", "cg_iterations")}
        r["mem"] = d.get("peak_anon_gb")
        for k, v in d.get("stages", {}).items():
            r["stage_" + k] = v
        rows.append(r)
    return pd.DataFrame(rows)


def load_rice():
    d = pd.read_csv(A / "ablation_rice_20261116/ablation_rice.csv")
    d = d[d.variant != "svd"].copy()
    d["mem"] = d.peak_rss_gb
    return d


sets = []
f1 = load_f1()
if len(f1):
    sets.append(("Simulated maize F1 · 100,000 individuals", f1))
sets.append(("Rice · 18,409 lines", load_rice()))
summary = []
fig, axes = plt.subplots(len(sets), 3, figsize=(WIDTH, (46 + 36 * len(sets)) * MM), squeeze=False,
                         gridspec_kw={"width_ratios": [0.8, 1.5, 0.8], "wspace": 0.12, "hspace": 0.55, "left": 0.2, "right": 0.985,
                                      "top": 0.93, "bottom": 0.2 if len(sets) == 2 else 0.3})
y = np.arange(len(VER))
for row, (title, d) in enumerate(sets):
    for c, _, _ in STG[:-1]:
        if c not in d:
            d[c] = 0.0
        d[c] = d[c].fillna(0.0)
    d["other"] = (d.fit_s - d[[c for c, _, _ in STG[:-1]]].sum(axis=1)).clip(lower=0)
    g = d.groupby("variant")
    m, se = g.mean(numeric_only=True).reindex([v for v, _ in VER]), g.sem(numeric_only=True).reindex([v for v, _ in VER])
    for v, lab in VER:
        summary.append({"dataset": title, "version": lab, "traits": int(g.size().get(v, 0)), "pcc": m.loc[v, "pcc"], "fit_s": m.loc[v, "fit_s"],
                        "peak_memory_gb": m.loc[v, "mem"], "cg_iterations": m.loc[v, "cg_iterations"], "kernel_build_s": m.loc[v, "stage_grm_build"],
                        "cg_s": m.loc[v, "stage_cg"], "rsvd_s": m.loc[v, "stage_rsvd"]})
    ax_p, ax_t, ax_m = axes[row]
    # accuracy
    # traits differ in accuracy, so the spread across traits is shown as points rather than an error bar
    ax_p.barh(y, m.pcc, 0.62, color=[COL["TOPMAX"] if v == "full" else "#C9CED6" for v, _ in VER], edgecolor="none")
    jit = np.linspace(-0.18, 0.18, d.trait.nunique())
    for k, (v, _) in enumerate(VER):
        pts = d[d.variant == v].sort_values("trait").pcc.to_numpy()
        ax_p.scatter(pts, k + jit[:len(pts)], s=3.5, color=INK, alpha=0.55, lw=0, zorder=3)
    lo = np.floor((d.pcc.min() - 0.03) * 10) / 10
    ax_p.set_xlim(lo, np.ceil((d.pcc.max() + 0.08) * 20) / 20)
    for yy, v in zip(y, m.pcc):
        ax_p.text(ax_p.get_xlim()[1] - (ax_p.get_xlim()[1] - lo) * 0.02, yy, f"{v:.3f}", va="center", ha="right", fontsize=6.3, color=INK)
    ax_p.set_yticks(y, [lab for _, lab in VER] if True else [])
    ax_p.set_xlabel("PCC")
    # time by stage
    left = np.zeros(len(VER))
    for c, lab, col in STG:
        ax_t.barh(y, m[c], 0.62, left=left, color=col, edgecolor="none", label=lab)
        left += m[c].to_numpy()
    for yy, v, it in zip(y, left, m.cg_iterations):
        ax_t.text(v + left.max() * 0.015, yy, f"{v:.0f} s" + (f"  ({it:.0f} CG it.)" if it > 0 else ""), va="center", fontsize=6.3, color=INK)
    ax_t.set_xlim(0, left.max() * 1.42)
    ax_t.set_xlabel("Fit time per fold (s)")
    ax_t.set_yticks(y, [])
    # memory
    ax_m.barh(y, m.mem, 0.62, color=[COL["TOPMAX"] if v == "full" else "#C9CED6" for v, _ in VER], edgecolor="none")
    for yy, v in zip(y, m.mem):
        ax_m.text(v + m.mem.max() * 0.03, yy, f"{v:.1f}", va="center", fontsize=6.3, color=INK)
    ax_m.set_xlim(0, m.mem.max() * 1.35)
    ax_m.set_xlabel("Peak memory (GB)")
    ax_m.set_yticks(y, [])
    for a in (ax_p, ax_t, ax_m):
        a.invert_yaxis()
        a.tick_params(axis="y", length=0)
        a.xaxis.grid(True, color=RULE)
        a.set_axisbelow(True)
    ax_p.set_title(title, loc="left", color=INK, x=-1.05 if True else 0, ha="left")
    panel(ax_p, "ab"[row], x=-1.25, y=1.12)
h, l = axes[0][1].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.58, 0.0), handlelength=1, handleheight=0.8, frameon=False)
pd.DataFrame(summary).round(4).to_csv(A / "ablation_stepwise_summary.csv", index=False)
save(fig, "FigS2_ablation")
print(pd.DataFrame(summary).round(3).to_string())
