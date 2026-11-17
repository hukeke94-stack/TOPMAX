"""Compare penalty-selection rules on one fold: exact spectrum of the standardized training kernel vs rank-k truncations.
Usage: alpha_rules.py <fold> <out.csv>  (rice 18K, 50,000 markers, benchmark split)"""
import sys, time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.utils.extmath import randomized_svd
from timed_topx import load_data, select_split, pick_cols
fold, out = int(sys.argv[1]), sys.argv[2]
here = Path(__file__).resolve().parent
TR = ["Plant height (cm)_BLUP", "Culm length (cm)_BLUP", "Heading date (d)_BLUP", "Panicle length (cm)_BLUP", "Leaf width (cm)_BLUP", "Panicle number_BLUP", "Leaf length (cm)_BLUP"]
AL = np.logspace(-2, 7, 100)
task = {"n": 0, "fold": fold, "sample_seed": 0, "fold_seed": 42, "p": 50000, "trait": TR[0], "geno_npz": str(here / "data/rice/genotype.npz"), "pheno_csv": str(here / "data/rice/phenotype.csv")}
X, y0, names, snps = load_data(task)
cols = pick_cols(task, X.shape[1])
X = np.asarray(X if cols is None else X[:, cols], dtype=np.float32)
tr, te = select_split(len(y0), task)
mu = X[tr].mean(0); sd = X[tr].std(0); sd[sd == 0] = 1
Z = (X - mu) / sd; Ztr, Zte = Z[tr], Z[te]; n = len(tr)
t = time.time(); K = (Ztr @ Ztr.T).astype(np.float64); w, U = np.linalg.eigh(K); w, U = w[::-1], U[:, ::-1]
r = int((w > w[0] * 1e-10).sum()); w, U = w[:r], U[:, :r]; total = float(np.trace(K)); A = (Zte @ Ztr.T).astype(np.float64) @ U
print("eigh", round(time.time() - t), "s rank", r, flush=True)
Ur, Sr, _ = randomized_svd(Ztr, n_components=500, n_iter=5, random_state=42)
ph = pd.read_csv(task["pheno_csv"]).set_index("ID")
def gcv(S2, c, yp, ne, extra_dof=lambda a: 0.0, extra_rss=lambda a: 0.0):
    g = [(((c * a / (S2 + a)) ** 2).sum() + yp(a)) / ne / ((1 - ((S2 / (S2 + a)).sum() + extra_dof(a)) / ne) ** 2 + 1e-300) for a in AL]
    return float(AL[int(np.argmin(g))])
rows = []
for trait in TR:
    yall = ph.reindex(names)[trait].to_numpy(float); ytr = yall[tr]; yt = ytr - ytr.mean(); y2 = float(yt @ yt)
    c = U.T @ yt; P = lambda a: float(np.corrcoef(A @ (c / (w + a)), yall[te])[0, 1])
    res = {"fold": fold, "trait": trait}
    res["a_exact"] = gcv(w, c, lambda a: max(y2 - c @ c, 0), n - 1)
    for k in (500, 1000, 2000):
        ck, wk = c[:k], w[:k]; m = r - k; tk = max(total - wk.sum(), 0) / m
        res[f"a_trunc{k}"] = gcv(wk, ck, lambda a: y2 - ck @ ck, n)                       # current rule
        # tail rule: the r-k remaining eigenvalues replaced by their mean, from the trace (no extra decomposition)
        res[f"a_tail{k}"] = gcv(wk, ck, lambda a: (y2 - ck @ ck) * (a / (tk + a)) ** 2, n - 1, extra_dof=lambda a: m * tk / (tk + a))
    # tail rule on the actual RSVD factors (what TOPMAX would compute)
    cr = Ur.T @ yt; S2r = Sr.astype(np.float64) ** 2; mr = r - 500; tr_ = max(total - S2r.sum(), 0) / mr
    res["a_rsvd_now"] = gcv(S2r, cr, lambda a: y2 - cr @ cr, n)
    res["a_rsvd_tail"] = gcv(S2r, cr, lambda a: (y2 - cr @ cr) * (a / (tr_ + a)) ** 2, n - 1, extra_dof=lambda a: mr * tr_ / (tr_ + a))
    ps = [P(a) for a in AL]; res["a_oracle"] = float(AL[int(np.argmax(ps))]); res["p_oracle"] = max(ps)
    for key in [k for k in list(res) if k.startswith("a_") and k != "a_oracle"]:
        res["p" + key[1:]] = P(res[key])
    rows.append(res); print(trait, {k: round(v, 4) for k, v in res.items() if k.startswith("p_")}, flush=True)
pd.DataFrame(rows).to_csv(out, index=False)
