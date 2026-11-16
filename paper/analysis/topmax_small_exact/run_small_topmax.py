"""Rerun TOPMAX (prediction/ in this repository) on the four smaller panels of Fig. 2, with the benchmark rows, markers and
five-fold split. Usage: python run_small_topmax.py <folder with CUBIC/, Millet_2A/, Soybean/, Chickpea_2A/> <output.csv>
Each panel folder holds genotype.npz and phenotype.csv (from CropGS-Hub and the original studies)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "prediction"))
from methods import get_method

data, out = Path(sys.argv[1]), sys.argv[2]
rows = []
for ds, trait, p in [("CUBIC", "EW", 50000), ("Millet_2A", "PGW", None), ("Soybean", "HSW_BLUP", None), ("Chickpea_2A", "Yield", None)]:
    z = np.load(data / ds / "genotype.npz", allow_pickle=True)
    X, names = z["X"], z["sample_names"].astype(str)
    if p and p < X.shape[1]:
        X = X[:, np.sort(np.random.default_rng(42).choice(X.shape[1], size=p, replace=False))]
    ph = pd.read_csv(data / ds / "phenotype.csv")
    lookup = dict(zip(ph[ph.columns[0]].astype(str), pd.to_numeric(ph[trait], errors="coerce")))
    y = np.array([lookup.get(s, np.nan) for s in names])
    keep = ~np.isnan(y)
    X, y = np.asarray(X[keep], dtype=np.float32), y[keep].astype(np.float32)
    for f, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=42).split(y), start=1):
        m = get_method("TOPMAX")
        yh, sec = m.fit_predict(X[tr], y[tr], X[te])
        rows.append(dict(dataset=ds, trait=trait, fold=f, n_train=len(tr), exact_svd=m.used_exact_svd, best_alpha=m.best_alpha,
                         cg_iterations=m.cg_iterations, pcc=float(np.corrcoef(yh, y[te])[0, 1]), seconds=round(sec, 2)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(out, index=False)
