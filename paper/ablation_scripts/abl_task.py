#!/usr/bin/env python3
"""One ablation task: rice 18K, fold 1, one trait, one variant. Writes results/<variant>__<trait>.json (skips if present)."""
import json, os, resource, sys, time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from timed_topx import load_data, select_split, pick_cols, run_topx

variant, trait, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
tag = "".join(c for c in trait.split("_BLUP")[0] if c.isalnum())
res = out / f"{variant}__{tag}.json"
if res.exists():
    sys.exit(0)
here = Path(__file__).resolve().parent
task = {"n": 0, "fold": 1, "sample_seed": 0, "fold_seed": 42, "threads": int(os.environ.get("THREADS", 8)), "p": 50000,
        "variant": variant, "trait": trait, "geno_npz": str(here / "data/rice/genotype.npz"), "pheno_csv": str(here / "data/rice/phenotype.csv")}
t0 = time.time()
X, y, names, snps = load_data(task)
tr, te = select_split(len(y), task)
cols = pick_cols(task, X.shape[1])
if cols is not None:
    X = np.ascontiguousarray(X[:, cols]); cols = None
load_s = time.time() - t0
pred, info = run_topx(task, X, y, tr, te, cols)
pcc = float(np.corrcoef(pred, y[te])[0, 1])
info.update(variant=variant, trait=trait, n_train=int(len(tr)), n_test=int(len(te)), n_snps=int(X.shape[1]), pcc=pcc, load_s=load_s,
            peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2),
            threads=task["threads"])
out.mkdir(parents=True, exist_ok=True)
tmp = res.with_suffix(".tmp"); tmp.write_text(json.dumps(info, indent=1, default=float)); tmp.rename(res)
print(variant, tag, "PCC %.4f" % pcc, "fit %.1fs" % info["fit_s"], "mem %.1fGB" % info["peak_rss_gb"], flush=True)
