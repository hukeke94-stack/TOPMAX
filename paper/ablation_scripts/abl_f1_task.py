#!/usr/bin/env python3
"""One ablation task on the simulated maize F1 panel: 100,000 individuals (sample seed 20260611, as in Fig. 6), fold 1,
50,000 markers, one trait, one variant. Peak memory = maximum RssAnon of this process (polled), i.e. without file pages
of the memory-mapped genotype matrix. Writes <out>/<variant>__<trait>.json; skips if present."""
import json, os, sys, threading, time
from pathlib import Path
import numpy as np
from timed_topx import load_data, select_split, run_topx

variant, trait, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
res = out / f"{variant}__{trait}.json"
if res.exists():
    sys.exit(0)
peak = [0]
def poll():
    while True:
        for line in open("/proc/self/status"):
            if line.startswith("RssAnon:"):
                peak[0] = max(peak[0], int(line.split()[1])); break
        time.sleep(0.2)
threading.Thread(target=poll, daemon=True).start()
task = {"n": 100000, "fold": 1, "sample_seed": 20260611, "fold_seed": 42, "threads": int(os.environ.get("THREADS", 8)),
        "variant": variant, "trait": trait}
t0 = time.time()
X, y, names, snps = load_data(task)
tr, te = select_split(len(y), task)
load_s = time.time() - t0
pred, info = run_topx(task, X, y, tr, te, None)
pcc = float(np.corrcoef(pred, y[te])[0, 1])
time.sleep(0.5)
info.update(variant=variant, trait=trait, n_train=int(len(tr)), n_test=int(len(te)), n_snps=int(X.shape[1]), pcc=pcc, load_s=load_s,
            peak_anon_gb=peak[0] / 1024 ** 2, threads=task["threads"])
out.mkdir(parents=True, exist_ok=True)
tmp = res.with_suffix(".tmp"); tmp.write_text(json.dumps(info, indent=1, default=float)); tmp.rename(res)
print(variant, trait, "PCC %.4f" % pcc, "fit %.1fs" % info["fit_s"], "anon %.1fGB" % info["peak_anon_gb"], flush=True)
