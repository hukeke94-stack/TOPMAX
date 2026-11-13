#!/usr/bin/env python3
"""TOPMAX genomic prediction.

Cross-validates on the training samples, then fits one model on all training samples and predicts the
prediction samples. Training and prediction samples can come from one dataset (samples listed with
--predict-ids, or samples whose phenotype is missing) or from two datasets (--predict-dir).

Example:
    python topmax_predict.py --data-dir ../example_data/maize_ncii --trait CW \
        --predict-ids ../example_data/maize_ncii/predict_ids.txt --output results
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


def parse_args():
    p = argparse.ArgumentParser(description="TOPMAX genomic prediction",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    io = p.add_argument_group("input and output")
    io.add_argument("--data-dir", required=True, help="Folder with genotype.npz and phenotype.csv")
    io.add_argument("--trait", default="all", help="Trait name, or 'all' for every trait in phenotype.csv")
    io.add_argument("--predict-ids", help="Text file with one sample ID per line; these samples are predicted, not used for training")
    io.add_argument("--predict-dir", help="Second folder with genotype.npz (and optionally phenotype.csv) to predict")
    io.add_argument("--output", default="results", help="Output folder")

    run = p.add_argument_group("run")
    run.add_argument("--folds", type=int, default=5, help="Cross-validation folds on the training samples (0 skips cross-validation)")
    run.add_argument("--seed", type=int, default=42, help="Seed for the fold split and the randomized SVD")
    run.add_argument("--threads", type=int, default=8, help="CPU threads for linear algebra")
    run.add_argument("--gpu", action="store_true", help="Run on an NVIDIA GPU (needs CuPy)")

    model = p.add_argument_group("model (defaults are recommended)")
    model.add_argument("--rsvd-components", type=int, default=500, help="Rank of the randomized SVD used for the GCV search and the warm start")
    model.add_argument("--alpha-min", type=float, default=-2, help="log10 of the smallest ridge penalty in the GCV grid")
    model.add_argument("--alpha-max", type=float, default=7, help="log10 of the largest ridge penalty in the GCV grid")
    model.add_argument("--alpha-points", type=int, default=100, help="Number of penalties in the GCV grid")
    model.add_argument("--cg-tol", type=float, default=1e-5, help="Relative residual at which conjugate gradient stops")
    model.add_argument("--cg-max-iter", type=int, default=None, help="Maximum conjugate gradient iterations (default 500 on CPU, 100 on GPU)")
    return p.parse_args()


def set_thread_limits(threads):
    for name in THREAD_ENV_VARS:
        os.environ[name] = str(max(1, int(threads)))


def read_genotypes(folder):
    import numpy as np
    path = Path(folder) / "genotype.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    with np.load(path, allow_pickle=False) as g:
        return np.asarray(g["X"], dtype=np.float32), g["sample_names"].astype(str), g["snp_names"].astype(str)


def read_phenotypes(folder, names):
    import pandas as pd
    path = Path(folder) / "phenotype.csv"
    if not path.is_file():
        return None
    pheno = pd.read_csv(path, index_col=0)
    pheno.index = pheno.index.astype(str)
    return pheno.reindex(names)


def load(args):
    """Return training and prediction arrays: X, ids, phenotypes for each set (prediction parts may be None)."""
    import numpy as np
    X, names, snps = read_genotypes(args.data_dir)
    pheno = read_phenotypes(args.data_dir, names)
    if pheno is None:
        raise FileNotFoundError(f"Missing {Path(args.data_dir) / 'phenotype.csv'}")

    if args.predict_dir:
        X_pred, pred_names, pred_snps = read_genotypes(args.predict_dir)
        if not np.array_equal(snps, pred_snps):
            raise ValueError("Training and prediction genotypes must have the same markers in the same order")
        return X, names, pheno, X_pred, pred_names, read_phenotypes(args.predict_dir, pred_names)

    if args.predict_ids:
        wanted = {line.strip() for line in open(args.predict_ids, encoding="utf-8") if line.strip()}
        missing = wanted - set(names)
        if missing:
            raise ValueError(f"{len(missing)} IDs in {args.predict_ids} are not in genotype.npz, e.g. {sorted(missing)[:3]}")
        is_pred = np.isin(names, list(wanted))
        return X[~is_pred], names[~is_pred], pheno[~is_pred], X[is_pred], names[is_pred], pheno[is_pred]

    return X, names, pheno, None, None, None


def make_model(args):
    from methods import get_method
    kwargs = dict(rsvd_components=args.rsvd_components, alpha_range=(args.alpha_min, args.alpha_max),
                  grid_points=args.alpha_points, seed=args.seed, cg_tol=args.cg_tol)
    if args.cg_max_iter is not None:
        kwargs["cg_max_iter"] = args.cg_max_iter
    return get_method("TOPMAX_GPU" if args.gpu else "TOPMAX", **kwargs)


def run_trait(trait, data, args):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold
    from utils.metrics import calculate_pcc, calculate_r2

    X, names, pheno, X_pred, pred_names, pred_pheno = data
    y_all = pd.to_numeric(pheno[trait], errors="coerce").to_numpy(dtype=np.float32)
    has_y = np.isfinite(y_all)
    X_tr, y, ids = X[has_y], y_all[has_y], names[has_y]
    folds, cv_rows = [], []

    if args.folds > 1:
        for fold, (tr, te) in enumerate(KFold(args.folds, shuffle=True, random_state=args.seed).split(y), start=1):
            model = make_model(args)
            y_hat, seconds = model.fit_predict(X_tr[tr], y[tr], X_tr[te])
            pcc, r2 = calculate_pcc(y[te], y_hat), calculate_r2(y[te], y_hat)
            folds.append({"Trait": trait, "Fold": fold, "TrainSize": len(tr), "TestSize": len(te), "PCC": pcc, "R2": r2,
                          "Alpha": model.best_alpha, "CGIterations": model.cg_iterations, "Seconds": seconds})
            cv_rows += [{"Trait": trait, "Fold": fold, "ID": i, "Observed": float(o), "Predicted": float(p)}
                        for i, o, p in zip(ids[te], y[te], y_hat)]
            print(f"  fold {fold}: PCC = {pcc:.4f}, R2 = {r2:.4f}, {seconds:.1f} s", flush=True)

    # Prediction samples: those given by --predict-ids/--predict-dir, plus training samples without a phenotype
    targets = [(X[~has_y], names[~has_y], None)] if (~has_y).any() else []
    if X_pred is not None:
        truth = None
        if pred_pheno is not None and trait in pred_pheno.columns:
            truth = pd.to_numeric(pred_pheno[trait], errors="coerce").to_numpy(dtype=float)
        targets.append((X_pred, pred_names, truth))
    pred_rows, pred_pcc = [], None
    if targets:
        model = make_model(args)
        model.fit(X_tr, y)
        for Xp, idp, truth in targets:
            y_hat = np.asarray(model.predict(Xp), dtype=float)
            for k, i in enumerate(idp):
                obs = float(truth[k]) if truth is not None and np.isfinite(truth[k]) else None
                pred_rows.append({"Trait": trait, "ID": i, "Predicted": y_hat[k], "Observed": obs})
            if truth is not None and np.isfinite(truth).sum() >= 3:
                ok = np.isfinite(truth)
                pred_pcc = calculate_pcc(truth[ok], y_hat[ok])
        print(f"  predicted {len(pred_rows)} samples" + (f", PCC against the observed values = {pred_pcc:.4f}" if pred_pcc is not None else ""), flush=True)
    return folds, cv_rows, pred_rows, pred_pcc


def main():
    args = parse_args()
    set_thread_limits(args.threads)  # must precede the NumPy import
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pandas as pd

    data = load(args)
    traits = list(data[2].columns) if args.trait == "all" else [args.trait]
    unknown = [t for t in traits if t not in data[2].columns]
    if unknown:
        raise SystemExit(f"Trait(s) {unknown} not in phenotype.csv. Available: {list(data[2].columns)}")
    n_pred = 0 if data[3] is None else len(data[3])
    print(f"Training set: {len(data[1])} samples x {data[0].shape[1]} markers; prediction set: {n_pred} samples")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    all_folds, all_cv, all_pred, summary = [], [], [], []
    start = time.time()
    for trait in traits:
        print(f"{trait}:", flush=True)
        folds, cv_rows, pred_rows, pred_pcc = run_trait(trait, data, args)
        all_folds += folds
        all_cv += cv_rows
        all_pred += pred_rows
        f = pd.DataFrame(folds)
        summary.append({"Trait": trait, "CV_PCC_mean": f.PCC.mean() if len(f) else None, "CV_PCC_sd": f.PCC.std() if len(f) else None,
                        "Prediction_PCC": pred_pcc, "Predicted_samples": len(pred_rows)})

    if all_folds:
        pd.DataFrame(all_folds).to_csv(out / "cv_folds.csv", index=False)
        pd.DataFrame(all_cv).to_csv(out / "cv_predictions.csv", index=False)
    if all_pred:
        pd.DataFrame(all_pred).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    print(f"Done in {time.time() - start:.1f} s. Results written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
