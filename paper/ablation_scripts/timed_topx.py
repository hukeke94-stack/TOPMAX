"""TimedTOPX with stage timers and ablation variants, extracted from the benchmark runner (run_task.py)."""
import sys, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.linalg as sla
from sklearn.model_selection import KFold
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parent / "topx_release"))
from methods.topx_cpu import TOPX, _conjugate_gradient, _gcv_search  # noqa: E402

F1_DIR = Path("/root/autodl-tmp/f1_benchmark_dataset_504520")

class ColView:
    """Column subset of a memmap genotype matrix, used by the marker-density test."""

    def __init__(self, X, cols):
        self.X, self.cols = X, cols
        self.shape = (X.shape[0], len(cols))

    def __getitem__(self, idx):
        return np.asarray(self.X[idx])[:, self.cols]


class TimedTOPX(TOPX):
    """TOPX with stage timers and ablation switches. variant='full' is the released algorithm."""

    def __init__(self, variant="full", **kw):
        super().__init__(**kw)
        self.variant = variant
        self.stages = {}
        self.extra = {}

    def _fit_from_scaled_matrix(self, X, y_scaled):
        tick = time.perf_counter
        self.y_intercept_ = float(np.mean(y_scaled, dtype=np.float64))
        y = y_scaled - self.y_intercept_
        n, p = X.shape
        k = min(self.rsvd_components, min(n, p) - 1)
        alphas = np.logspace(self.alpha_range[0], self.alpha_range[1], self.grid_points)

        # Stepwise ablation (2026-11-16): svd -> rsvd -> rsvd_cg_explicit -> cold_cg (matrix-free products, EC) -> full (warm start)
        if self.variant == "svd":
            # exact spectrum of the n x n kernel, GCV with n - 1 (centring) in float64, closed-form solution
            t = tick()
            K = (X @ X.T).astype(np.float64)
            self.stages["grm_build"] = tick() - t
            self.extra["grm_gb"] = K.nbytes / 2**30
            t = tick()
            w, U = np.linalg.eigh(K)
            del K
            self.stages["eigh"] = tick() - t
            order = np.argsort(w)[::-1]
            w, U = w[order], U[:, order]
            keep = w > max(w[0], 0) * 1e-10
            w, U = w[keep], U[:, keep]
            t = tick()
            y64 = y.astype(np.float64)
            c = U.T @ y64
            yp = max(float(y64 @ y64 - c @ c), 0.0)
            ne = n - 1
            gcv = [((((c * a / (w + a)) ** 2).sum() + yp) / ne) / ((1.0 - (w / (w + a)).sum() / ne) ** 2 + 1e-300) for a in alphas]
            self.best_alpha = float(alphas[int(np.argmin(gcv))])
            self.stages["gcv"] = tick() - t
            t = tick()
            z = U @ (c / (w + self.best_alpha))
            self.coef_ = (X.T @ z.astype(np.float32)).ravel()
            self.stages["solve"] = tick() - t
            self.cg_iterations = 0
            del U, w
            return

        if self.variant == "rsvd_cg_explicit":
            # RSVD + GCV, then CG on the explicitly formed kernel (X X^T or X^T X), cold start
            t = tick()
            U, S, Vt = randomized_svd(X, n_components=k, n_iter=5, random_state=self.seed)
            self.stages["rsvd"] = tick() - t
            t = tick()
            self.best_alpha = _gcv_search(U, S, y, alphas)
            self.stages["gcv"] = tick() - t
            del U, S, Vt
            t = tick()
            K = X @ X.T if p > n else X.T @ X
            self.stages["grm_build"] = tick() - t
            self.extra["grm_gb"] = K.nbytes / 2**30
            b = (y if p > n else X.T @ y).reshape(-1, 1).astype(np.float32)

            def matvec(v):
                return K @ v.reshape(-1, 1) + self.best_alpha * v.reshape(-1, 1)

            t = tick()
            sol, self.cg_iterations = _conjugate_gradient(matvec, b, self.cg_tol, self.cg_max_iter, x0=None)
            self.stages["cg"] = tick() - t
            self.coef_ = (X.T @ sol).flatten() if p > n else sol.flatten()
            del K
            return

        if self.variant in ("explicit_grm", "exact_gcv"):
            t = tick()
            K = X @ X.T
            self.stages["grm_build"] = tick() - t
            self.extra["grm_gb"] = K.nbytes / 2**30
            if self.variant == "exact_gcv":
                t = tick()
                w, U = np.linalg.eigh(K.astype(np.float64))
                self.stages["eigh"] = tick() - t
                w = np.clip(w, 0, None)
                t = tick()
                self.best_alpha = _gcv_search(U, np.sqrt(w), y.astype(np.float64), alphas)
                self.stages["gcv"] = tick() - t
                del w, U
            else:
                t = tick()
                U, S, _ = randomized_svd(X, n_components=k, n_iter=5, random_state=self.seed)
                self.stages["rsvd"] = tick() - t
                t = tick()
                self.best_alpha = _gcv_search(U, S, y, alphas)
                self.stages["gcv"] = tick() - t
                del U, S
            t = tick()
            K[np.diag_indices_from(K)] += self.best_alpha
            cf = sla.cho_factor(K, lower=True, overwrite_a=True, check_finite=False)
            z = sla.cho_solve(cf, y, check_finite=False)
            self.stages["cholesky_solve"] = tick() - t
            self.coef_ = (X.T @ z).ravel()
            self.cg_iterations = 0
            return

        t = tick()
        U, S, Vt = randomized_svd(X, n_components=k, n_iter=5, random_state=self.seed)
        self.stages["rsvd"] = tick() - t
        S2 = S**2
        t = tick()
        self.best_alpha = _gcv_search(U, S, y, alphas)
        self.stages["gcv"] = tick() - t
        c = U.T @ y

        if self.variant == "rsvd_only":
            self.coef_ = (Vt.T @ ((S / (S2 + self.best_alpha)) * c)).astype(np.float32)
            self.cg_iterations = 0
            return

        t = tick()
        if p > n:
            x0 = (U @ ((1.0 / (S2 + self.best_alpha)) * c) + (y - U @ c) / self.best_alpha).reshape(-1, 1).astype(np.float32)

            def matvec(v):
                v = v.reshape(-1, 1)
                return X @ (X.T @ v) + self.best_alpha * v

            b = y.reshape(-1, 1)
        else:
            Xty = X.T @ y
            f = (S**3) / (self.best_alpha * (S2 + self.best_alpha))
            x0 = (Xty / self.best_alpha - Vt.T @ (f * c)).reshape(-1, 1).astype(np.float32)

            def matvec(v):
                v = v.reshape(-1, 1)
                return X.T @ (X @ v) + self.best_alpha * v

            b = Xty.reshape(-1, 1)
        self.stages["warm_start"] = tick() - t

        t = tick()
        sol, self.cg_iterations = _conjugate_gradient(
            matvec, b, self.cg_tol, self.cg_max_iter, x0=None if self.variant == "cold_cg" else x0
        )
        self.stages["cg"] = tick() - t
        self.coef_ = (X.T @ sol).flatten() if p > n else sol.flatten()
        del U, S, Vt
        gc.collect()


def select_split(total, task):
    if task["n"] <= 0 or task["n"] >= total:
        sel = np.arange(total, dtype=np.int64)
    else:
        rng = np.random.default_rng(task["sample_seed"])
        sel = np.sort(rng.choice(total, size=task["n"], replace=False).astype(np.int64))
    kf = KFold(n_splits=5, shuffle=True, random_state=task["fold_seed"])
    for i, (tr, te) in enumerate(kf.split(np.arange(len(sel))), start=1):
        if i == task["fold"]:
            return sel[tr], sel[te]
    raise RuntimeError("fold not found")


def load_data(task):
    if task.get("geno_npz"):
        z = np.load(task["geno_npz"], allow_pickle=True)
        X, names, snps = z["X"], z["sample_names"].astype(str), z["snp_names"].astype(str)
        ph = pd.read_csv(task["pheno_csv"])
        lookup = dict(zip(ph[ph.columns[0]].astype(str), pd.to_numeric(ph[task["trait"]], errors="coerce")))
        y = np.array([lookup.get(s, np.nan) for s in names], dtype=np.float64)
        keep = ~np.isnan(y)
        return X[keep], y[keep], names[keep], snps
    d = Path(task.get("dataset_dir", F1_DIR))
    names = np.load(d / "genotype.sample_names.npy", allow_pickle=True).astype(str)
    snps = np.load(d / "genotype.snp_names.npy", allow_pickle=True).astype(str)
    X = np.load(d / "genotype.X.npy", mmap_mode="r")
    y = pd.read_csv(d / "phenotype.csv", usecols=[task["trait"]])[task["trait"]].to_numpy(dtype=np.float64)
    if len(y) != X.shape[0]:
        raise RuntimeError(f"phenotype rows {len(y)} != genotype rows {X.shape[0]}")
    return X, y, names, snps


def pick_cols(task, p_total):
    p = task.get("p")
    if not p or p >= p_total:
        return None
    return np.sort(np.random.default_rng(42).choice(p_total, size=p, replace=False))


def run_topx(task, X, y, tr, te, cols):
    src = ColView(X, cols) if cols is not None else X
    m = TimedTOPX(
        variant=task.get("variant", "full"),
        rsvd_components=task.get("rsvd_k", 500),
        seed=task.get("rsvd_seed", 42),
        cg_tol=task.get("cg_tol", 1e-5),
        cg_max_iter=task.get("cg_max_iter", 500),
    )
    with threadpool_limits(limits=task["threads"]):
        t = time.perf_counter()
        m.fit_from_rows(src, tr, y[tr].astype(np.float32))
        fit = time.perf_counter() - t
        t = time.perf_counter()
        pred = m.predict_from_rows(src, te)
        predict = time.perf_counter() - t
    return pred, {
        "fit_s": fit,
        "predict_s": predict,
        "stages": m.stages,
        "best_alpha": m.best_alpha,
        "cg_iterations": int(m.cg_iterations),
        **m.extra,
    }


