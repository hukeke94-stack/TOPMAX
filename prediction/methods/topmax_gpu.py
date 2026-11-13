#!/usr/bin/env python3
"""
TOPMAX on the GPU.

A separate method class, `TOPMAX_GPU`, with the same interface as the CPU version.
Standardization, randomized SVD, GCV, the warm start, and CG all run on the GPU;
predictions are returned as NumPy arrays.
"""

from __future__ import annotations

import gc
from pathlib import Path
import sys

import numpy as np

from .base import BaseMethod, register_method

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.genotype_cache import DEFAULT_ROW_BATCH_ROWS, iter_row_batches


def _load_cupy():
    try:
        import cupy as cp
    except ImportError as exc:
        raise ImportError(
            "TOPMAX_GPU needs CuPy. Install it with: pip install -r requirements-gpu.txt"
        ) from exc

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        raise RuntimeError(
            "No usable GPU was found. Check the NVIDIA driver, the CUDA runtime, and the CuPy installation."
            " If the CUDA runtime is missing, run: pip install -r requirements-gpu-bundled-cuda.txt"
        ) from exc

    if device_count < 1:
        raise RuntimeError("No GPU is available, so TOPMAX_GPU cannot run.")

    return cp


def _release_gpu_memory(cp):
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def _safe_scale(cp, scale):
    return cp.where(scale > 0, scale, cp.ones_like(scale))


def _standardize_matrix(cp, X_gpu):
    mean = cp.mean(X_gpu, axis=0, dtype=cp.float64).astype(cp.float32)
    var = cp.var(X_gpu, axis=0, dtype=cp.float64).astype(cp.float32)
    scale = _safe_scale(cp, cp.sqrt(var))
    # Standardize in place to avoid a second full matrix allocation on GPU.
    X_gpu -= mean
    X_gpu /= scale
    return X_gpu, mean, scale


def _standardize_vector(cp, y_gpu):
    y_min = cp.min(y_gpu).astype(cp.float32)
    y_max = cp.max(y_gpu).astype(cp.float32)
    span = y_max - y_min
    if float(span.get()) <= 0:
        return cp.zeros_like(y_gpu), y_min, cp.array(0.0, dtype=cp.float32)
    scale = cp.array(2.0, dtype=cp.float32) / span
    y_gpu -= y_min
    y_gpu *= scale
    y_gpu -= cp.array(1.0, dtype=cp.float32)
    return y_gpu, y_min, scale


def _gpu_randomized_svd(cp, X_gpu, n_components, n_iter=5, n_oversamples=10, seed=42):
    n_samples, n_features = X_gpu.shape
    n_random = n_components + n_oversamples
    transpose = n_samples < n_features
    M = X_gpu.T if transpose else X_gpu

    def qr_reduced_gpu(matrix_gpu):
        # Some rented images expose GPUs correctly but ship an unstable cuSOLVER
        # QR path. Keep the expensive matrix products on GPU and use NumPy for
        # the small orthogonalization step so TOPMAX_GPU remains portable.
        q_cpu, _ = np.linalg.qr(cp.asnumpy(matrix_gpu), mode="reduced")
        return cp.asarray(q_cpu.astype(np.float32, copy=False))

    rs = cp.random.RandomState(seed)
    omega = rs.standard_normal((M.shape[1], n_random), dtype=cp.float32)
    Y = M @ omega
    Q = qr_reduced_gpu(Y)

    for _ in range(n_iter):
        Z = M.T @ Q
        Q = qr_reduced_gpu(Z)
        Y = M @ Q
        Q = qr_reduced_gpu(Y)

    B = Q.T @ M
    U_hat_cpu, s_cpu, Vt_cpu = np.linalg.svd(cp.asnumpy(B), full_matrices=False)
    U_hat = cp.asarray(U_hat_cpu.astype(np.float32, copy=False))
    s = cp.asarray(s_cpu.astype(np.float32, copy=False))
    Vt = cp.asarray(Vt_cpu.astype(np.float32, copy=False))
    U = Q @ U_hat

    if transpose:
        return Vt[:n_components, :].T, s[:n_components], U[:, :n_components].T
    return U[:, :n_components], s[:n_components], Vt[:n_components, :]


def _gpu_gcv_search(cp, U, S, y_scaled, alphas):
    n_samples = y_scaled.shape[0]
    S2 = S**2
    c = U.T @ y_scaled
    y_norm_sq = cp.sum(y_scaled**2)
    c_norm_sq = cp.sum(c**2)
    y_perp_norm_sq = y_norm_sq - c_norm_sq

    alpha_gpu = cp.asarray(alphas, dtype=cp.float32)
    denom = S2[:, None] + alpha_gpu[None, :]
    factors = S2[:, None] / denom
    dof = cp.sum(factors, axis=0)
    res_low = c[:, None] * (alpha_gpu[None, :] / denom)
    rss_total = cp.sum(res_low**2, axis=0) + y_perp_norm_sq
    gcv = (rss_total / n_samples) / ((1 - dof / n_samples) ** 2 + 1e-10)
    best_idx = int(cp.argmin(gcv).get())
    return float(alpha_gpu[best_idx].get())


def _gpu_conjugate_grad(cp, matvec, b, tol, max_iter, x0=None):
    residuals = []
    m = int(b.shape[0])

    if x0 is None:
        x = cp.zeros_like(b)
        r = b.copy()
    else:
        x = x0.copy()
        r = b - matvec(x)

    p = r.copy()
    r_k_norm = cp.sum(r * r)
    r_k_norm_val = float(r_k_norm.get())
    r_sqrt_initial = np.sqrt(max(r_k_norm_val, 0.0))
    residuals.append(1.0)

    max_iter_cg = 2 * m
    if max_iter > max_iter_cg:
        max_iter = max_iter_cg

    iterations = max_iter - 1
    for i in range(max_iter):
        Ap = matvec(p)
        p_dot_Ap = cp.sum(p * Ap)
        p_dot_Ap_val = float(p_dot_Ap.get())
        if p_dot_Ap_val == 0.0:
            break

        alpha = r_k_norm / p_dot_Ap
        x = x + alpha * p
        r = r - alpha * Ap

        r_k_plus_1_norm = cp.sum(r * r)
        beta = r_k_plus_1_norm / r_k_norm
        p = r + beta * p
        r_k_norm = r_k_plus_1_norm

        current_res_ratio = float(cp.sqrt(r_k_plus_1_norm).get()) / r_sqrt_initial if r_sqrt_initial > 0 else 0.0
        residuals.append(current_res_ratio)
        if current_res_ratio < tol:
            iterations = i
            break

        iterations = i

    return x, residuals, iterations


@register_method
class TOPMAXGPU(BaseMethod):
    """
    GPU version of TOPMAX.

    Standardization, rSVD, GCV, the warm start, and CG run on the GPU;
    only the coefficients and scaling parameters are copied back to the CPU.
    """

    name = "TOPMAX_GPU"

    def __init__(
        self,
        rsvd_components=500,
        alpha_range=(-2, 7),
        grid_points=100,
        seed=42,
        cg_tol=1e-5,
        cg_max_iter=100,
        gpu_n_oversamples=10,
        gpu_n_iter=5,
        fit_batch_rows=DEFAULT_ROW_BATCH_ROWS,
        predict_batch_rows=DEFAULT_ROW_BATCH_ROWS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rsvd_components = rsvd_components
        self.alpha_range = alpha_range
        self.grid_points = grid_points
        self.seed = seed
        self.cg_tol = cg_tol
        self.cg_max_iter = cg_max_iter
        self.gpu_n_oversamples = gpu_n_oversamples
        self.gpu_n_iter = gpu_n_iter
        self.fit_batch_rows = max(1, int(fit_batch_rows))
        self.predict_batch_rows = None if predict_batch_rows is None else max(1, int(predict_batch_rows))

        self.coef_ = None
        self.cg_iterations = 0
        self._x_mean = None
        self._x_scale = None
        self._y_min = None
        self._y_scale = None
        self.y_intercept_ = 0.0

    def _fit_from_gpu_arrays(self, cp, X_train_gpu, y_train_gpu):
        X_train_scaled_gpu, x_mean_gpu, x_scale_gpu = _standardize_matrix(cp, X_train_gpu)
        y_train_scaled_gpu, y_min_gpu, y_scale_gpu = _standardize_vector(cp, y_train_gpu)
        y_intercept_gpu = cp.mean(y_train_scaled_gpu, dtype=cp.float64).astype(cp.float32)
        y_centered_gpu = y_train_scaled_gpu - y_intercept_gpu

        n_samples, n_features = X_train_scaled_gpu.shape
        n_components = min(self.rsvd_components, min(n_samples, n_features) - 1)
        if n_components < 1:
            raise ValueError("rSVD component count must be positive")

        U_gpu, S_gpu, Vt_gpu = _gpu_randomized_svd(
            cp,
            X_train_scaled_gpu,
            n_components=n_components,
            n_iter=self.gpu_n_iter,
            n_oversamples=self.gpu_n_oversamples,
            seed=self.seed,
        )
        S2_gpu = S_gpu**2

        alphas = np.logspace(self.alpha_range[0], self.alpha_range[1], self.grid_points)
        self.best_alpha = _gpu_gcv_search(cp, U_gpu, S_gpu, y_centered_gpu, alphas)

        c_gpu = U_gpu.T @ y_centered_gpu
        best_alpha_gpu = cp.array(self.best_alpha, dtype=cp.float32)

        if n_features > n_samples:
            del Vt_gpu
            damping_gpu = 1.0 / (S2_gpu + best_alpha_gpu)
            ws_gpu = (
                U_gpu @ (damping_gpu * c_gpu) + (y_centered_gpu - U_gpu @ c_gpu) / best_alpha_gpu
            ).reshape(-1, 1).astype(cp.float32)
            del U_gpu, S_gpu, S2_gpu, c_gpu, damping_gpu
            _release_gpu_memory(cp)

            b_gpu = y_centered_gpu.reshape(-1, 1)

            def matvec(v):
                Xt_v = X_train_scaled_gpu.T @ v
                return X_train_scaled_gpu @ Xt_v + best_alpha_gpu * v

            dual_coef_gpu, _, self.cg_iterations = _gpu_conjugate_grad(
                cp,
                matvec,
                b_gpu,
                self.cg_tol,
                self.cg_max_iter,
                x0=ws_gpu,
            )
            coef_gpu = X_train_scaled_gpu.T @ dual_coef_gpu
            del dual_coef_gpu
        else:
            Xt_y_gpu = X_train_scaled_gpu.T @ y_centered_gpu
            factor_gpu = (S_gpu**3) / (best_alpha_gpu * (S2_gpu + best_alpha_gpu))
            ws_gpu = (Xt_y_gpu / best_alpha_gpu - Vt_gpu.T @ (factor_gpu * c_gpu)).reshape(-1, 1).astype(cp.float32)
            del U_gpu, Vt_gpu, S_gpu, S2_gpu, c_gpu, factor_gpu
            _release_gpu_memory(cp)

            b_gpu = Xt_y_gpu.reshape(-1, 1)

            def matvec(v):
                X_v = X_train_scaled_gpu @ v
                return X_train_scaled_gpu.T @ X_v + best_alpha_gpu * v

            coef_gpu, _, self.cg_iterations = _gpu_conjugate_grad(
                cp,
                matvec,
                b_gpu,
                self.cg_tol,
                self.cg_max_iter,
                x0=ws_gpu,
            )
            del Xt_y_gpu

        del ws_gpu, b_gpu
        self.coef_ = cp.asnumpy(coef_gpu).flatten()
        self._x_mean = cp.asnumpy(x_mean_gpu)
        self._x_scale = cp.asnumpy(x_scale_gpu)
        self._y_min = float(y_min_gpu.get())
        self._y_scale = float(y_scale_gpu.get())
        self.y_intercept_ = float(y_intercept_gpu.get())

        del X_train_gpu, y_train_gpu, X_train_scaled_gpu, y_train_scaled_gpu, y_centered_gpu
        del best_alpha_gpu, coef_gpu
        del x_mean_gpu, x_scale_gpu, y_min_gpu, y_scale_gpu, y_intercept_gpu
        _release_gpu_memory(cp)
        gc.collect()

    def _copy_rows_to_gpu(self, cp, X_source, row_indices, batch_rows=None):
        row_indices = np.asarray(row_indices, dtype=np.int64)
        if row_indices.ndim != 1:
            raise ValueError("row_indices must be a 1D array")

        n_rows = len(row_indices)
        n_features = int(X_source.shape[1])
        X_train_gpu = cp.empty((n_rows, n_features), dtype=cp.float32)
        batch_rows = self.fit_batch_rows if batch_rows is None else max(1, int(batch_rows))

        for start, stop, chunk in iter_row_batches(
            X_source,
            row_indices,
            batch_rows=batch_rows,
            dtype=np.float32,
        ):
            X_train_gpu[start:stop] = cp.asarray(chunk, dtype=cp.float32)
            del chunk

        gc.collect()
        return X_train_gpu

    def fit(self, X_train, y_train):
        cp = _load_cupy()
        X_train_gpu = cp.asarray(X_train, dtype=cp.float32)
        y_train_gpu = cp.asarray(y_train, dtype=cp.float32).ravel()
        self._fit_from_gpu_arrays(cp, X_train_gpu, y_train_gpu)

    def fit_from_rows(self, X_source, row_indices, y_train, batch_rows=None):
        row_indices = np.asarray(row_indices, dtype=np.int64)
        y_train = np.asarray(y_train, dtype=np.float32).ravel()
        if len(row_indices) != len(y_train):
            raise ValueError("row_indices and y_train must have the same length")

        cp = _load_cupy()
        X_train_gpu = self._copy_rows_to_gpu(cp, X_source, row_indices, batch_rows=batch_rows)
        y_train_gpu = cp.asarray(y_train, dtype=cp.float32)
        self._fit_from_gpu_arrays(cp, X_train_gpu, y_train_gpu)

    def _predict_chunk_inplace(self, X_chunk):
        X_chunk -= self._x_mean
        np.divide(X_chunk, self._x_scale, out=X_chunk)
        y_pred_scaled = X_chunk @ self.coef_ + self.y_intercept_
        if self._y_scale <= 0:
            return np.full_like(y_pred_scaled, self._y_min, dtype=np.float32)
        return ((y_pred_scaled + 1.0) / self._y_scale + self._y_min).astype(np.float32, copy=False)

    def predict(self, X_test):
        if self.coef_ is None:
            raise ValueError("Call fit before predict")

        X_test = np.asarray(X_test)
        n_rows = int(X_test.shape[0])
        batch_rows = self.predict_batch_rows
        if batch_rows is None or n_rows <= batch_rows:
            return self._predict_chunk_inplace(np.array(X_test, dtype=np.float32, copy=True))

        y_pred = np.empty(n_rows, dtype=np.float32)
        for start in range(0, n_rows, batch_rows):
            stop = min(start + batch_rows, n_rows)
            chunk = np.array(X_test[start:stop], dtype=np.float32, copy=True)
            y_pred[start:stop] = self._predict_chunk_inplace(chunk)
        gc.collect()
        return y_pred

    def predict_from_rows(self, X_source, row_indices, batch_rows=None):
        if self.coef_ is None:
            raise ValueError("Call fit before predict")

        row_indices = np.asarray(row_indices, dtype=np.int64)
        batch_rows = self.predict_batch_rows if batch_rows is None else max(1, int(batch_rows))
        y_pred = np.empty(len(row_indices), dtype=np.float32)

        for start, stop, chunk in iter_row_batches(
            X_source,
            row_indices,
            batch_rows=batch_rows,
            dtype=np.float32,
        ):
            y_pred[start:stop] = self._predict_chunk_inplace(chunk)

        gc.collect()
        return y_pred
