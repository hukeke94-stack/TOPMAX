#!/usr/bin/env python3
"""TOPMAX CPU implementation."""

from __future__ import annotations

import gc

import numpy as np
from sklearn.utils.extmath import randomized_svd

from .base import BaseMethod, register_method
from utils.genotype_cache import DEFAULT_ROW_BATCH_ROWS, iter_row_batches


DEFAULT_CONFIG = {
    "rsvd_components": 500,
    "alpha_range": (-2, 7),
    "grid_points": 100,
    "seed": 42,
    "cg_tol": 1e-5,
    "cg_max_iter": 500,
}


def _safe_feature_scale(scale):
    scale = np.asarray(scale, dtype=np.float32)
    scale[scale <= 0] = 1.0
    return scale


def _fit_target_scaler(y):
    y = np.asarray(y, dtype=np.float32).ravel()
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    span = y_max - y_min
    if span <= 0:
        return np.zeros_like(y, dtype=np.float32), y_min, 0.0
    y_scale = float(2.0 / span)
    y_scaled = ((y - y_min) * y_scale - 1.0).astype(np.float32, copy=False)
    return y_scaled, y_min, y_scale


def _inverse_target_scaler(y_scaled, y_min, y_scale):
    y_scaled = np.asarray(y_scaled, dtype=np.float32)
    if y_scale <= 0:
        return np.full_like(y_scaled, y_min, dtype=np.float32)
    return ((y_scaled + 1.0) / y_scale + y_min).astype(np.float32, copy=False)


def _gcv_search(U, S, y_centered, alphas):
    n_samples = len(y_centered)
    S2 = S**2
    c = U.T @ y_centered
    y_norm_sq = np.sum(y_centered**2)
    c_norm_sq = np.sum(c**2)
    y_perp_norm_sq = y_norm_sq - c_norm_sq

    best_gcv = float("inf")
    best_alpha = float(alphas[0])
    for alpha in alphas:
        factors = S2 / (S2 + alpha)
        dof = np.sum(factors)
        res_low = c * (alpha / (S2 + alpha))
        rss_total = np.sum(res_low**2) + y_perp_norm_sq
        gcv = (rss_total / n_samples) / ((1.0 - dof / n_samples) ** 2 + 1e-10)
        if gcv < best_gcv:
            best_gcv = float(gcv)
            best_alpha = float(alpha)
    return best_alpha


def _conjugate_gradient(matvec, b, tol, max_iter, x0=None):
    b = np.asarray(b, dtype=np.float32)
    x = np.zeros_like(b) if x0 is None else np.asarray(x0, dtype=np.float32).copy()
    r = b.copy() if x0 is None else b - matvec(x)
    p = r.copy()
    r_norm = float(np.sum(r * r))
    initial = np.sqrt(max(r_norm, 0.0))
    iterations = 0

    if initial == 0:
        return x, iterations

    max_iter = min(int(max_iter), 2 * int(b.shape[0]))
    for i in range(max_iter):
        Ap = matvec(p)
        denom = float(np.sum(p * Ap))
        if denom == 0:
            break
        alpha = r_norm / denom
        x += alpha * p
        r -= alpha * Ap
        next_r_norm = float(np.sum(r * r))
        iterations = i + 1
        if np.sqrt(max(next_r_norm, 0.0)) / initial < tol:
            break
        beta = next_r_norm / r_norm
        p = r + beta * p
        r_norm = next_r_norm

    return x, iterations


@register_method
class TOPMAX(BaseMethod):
    """TOPMAX CPU version: randomized SVD for alpha search plus warm-start CG."""

    name = "TOPMAX"

    def __init__(
        self,
        rsvd_components=None,
        alpha_range=(-2, 7),
        grid_points=100,
        seed=42,
        cg_tol=1e-5,
        cg_max_iter=500,
        fit_batch_rows=DEFAULT_ROW_BATCH_ROWS,
        predict_batch_rows=DEFAULT_ROW_BATCH_ROWS,
        stats_batch_rows=256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rsvd_components = rsvd_components or DEFAULT_CONFIG["rsvd_components"]
        self.alpha_range = alpha_range
        self.grid_points = grid_points
        self.seed = seed
        self.cg_tol = cg_tol
        self.cg_max_iter = cg_max_iter
        self.fit_batch_rows = max(1, int(fit_batch_rows))
        self.predict_batch_rows = max(1, int(predict_batch_rows))
        self.stats_batch_rows = max(1, int(stats_batch_rows))

        self.coef_ = None
        self.best_alpha = None
        self.cg_iterations = 0
        self.y_intercept_ = 0.0
        self._x_mean = None
        self._x_scale = None
        self._y_min = None
        self._y_scale = None

    def _fit_from_scaled_matrix(self, X_train_scaled, y_train_scaled):
        self.y_intercept_ = float(np.mean(y_train_scaled, dtype=np.float64))
        y_centered = y_train_scaled - self.y_intercept_

        n_samples, n_features = X_train_scaled.shape
        n_components = min(self.rsvd_components, min(n_samples, n_features) - 1)
        if n_components < 1:
            raise ValueError("rSVD component count must be positive")

        U, S, Vt = randomized_svd(
            X_train_scaled,
            n_components=n_components,
            n_iter=5,
            random_state=self.seed,
        )
        S2 = S**2
        alphas = np.logspace(self.alpha_range[0], self.alpha_range[1], self.grid_points)
        self.best_alpha = _gcv_search(U, S, y_centered, alphas)

        c = U.T @ y_centered
        if n_features > n_samples:
            damping = 1.0 / (S2 + self.best_alpha)
            warm_start = (
                U @ (damping * c) + (y_centered - U @ c) / self.best_alpha
            ).reshape(-1, 1).astype(np.float32)

            def matvec(v):
                v = v.reshape(-1, 1)
                return X_train_scaled @ (X_train_scaled.T @ v) + self.best_alpha * v

            solution, self.cg_iterations = _conjugate_gradient(
                matvec,
                y_centered.reshape(-1, 1),
                self.cg_tol,
                self.cg_max_iter,
                x0=warm_start,
            )
            self.coef_ = (X_train_scaled.T @ solution).flatten()
        else:
            Xt_y = X_train_scaled.T @ y_centered
            factor = (S**3) / (self.best_alpha * (S2 + self.best_alpha))
            warm_start = (
                Xt_y / self.best_alpha - Vt.T @ (factor * c)
            ).reshape(-1, 1).astype(np.float32)

            def matvec(v):
                v = v.reshape(-1, 1)
                return X_train_scaled.T @ (X_train_scaled @ v) + self.best_alpha * v

            solution, self.cg_iterations = _conjugate_gradient(
                matvec,
                Xt_y.reshape(-1, 1),
                self.cg_tol,
                self.cg_max_iter,
                x0=warm_start,
            )
            self.coef_ = solution.flatten()

        del U, S, Vt, y_centered
        gc.collect()

    def fit(self, X_train, y_train):
        X_train_scaled = np.array(X_train, dtype=np.float32, copy=True)
        self._x_mean = np.mean(X_train_scaled, axis=0, dtype=np.float64).astype(np.float32)
        self._x_scale = _safe_feature_scale(
            np.std(X_train_scaled, axis=0, dtype=np.float64).astype(np.float32)
        )
        X_train_scaled -= self._x_mean
        np.divide(X_train_scaled, self._x_scale, out=X_train_scaled)

        y_train_scaled, self._y_min, self._y_scale = _fit_target_scaler(y_train)
        self._fit_from_scaled_matrix(X_train_scaled, y_train_scaled)

    def _compute_feature_stats_from_rows(self, X_source, row_indices, batch_rows):
        n_features = int(X_source.shape[1])
        sum_ = np.zeros(n_features, dtype=np.float64)
        sq_sum = np.zeros(n_features, dtype=np.float64)
        total_rows = 0

        for _, _, chunk in iter_row_batches(
            X_source,
            row_indices,
            batch_rows=batch_rows,
            dtype=np.float32,
        ):
            chunk64 = np.asarray(chunk, dtype=np.float64)
            sum_ += chunk64.sum(axis=0, dtype=np.float64)
            sq_sum += np.einsum("ij,ij->j", chunk64, chunk64, optimize=True)
            total_rows += int(chunk64.shape[0])

        mean = (sum_ / total_rows).astype(np.float32)
        mean64 = mean.astype(np.float64, copy=False)
        var = np.maximum((sq_sum / total_rows) - (mean64 * mean64), 0.0).astype(np.float32)
        return mean, _safe_feature_scale(np.sqrt(var).astype(np.float32, copy=False))

    def _build_scaled_training_matrix(self, X_source, row_indices, batch_rows):
        X_train_scaled = np.empty((len(row_indices), int(X_source.shape[1])), dtype=np.float32)
        for start, stop, chunk in iter_row_batches(
            X_source,
            row_indices,
            batch_rows=batch_rows,
            dtype=np.float32,
        ):
            chunk = np.array(chunk, dtype=np.float32, copy=False)
            chunk -= self._x_mean
            np.divide(chunk, self._x_scale, out=chunk)
            X_train_scaled[start:stop] = chunk
        return X_train_scaled

    def fit_from_rows(self, X_source, row_indices, y_train, batch_rows=None):
        row_indices = np.asarray(row_indices, dtype=np.int64)
        y_train = np.asarray(y_train, dtype=np.float32).ravel()
        if len(row_indices) != len(y_train):
            raise ValueError("row_indices and y_train must have the same length")

        batch_rows = self.fit_batch_rows if batch_rows is None else max(1, int(batch_rows))
        self._x_mean, self._x_scale = self._compute_feature_stats_from_rows(
            X_source,
            row_indices,
            batch_rows=min(batch_rows, self.stats_batch_rows),
        )
        y_train_scaled, self._y_min, self._y_scale = _fit_target_scaler(y_train)
        X_train_scaled = self._build_scaled_training_matrix(X_source, row_indices, batch_rows)
        self._fit_from_scaled_matrix(X_train_scaled, y_train_scaled)

    def _predict_chunk_inplace(self, X_chunk):
        X_chunk = np.array(X_chunk, dtype=np.float32, copy=False)
        X_chunk -= self._x_mean
        np.divide(X_chunk, self._x_scale, out=X_chunk)
        y_pred_scaled = X_chunk @ self.coef_ + self.y_intercept_
        return _inverse_target_scaler(y_pred_scaled, self._y_min, self._y_scale)

    def predict(self, X_test):
        if self.coef_ is None:
            raise RuntimeError("TOPMAX.predict called before fit")

        X_test = np.asarray(X_test)
        n_rows = int(X_test.shape[0])
        if n_rows <= self.predict_batch_rows:
            return self._predict_chunk_inplace(np.array(X_test, dtype=np.float32, copy=True))

        y_pred = np.empty(n_rows, dtype=np.float32)
        for start in range(0, n_rows, self.predict_batch_rows):
            stop = min(start + self.predict_batch_rows, n_rows)
            chunk = np.array(X_test[start:stop], dtype=np.float32, copy=True)
            y_pred[start:stop] = self._predict_chunk_inplace(chunk)
        gc.collect()
        return y_pred

    def predict_from_rows(self, X_source, row_indices, batch_rows=None):
        if self.coef_ is None:
            raise RuntimeError("TOPMAX.predict_from_rows called before fit")

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
