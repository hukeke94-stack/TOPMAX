#!/usr/bin/env python3
"""
Utilities for memory-aware genotype loading.

This module converts compressed ``genotype.npz`` archives into ``.npy`` caches
that can be opened with ``mmap_mode='r'``. It also provides row-batch helpers
so callers can avoid materializing the whole sample subset in RAM.
"""

from __future__ import annotations

from pathlib import Path
import os
import zipfile

import numpy as np
from numpy.lib import format as npformat


DEFAULT_CACHE_CHUNK_ROWS = 256
DEFAULT_ROW_BATCH_ROWS = 1024


def _dataset_dir(dataset_dir: str | os.PathLike[str]) -> Path:
    return Path(dataset_dir).expanduser().resolve()


def _cache_paths(
    dataset_dir: str | os.PathLike[str],
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    max_rows: int | None = None,
) -> dict[str, Path]:
    dataset_path = _dataset_dir(dataset_dir)
    cache_root = dataset_path if cache_dir is None else _dataset_dir(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    suffix = "" if max_rows is None else f".head{int(max_rows)}"
    return {
        "dataset_dir": dataset_path,
        "cache_dir": cache_root,
        "npz": dataset_path / "genotype.npz",
        "x_cache": cache_root / f"genotype.X{suffix}.npy",
        "sample_cache": cache_root / f"genotype.sample_names{suffix}.npy",
        "snp_cache": cache_root / "genotype.snp_names.npy",
    }


def _read_npy_header(stream) -> tuple[tuple[int, ...], bool, np.dtype]:
    version = npformat.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = npformat.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran_order, dtype = npformat.read_array_header_2_0(stream)
    elif version == (3, 0):
        shape, fortran_order, dtype = npformat.read_array_header_3_0(stream)
    else:
        raise ValueError(f"Unsupported NPY header version: {version}")
    return shape, fortran_order, np.dtype(dtype)


def _stream_npz_member_to_npy(
    npz_path: Path,
    member_name: str,
    out_path: Path,
    chunk_rows: int,
    *,
    max_rows: int | None = None,
) -> None:
    with zipfile.ZipFile(npz_path, "r") as zf:
        if member_name not in zf.namelist():
            raise KeyError(f"Member {member_name!r} not found in {npz_path}")

        with zf.open(member_name, "r") as src:
            shape, fortran_order, dtype = _read_npy_header(src)
            if len(shape) == 0:
                raise ValueError(f"Expected array member {member_name!r} to be at least 1D")

            out_rows = shape[0] if max_rows is None else min(int(max_rows), int(shape[0]))
            out_shape = (out_rows, *shape[1:])
            out = npformat.open_memmap(out_path, mode="w+", dtype=dtype, shape=out_shape, fortran_order=False)

            if not fortran_order or len(shape) <= 1:
                row_items = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
                row_bytes = dtype.itemsize * row_items

                for start in range(0, out_rows, chunk_rows):
                    stop = min(start + chunk_rows, out_rows)
                    n_rows = stop - start
                    expected_bytes = row_bytes * n_rows
                    raw = src.read(expected_bytes)
                    if len(raw) != expected_bytes:
                        raise ValueError(
                            f"Unexpected EOF while reading {member_name!r}: "
                            f"expected {expected_bytes} bytes, got {len(raw)}"
                        )
                    chunk = np.frombuffer(raw, dtype=dtype, count=n_rows * row_items)
                    out[start:stop] = chunk.reshape((n_rows, *shape[1:]))
            else:
                if len(shape) != 2:
                    raise ValueError(
                        f"Fortran-order array {member_name!r} with shape {shape} is only supported for 2D streaming cache build"
                    )

                total_rows = int(shape[0])
                total_cols = int(shape[1])
                row_bytes = dtype.itemsize * total_cols
                target_chunk_bytes = max(dtype.itemsize * total_rows, row_bytes * max(1, int(chunk_rows)))
                col_chunk = max(1, min(total_cols, target_chunk_bytes // (dtype.itemsize * total_rows)))

                for col_start in range(0, total_cols, col_chunk):
                    col_stop = min(col_start + col_chunk, total_cols)
                    n_cols = col_stop - col_start
                    expected_bytes = dtype.itemsize * total_rows * n_cols
                    raw = src.read(expected_bytes)
                    if len(raw) != expected_bytes:
                        raise ValueError(
                            f"Unexpected EOF while reading {member_name!r}: "
                            f"expected {expected_bytes} bytes, got {len(raw)}"
                        )
                    chunk = np.frombuffer(raw, dtype=dtype, count=total_rows * n_cols)
                    block = chunk.reshape((total_rows, n_cols), order="F")
                    out[:, col_start:col_stop] = block[:out_rows]

            out.flush()


def _save_array_atomic(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as fh:
        np.save(fh, array, allow_pickle=True)
    tmp_path.replace(path)


def ensure_genotype_cache(
    dataset_dir: str | os.PathLike[str],
    *,
    force: bool = False,
    chunk_rows: int = DEFAULT_CACHE_CHUNK_ROWS,
    cache_dir: str | os.PathLike[str] | None = None,
    max_rows: int | None = None,
) -> dict[str, Path]:
    """
    Ensure ``genotype.npz`` has a memory-mappable ``.npy`` cache next to it.
    """
    paths = _cache_paths(dataset_dir, cache_dir=cache_dir, max_rows=max_rows)
    npz_path = paths["npz"]

    if not npz_path.exists():
        raise FileNotFoundError(f"Genotype archive not found: {npz_path}")

    cache_paths = [paths["x_cache"], paths["sample_cache"], paths["snp_cache"]]
    cache_ready = all(path.exists() for path in cache_paths)
    if cache_ready and not force:
        npz_mtime = npz_path.stat().st_mtime
        cache_ready = all(path.stat().st_mtime >= npz_mtime for path in cache_paths)
        if cache_ready:
            return paths

    x_tmp = paths["x_cache"].with_suffix(paths["x_cache"].suffix + ".tmp")
    _stream_npz_member_to_npy(
        npz_path,
        "X.npy",
        x_tmp,
        max(1, int(chunk_rows)),
        max_rows=max_rows,
    )
    x_tmp.replace(paths["x_cache"])

    with np.load(npz_path, allow_pickle=True) as geno:
        sample_names = geno["sample_names"]
        if max_rows is not None:
            sample_names = sample_names[: int(max_rows)]
        _save_array_atomic(paths["sample_cache"], sample_names)
        _save_array_atomic(paths["snp_cache"], geno["snp_names"])

    return paths


def open_genotype_memmap(
    dataset_dir: str | os.PathLike[str],
    *,
    ensure_cache: bool = True,
    chunk_rows: int = DEFAULT_CACHE_CHUNK_ROWS,
    mmap_mode: str = "r",
    cache_dir: str | os.PathLike[str] | None = None,
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Open genotype matrix as a memory map when cache is available.
    """
    paths = _cache_paths(dataset_dir, cache_dir=cache_dir, max_rows=max_rows)

    if ensure_cache:
        ensure_genotype_cache(
            dataset_dir,
            chunk_rows=chunk_rows,
            cache_dir=cache_dir,
            max_rows=max_rows,
        )

    if paths["x_cache"].exists():
        X = np.load(paths["x_cache"], mmap_mode=mmap_mode)
        sample_names = np.load(paths["sample_cache"], allow_pickle=True)
        snp_names = np.load(paths["snp_cache"], allow_pickle=True)
        return X, sample_names, snp_names, True

    with np.load(paths["npz"], allow_pickle=True) as geno:
        X = geno["X"]
        sample_names = geno["sample_names"]
        snp_names = geno["snp_names"]
        if max_rows is not None:
            X = X[: int(max_rows)]
            sample_names = sample_names[: int(max_rows)]
    return X, sample_names, snp_names, False


def iter_index_chunks(row_indices: np.ndarray, batch_rows: int = DEFAULT_ROW_BATCH_ROWS):
    row_indices = np.asarray(row_indices, dtype=np.int64)
    if row_indices.ndim != 1:
        raise ValueError("row_indices must be a 1D array")

    batch_rows = max(1, int(batch_rows))
    for start in range(0, len(row_indices), batch_rows):
        stop = min(start + batch_rows, len(row_indices))
        yield start, stop, row_indices[start:stop]


def iter_row_batches(
    X_source,
    row_indices: np.ndarray,
    *,
    batch_rows: int = DEFAULT_ROW_BATCH_ROWS,
    dtype: np.dtype | type | None = np.float32,
):
    for start, stop, idx_chunk in iter_index_chunks(row_indices, batch_rows=batch_rows):
        chunk = np.asarray(X_source[idx_chunk])
        if dtype is not None and chunk.dtype != dtype:
            chunk = chunk.astype(dtype, copy=False)
        yield start, stop, chunk


def materialize_row_subset(
    X_source,
    row_indices: np.ndarray,
    *,
    batch_rows: int = DEFAULT_ROW_BATCH_ROWS,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    row_indices = np.asarray(row_indices, dtype=np.int64)
    if row_indices.ndim != 1:
        raise ValueError("row_indices must be a 1D array")

    n_rows = len(row_indices)
    feature_shape = tuple(int(dim) for dim in X_source.shape[1:])
    out = np.empty((n_rows, *feature_shape), dtype=dtype)

    for start, stop, chunk in iter_row_batches(X_source, row_indices, batch_rows=batch_rows, dtype=dtype):
        out[start:stop] = chunk

    return out
