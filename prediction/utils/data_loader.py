#!/usr/bin/env python3
"""Load genotype/phenotype data from one TOPMAX dataset directory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .genotype_cache import DEFAULT_CACHE_CHUNK_ROWS, open_genotype_memmap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT.parent / "example_data" / "maize_ncii"


def _as_path(path):
    return Path(path).expanduser().resolve()


def list_traits(data_dir=DEFAULT_DATA_DIR):
    phenotype_path = _as_path(data_dir) / "phenotype.csv"
    pheno = pd.read_csv(phenotype_path, index_col=0, nrows=1)
    return list(pheno.columns)


def load_dataset(
    data_dir=DEFAULT_DATA_DIR,
    trait="CW",
    max_samples=None,
    sample_seed=42,
    use_memmap_cache=False,
    cache_chunk_rows=DEFAULT_CACHE_CHUNK_ROWS,
):
    """
    Load one trait from a directory containing genotype.npz and phenotype.csv.

    genotype.npz must contain X, sample_names, and snp_names. phenotype.csv
    uses sample names as its first column and trait names as the remaining
    columns.
    """
    data_dir = _as_path(data_dir)
    genotype_path = data_dir / "genotype.npz"
    phenotype_path = data_dir / "phenotype.csv"

    if not genotype_path.exists():
        raise FileNotFoundError(f"Missing genotype file: {genotype_path}")
    if not phenotype_path.exists():
        raise FileNotFoundError(f"Missing phenotype file: {phenotype_path}")

    if use_memmap_cache:
        X, sample_names, snp_names, _ = open_genotype_memmap(
            data_dir,
            ensure_cache=True,
            chunk_rows=cache_chunk_rows,
            mmap_mode="r",
        )
    else:
        with np.load(genotype_path, allow_pickle=True) as genotype:
            X = genotype["X"]
            sample_names = genotype["sample_names"]
            snp_names = genotype["snp_names"]

    pheno = pd.read_csv(phenotype_path, index_col=0)
    if trait not in pheno.columns:
        raise ValueError(f"Trait {trait!r} not found. Available traits: {list(pheno.columns)}")

    pheno = pheno.loc[sample_names]
    y = pheno[trait].to_numpy(dtype=np.float32)
    valid_mask = np.isfinite(y)
    if not np.all(valid_mask):
        X = X[valid_mask]
        sample_names = sample_names[valid_mask]
        y = y[valid_mask]

    if max_samples is not None and int(max_samples) < len(y):
        rng = np.random.default_rng(int(sample_seed))
        selected = np.sort(rng.choice(len(y), size=int(max_samples), replace=False))
        X = X[selected]
        sample_names = sample_names[selected]
        y = y[selected]

    if not isinstance(X, np.memmap):
        X = np.asarray(X, dtype=np.float32)
    elif X.dtype != np.float32:
        print(f"Genotypes kept as memmap/{X.dtype}; rows are converted to float32 when used.")

    return X, y, sample_names, snp_names
