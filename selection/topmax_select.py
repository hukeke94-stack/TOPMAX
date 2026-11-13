#!/usr/bin/env python3
"""TOPMAX multi-trait selection.

Learns one weight per trait from the predicted and observed phenotypes of training samples, then ranks
candidates by their weighted similarity to a breeding target.

Example:
    python topmax_select.py --train-predictions ../example_data/maize_selection/training_predictions.csv \
        --train-observations ../example_data/maize_selection/training_observations.csv \
        --candidates ../example_data/maize_selection/candidate_predictions.csv \
        --target ../example_data/maize_selection/target.csv --output results
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from selection_core import scale_z, train_weights  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="TOPMAX multi-trait selection",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    io = p.add_argument_group("input and output")
    io.add_argument("--train-predictions", required=True, help="CSV: predicted traits of the training samples (ID column first)")
    io.add_argument("--train-observations", required=True, help="CSV: observed traits of the same samples")
    io.add_argument("--candidates", required=True, help="CSV: predicted traits of the candidates to rank")
    io.add_argument("--target", required=True, help="CSV: trait values of one or more breeding targets")
    io.add_argument("--target-id", help="Row of --target to use (default: the first row)")
    io.add_argument("--improve", nargs="*", default=[], metavar="TRAIT=RATIO",
                    help="Raise the target for these traits, e.g. EW=0.2 for 20%% above the target")
    io.add_argument("--output", default="results", help="Output folder")
    io.add_argument("--top-n", type=int, default=20, help="Number of top candidates printed to the screen")

    opt = p.add_argument_group("weight learning (defaults are recommended)")
    opt.add_argument("--pcc-threshold", type=float, default=0.15, help="Use only traits predicted with at least this PCC in the training samples")
    opt.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    opt.add_argument("--lr", type=float, default=0.001, help="Initial learning rate (cosine decay)")
    opt.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    opt.add_argument("--max-epochs", type=int, default=10000, help="Upper limit on epochs")
    opt.add_argument("--tol", type=float, default=1e-4, help="Stop when no weight changes by more than this ...")
    opt.add_argument("--patience", type=int, default=30, help="... for this many consecutive epochs")
    opt.add_argument("--seed", type=int, default=42, help="Seed for the mini-batch order")
    opt.add_argument("--gpu", action="store_true", help="Run weight learning on an NVIDIA GPU (needs PyTorch with CUDA)")
    return p.parse_args()


def read_table(path):
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df


def main():
    args = parse_args()
    improve = {}
    for item in args.improve:
        trait, _, ratio = item.partition("=")
        if not ratio:
            raise SystemExit(f"--improve expects TRAIT=RATIO, got {item!r}")
        improve[trait] = float(ratio)

    use_gpu = False
    if args.gpu:
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("--gpu was given but PyTorch finds no CUDA device")
        use_gpu = True

    opts = SimpleNamespace(train_predictions=args.train_predictions, train_observations=args.train_observations, num_train_samples=0,
                           pcc_threshold=args.pcc_threshold, batch_size=args.batch_size, n_epochs=args.max_epochs, lr=args.lr,
                           weight_decay=args.weight_decay, beta1=0.9, beta2=0.999, tol=args.tol, patience=args.patience,
                           no_adaptive_stop=False, seed=args.seed)
    start = time.time()
    weights, used, obs_mean, obs_sd, _ = train_weights(opts, {"use_gpu": use_gpu})
    seconds = time.time() - start

    cand = read_table(args.candidates)
    target_table = read_table(args.target)
    target_id = args.target_id or target_table.index[0]
    if target_id not in target_table.index:
        raise SystemExit(f"Target {target_id!r} not found in {args.target}")
    traits = list(cand.columns)
    missing = [t for t in traits if t not in target_table.columns]
    if missing:
        raise SystemExit(f"Target table lacks traits {missing}")
    for t in improve:
        if t not in traits:
            raise SystemExit(f"--improve trait {t!r} is not a candidate trait")
    goal = target_table.loc[target_id, traits].to_numpy(dtype=float).copy()
    for t, ratio in improve.items():
        goal[traits.index(t)] *= 1 + ratio

    # Similarity: exp of the weighted absolute distance to the target on the standardized scale (traits not used get weight 0)
    w = dict(zip(used, np.asarray(weights, dtype=float)))
    w_neg = -np.array([w.get(t, 0.0) for t in traits])
    mean = np.array([obs_mean.get(t, 0.0) for t in traits])
    sd = np.array([obs_sd.get(t, 1.0) for t in traits])
    dist = np.abs(scale_z(cand.to_numpy(dtype=float), mean, sd) - scale_z(goal.reshape(1, -1), mean, sd).ravel())
    score = np.exp(dist @ w_neg)

    rank = pd.DataFrame({"ID": cand.index, "SimilarityScore": score})
    rank["Rank"] = rank.SimilarityScore.rank(ascending=False, method="min").astype(int)
    rank["TopPercent"] = rank.Rank / len(rank) * 100
    rank = rank.sort_values(["Rank", "ID"])[["Rank", "ID", "SimilarityScore", "TopPercent"]]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rank.to_csv(out / "rank.csv", index=False)
    pd.DataFrame({"Trait": used, "Weight": [float(x) for x in weights]}).sort_values("Weight", ascending=False).to_csv(out / "weights.csv", index=False)

    print(f"\nTarget: {target_id}" + (f", improved: {improve}" if improve else ""))
    print(f"Weight learning: {seconds:.1f} s, {len(used)} traits; {len(cand)} candidates ranked")
    print(rank.head(args.top_n).to_string(index=False))
    print(f"\nResults written to {out.resolve()} (rank.csv, weights.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
