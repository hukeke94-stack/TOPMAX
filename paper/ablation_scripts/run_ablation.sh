#!/bin/bash
# Runs all 35 tasks one at a time (8 BLAS threads each), as in the benchmark. OUT defaults to /mnt/data/topmax_ablation.
OUT=${OUT:-/mnt/data/topmax_ablation}; export THREADS=${THREADS:-8}
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
mkdir -p "$OUT"; cd "$(dirname "$0")"
lscpu > "$OUT/lscpu.txt" 2>/dev/null; free -g > "$OUT/free.txt" 2>/dev/null; python -c "import numpy; numpy.show_config()" > "$OUT/numpy_config.txt" 2>&1
TRAITS=("Plant height (cm)_BLUP" "Culm length (cm)_BLUP" "Heading date (d)_BLUP" "Panicle length (cm)_BLUP" "Leaf width (cm)_BLUP" "Panicle number_BLUP" "Leaf length (cm)_BLUP")
for V in full cold_cg rsvd_cg_explicit rsvd_only svd; do
  for T in "${TRAITS[@]}"; do python abl_task.py "$V" "$T" "$OUT" >> "$OUT/log.txt" 2>&1; done
done
echo ALL_DONE >> "$OUT/log.txt"
