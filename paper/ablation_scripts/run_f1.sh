#!/bin/bash
# F1 100K ablation: 4 variants x traits, one task at a time with 8 BLAS threads (same as the benchmark). Shuts the instance down at the end.
OUT=/root/autodl-tmp/topx_ablation_f1; mkdir -p $OUT; cd "$(dirname "$0")"
export THREADS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=${PY:-python}
lscpu > $OUT/lscpu.txt; free -g > $OUT/free.txt
for T in $TRAITS; do
  for V in full cold_cg rsvd_only rsvd_cg_explicit; do $PY abl_f1_task.py $V "$T" $OUT >> $OUT/log.txt 2>&1; done
done
echo ALL_DONE >> $OUT/log.txt
touch $OUT/FINISHED
# safety: power off 30 min after finishing even if nobody collects the results (they stay on the data disk)
sleep 1800; shutdown -h now
