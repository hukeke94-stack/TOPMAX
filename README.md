# TOPMAX

TOPMAX is a fast and memory-efficient tool for genomic prediction and multi-trait selection in large breeding populations. It has two modules:

- **Module I, genomic prediction.** Ridge regression on all markers, with accuracy matching GBLUP. A randomized singular value decomposition (RSVD) gives a low-rank view of the genotype matrix, generalized cross-validation (GCV) chooses the ridge penalty, and a warm-started conjugate gradient (CG) solver gives the exact solution. No genomic relationship matrix is built, so memory grows roughly linearly with the number of individuals.
- **Module II, multi-trait selection.** Learns one weight per trait from the predicted and observed phenotypes of training samples, as in TOP, with a mini-batch AdamW optimizer, then ranks candidates by their weighted similarity to a breeding target.

An online version for small datasets is available at <https://topxbio.com>.

## Installation

TOPMAX needs Python 3.9 to 3.12 and four packages (NumPy, pandas, SciPy, scikit-learn).

```bash
git clone https://github.com/hukeke94-stack/TOPMAX.git
cd TOPMAX
conda env create -f environment.yml     # or: pip install -r requirements.txt
conda activate topmax
```

A GPU is optional. To use `--gpu` on an NVIDIA GPU with CUDA 12, also run `pip install -r requirements-gpu.txt`.

## Quick start

The example data are 5,820 maize hybrids from the North Carolina design II (NCII) population of Yang et al. (Genome Biol 2022;23:80), with 20,000 randomly sampled SNP markers and 18 traits. Both commands finish within about a minute on a laptop.

**Prediction.** Five-fold cross-validation on 4,656 training hybrids for cob weight (CW), then prediction of the other 1,164 hybrids:

```bash
cd prediction
python topmax_predict.py --data-dir ../example_data/maize_ncii --trait CW \
    --predict-ids ../example_data/maize_ncii/predict_ids.txt --output results
```

Expected output: cross-validation PCC 0.843 (mean of five folds) and PCC 0.842 between the predicted and observed values of the 1,164 hybrids.

**Selection.** Learn trait weights from 569 training hybrids and rank 5,250 candidates against the target hybrid CHANG7-2 × MG_1522:

```bash
cd selection
python topmax_select.py \
    --train-predictions ../example_data/maize_selection/training_predictions.csv \
    --train-observations ../example_data/maize_selection/training_observations.csv \
    --candidates ../example_data/maize_selection/candidate_predictions.csv \
    --target ../example_data/maize_selection/target.csv --output results
```

To ask for candidates that exceed the target in some traits, add for example `--improve EW=0.2 KNPE=0.1` (20% higher ear weight and 10% more kernels per ear than the target).

Run either script with `--help` to list all options.

## Input files

### Prediction

A dataset is a folder with two files:

| File | Contents |
|---|---|
| `genotype.npz` | NumPy archive with `X` (individuals × markers, coded 0/1/2), `sample_names`, and `snp_names` |
| `phenotype.csv` | Sample IDs in the first column, one trait per remaining column; empty cells are allowed |

A `genotype.npz` can be written from any 0/1/2 matrix, for example one exported from PLINK with `--recode A`:

```python
import numpy as np
np.savez_compressed("genotype.npz", X=X.astype(np.int8), sample_names=np.array(ids), snp_names=np.array(snps))
```

Missing genotypes should be imputed beforehand.

Individuals to predict can be given in three ways:

1. listed in a text file with `--predict-ids` (one ID per line; they are left out of training);
2. left without a phenotype in `phenotype.csv` (they are predicted for that trait);
3. stored in a second folder given with `--predict-dir` (same markers in the same order; a `phenotype.csv` there is used only to report accuracy).

### Selection

All tables are CSV files with sample IDs in the first column and one column per trait, using the same trait names.

| Option | Contents |
|---|---|
| `--train-predictions` | Predicted traits of the training samples, for example from cross-validation with the prediction module |
| `--train-observations` | Observed traits of the same samples |
| `--candidates` | Predicted traits of the candidates to rank |
| `--target` | Trait values of the breeding target (the first row is used unless `--target-id` is given) |

## Output files

| Module | File | Contents |
|---|---|---|
| Prediction | `summary.csv` | Cross-validation PCC (mean and SD) and prediction PCC per trait |
| | `cv_folds.csv` | PCC, R², chosen penalty, CG iterations, and time for each fold |
| | `cv_predictions.csv` | Observed and predicted values of the cross-validation |
| | `predictions.csv` | Predicted values of the individuals to predict |
| Selection | `weights.csv` | Learned weight of each trait |
| | `rank.csv` | Candidates ranked by similarity to the target |

## Parameters

The defaults were used for all results in the paper and work well across the crops and population sizes we tested, so most users do not need to change them.

### Prediction (`topmax_predict.py`)

| Option | Default | Meaning |
|---|---|---|
| `--folds` | 5 | Cross-validation folds; 0 skips cross-validation |
| `--threads` | 8 | CPU threads for linear algebra |
| `--seed` | 42 | Seed for the fold split and the RSVD |
| `--gpu` | off | Run on an NVIDIA GPU |
| `--rsvd-components` | 500 | Rank of the RSVD. It guides the penalty search and the CG starting point; the final solution comes from CG on the full data. In our tests, ranks of 250 and 1,000 gave practically the same predictions as 500. Larger values cost more time |
| `--alpha-min`, `--alpha-max`, `--alpha-points` | −2, 7, 100 | GCV searches 100 penalties from 10⁻² to 10⁷ on a log scale (markers are standardized) |
| `--cg-tol` | 1e-5 | CG stops when the relative residual falls below this value |
| `--cg-max-iter` | 500 (CPU), 100 (GPU) | Upper limit on CG iterations |

### Selection (`topmax_select.py`)

| Option | Default | Meaning |
|---|---|---|
| `--improve` | none | Raise the target for some traits, as `TRAIT=RATIO` |
| `--pcc-threshold` | 0.15 | Traits predicted with a lower PCC in the training samples get no weight |
| `--batch-size` | 64 | Samples per mini-batch |
| `--lr` | 0.001 | Initial learning rate, with cosine decay |
| `--weight-decay` | 0.01 | AdamW weight decay |
| `--tol`, `--patience` | 1e-4, 30 | Stop when no weight changes by more than `tol` for `patience` consecutive epochs |
| `--max-epochs` | 10,000 | Upper limit on epochs |
| `--seed` | 42 | Seed for the mini-batch order |
| `--gpu` | off | Run weight learning on an NVIDIA GPU |

## Repository layout

| Path | Contents |
|---|---|
| `prediction/` | Prediction module: `topmax_predict.py` (command line), `methods/` (CPU and GPU solvers), `utils/` |
| `selection/` | Selection module: `topmax_select.py` (command line), `selection_core.py` (weight learning) |
| `example_data/` | Maize NCII example data for both modules |

## Citation

If you use TOPMAX, please cite: Hu K, Wu B, Xiong H, Fang W, Xiao Y, Yan J, Yang W. TOPMAX: a fast and scalable software framework for genomic prediction and target-oriented multi-trait selection in large breeding populations. Manuscript submitted.

## License

MIT. See [LICENSE](LICENSE).
