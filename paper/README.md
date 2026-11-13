# Figures and source data of the TOPMAX paper

This folder reproduces the figures of the manuscript from the benchmark results. It is not needed to use TOPMAX.

## Run

```bash
cd paper/analysis
python fig_selection.py      # Fig. 4, and the selection table that Fig. 5 reads
python fig_final.py          # Fig. 2, Fig. 3, Fig. 5
python fig_supplement.py     # Fig. S1, S2, S3
python fig1_compose.py ../figures/fig1_gpt/logo_used_fig1_v5.png ../figures/final/Fig1_overview.png
python fig6_compose.py       # Fig. 6 from the screenshots in figures/fig6_shots
```

Figures are written to `figures/final/`; the copies there are the versions in the manuscript. The scripts need NumPy, pandas, SciPy, matplotlib, and Pillow, and the Arial and Arial Narrow fonts. `fig_style.py` holds the shared style (170 mm width, one colour and marker per method).

`fig6_screenshots.py` retakes the screenshots of the online platform (<https://topxbio.com>) by running the built-in maize example in a headless browser; it needs Playwright and Google Chrome.

## Data

| File | Contents |
|---|---|
| `analysis/results_summary.csv` | One row per prediction run (method, dataset, sample size, trait, fold): PCC, R², runtime, peak memory, chosen penalty, and stage times |
| `analysis/svr_default/svr_default_results.csv` | SVR rerun with default parameters |
| `analysis/source_data/selection_scores_*.csv` | Candidate scores of TOP and TOPMAX for maize, rice, and wheat |
| `analysis/source_data/selection_benchmarks.csv` | Runtime and peak memory of weight learning (real crops and simulated F1 panels) |
| `selection_demodata/` | Predicted traits of the candidates, the breeding target, and the observed traits used to standardize distances (Fig. 4a-c) |
| Other CSV files in `analysis/` | Summary tables written by the scripts (numbers quoted in the text) |

Peak memory is the anonymous memory of each program's process tree, which excludes file pages mapped from the genotype matrix.

Fig. 1 starts from the original schematic (`figures/fig1_gpt/input_pic1_final.png`); `fig1_compose.py` replaces only the listed rectangles (logo, labels, result cards, the HIBLUP curve) and checks that every other pixel is unchanged. The logo was made with an image-generation tool.
