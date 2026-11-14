# Realized-Volatility Forecasting for Indian Equities: HAR-RV, EGARCH and LSTM

AI Lab project, M.Sc. Quantitative Finance, Ahmedabad University
Partha Guntur · Rajvi Panchal

Next-day realized volatility of 95 NIFTY 100 stocks is forecast from 5-minute data (2016 – April 2026)
with five models, and the results are compared overall, by liquidity group and by market regime.

| Model | What it uses |
|---|---|
| HAR-RV | daily / weekly / monthly realized variance, linear (Corsi, 2009) |
| log-HAR | same, in logs |
| **HAR-RV-LSTM** | same three RV components over 22 days, pooled LSTM with stock embedding |
| EGARCH-LSTM | EGARCH(1,1)-t volatility and shock from daily returns, pooled LSTM |
| HAR-EGARCH-LSTM | both input sets (ablation) |

The full write-up is `paper/Volatility_Forecasting_Paper.pdf`.

## Folder layout

```
project/
├── README.md                  this file
├── requirements.txt
├── src/volforecast/           all reusable code (imported by every notebook)
│   ├── config.py              paths, sample dates, cleaning thresholds, model settings
│   ├── data_pipeline.py       cleaning, realized variance, liquidity groups, HAR features
│   ├── har.py                 HAR-RV and log-HAR (one regression per stock)
│   ├── egarch.py              EGARCH(1,1)-t features fitted on training years only
│   ├── lstm.py                LSTM network, sequences, training, seeds
│   ├── evaluation.py          losses, Diebold–Mariano, bootstrap, Mincer–Zarnowitz
│   └── plotting.py            shared chart style and model colours
├── notebooks/                 run in order 01 → 06
│   ├── 01_data_cleaning_and_liquidity.ipynb
│   ├── 02_har_benchmarks.ipynb
│   ├── 03_egarch_features.ipynb
│   ├── 04_har_rv_lstm.ipynb
│   ├── 05_egarch_lstm.ipynb   (EGARCH-LSTM and the HAR-EGARCH-LSTM ablation)
│   └── 06_model_comparison.ipynb
├── data_processed/            daily panel, EGARCH features, liquidity groups, universe
├── results/
│   ├── forecasts/             forecast for every stock-day (validation and test)
│   ├── tables/                every table in the paper as CSV
│   └── models/                trained LSTM weights (3 seeds per model)
├── figures/                   every chart (PNG for viewing, PDF for the paper)
├── paper/                     LaTeX source and the compiled paper
└── docs/                      Milestone 1 proposal
```

## How to run

1. Keep the raw 5-minute files outside the project, next to it:
   `AI Lab/Project/data/<SYMBOL>_5minute.csv` and `AI Lab/Project/project/` (this folder).
   If they are elsewhere, set `VOL_DATA_DIR=/path/to/data` before starting Jupyter.
2. `pip install -r requirements.txt`
3. Run the notebooks in order. Notebook 01 takes about 2 minutes; 04 and 05 train
   3 networks each and take roughly 20–40 minutes each on a laptop CPU.
4. Notebook 06 rebuilds every table and figure used in the paper from the saved forecasts,
   so it can be re-run on its own.

To rebuild the paper: `cd paper && latexmk -pdf main.tex`.

## Data cleaning in one paragraph

Bars outside 09:15–15:25 are dropped; rows before the listing date (IRFC, SBILIFE, HDFCLIFE,
MAZDOCK) are removed; zero prices and zero-volume quotes carry no price; isolated spikes that are
reversed in the next bar are removed (e.g. VBL 14-Jun-2023, 322 → 644 → 322); days with fewer
than 60 traded bars or with RV = 0 are dropped. RV uses intraday returns only, so overnight
splits, bonus issues and demergers never enter the target. Details: `src/volforecast/config.py`
and `results/tables/data_cleaning_log.csv`.

## Changes from the first version of the project

| Issue in the first version | Fix |
|---|---|
| Liquidity slope divided signed volume by its own near-zero mean, so the groups were close to random (ICICIBANK "Medium", INFY "Low") | 5-minute Amihud ratio on training years only, as in the proposal |
| EGARCH estimated on 2016–2026, so test-year information leaked into the features | parameters estimated on 2016–2022, then run forward with fixed parameters |
| RV included overnight gaps and bad prints (VBL 2023 split day), inflating HAR errors | intraday-only RV and the cleaning rules above |
| Level-HAR forecasts ≤ 0 were silently dropped from scoring | floored and counted (none occur after cleaning) |
| One training run per LSTM | three seeds, averaged, seed spread reported |
| Volatility regimes defined with the realized outcome (look-ahead) | ex-ante regime from training-period cut-offs; ex-post kept as descriptive |
| No log-HAR benchmark | added; it is the fair benchmark for a model with log inputs |
