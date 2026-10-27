# Groundwater Depletion Forecasting & Early-Warning Risk Dashboard

Interactive dashboard for long-horizon groundwater level forecasting across India
(1994–2025). Every forecast shown is a **held-out example on wholly unseen monitoring
stations** — the stations on display never appeared in training or validation.

**Live app:** _Streamlit Community Cloud — link on resume / repo description_

## What the app shows
- **Per-station forecast viewer** — pick an unseen test station and observation date; see
  current level, the t+30/60/90-day forecast vs. the observed held-out target, and the
  station's full raw history.
- **SHAP explainability** — per-forecast feature-attribution chart (tree models).
- **National unseen-station map** — predicted Safe / Semi-critical / Critical tiers per
  station on an interactive map.
- **Benchmark & robustness** — 3 model families × 3 horizons on unseen stations,
  risk-tier alert metrics, hyperparameter validation search, outlier sensitivity.

## Benchmark (test = fully unseen stations, seed 42)

| Model (config) | Horizon | MAE (m) | RMSE | R² |
|---|---|---|---|---|
| Random Forest (depth 12, 90 trees) | 30 d | **1.82** | 5.16 | **0.903** |
| XGBoost (depth 4, 200 trees) | 30 d | 2.86 | 11.06 | 0.555 |
| Linear Regression | 30 d | 7.65 | 10.77 | 0.578 |
| XGBoost | 60 d | 2.80 | 6.50 | 0.843 |
| Random Forest | 60 d | 3.28 | 9.98 | 0.629 |
| XGBoost | 90 d | 3.46 | 6.40 | 0.868 |
| Random Forest | 90 d | 3.82 | 6.80 | 0.851 |

## Data & protocol
- 416,951 observations · 6,440 stations · 1994–2025 · zero missing values after cleaning.
- **Station holdout split** — development and test sets contain disjoint stations, so R²
  measures spatial generalization to unseen regions, not memorized time series. Linear
  Regression collapses at 60 d (R² −2.45) while tree ensembles hold, which is exactly the
  failure mode this protocol is designed to expose.
- Exact-horizon targets: the same station must be observed exactly *n* days after the cutoff.
- Risk tiers are a movement-magnitude proxy (percentiles of |target(t+h) − target(t)| on
  development stations), not official CGWB resource categories.
- Model selection by lowest validation MAE; test stations never used for any comparison.

## Files
- `app.py` — Streamlit dashboard.
- `outputs/models/` — trained per-horizon models (joblib bundles).
- `outputs/metrics/` — precomputed held-out predictions, final test metrics, alert metrics,
  validation search, outlier robustness.

## Reproducing / deploying
No retraining happens in the app — it loads precomputed artifacts. Deploy: Streamlit
Community Cloud → this repo → main file `app.py`. Pinned versions in `requirements.txt`
(match `environment_versions.json` from the packaging machine). The 34 MB training CSV and
`phase2.py` are intentionally not committed.
