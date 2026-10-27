# Groundwater Level Forecasting — India

Long-horizon forecasting of groundwater level (depth-to-water, metres) from 27 years
of station observations across India, evaluated on a strict **station holdout**
(held-out test stations never appear in training).

**Live demo:** _Streamlit Cloud deployment — see RESUME link_

| Model | Horizon | MAE (m) | R² |
|---|---|---|---|
| Linear Regression | 30 d | 7.65 | 0.58 |
| Random Forest | **30 d** | **1.97** | **0.90** |
| XGBoost | 30 d | 2.78 | 0.71 |
| Random Forest | 60 d | 3.37 | 0.67 |
| XGBoost | 60 d | 3.27 | 0.75 |
| Random Forest | 90 d | 3.87 | 0.85 |
| XGBoost | 90 d | 3.54 | 0.88 |

## Highlights
- **416,951 observations · 6,440 stations · 1994–2025**, zero missing values after cleaning.
- Exact-horizon target construction (the same station observed exactly *n* days after the cutoff), not sampled future snapshots.
- **Station holdout split** — R² scores measure generalization to unseen regions, not memorized time series. Linear Regression collapses (R² −2.45 @ 60d) while tree ensembles hold up, which is exactly the failure mode this split is designed to expose.
- Three models × three horizons benchmarked head-to-head (MAE / RMSE / R²).

## Pipeline
1. Load & clean raw station observations.
2. Build exact-horizon forecast datasets (30/60/90 days ahead) from engineered date features.
3. Station holdout split (~80% of stations train / held-out stations test).
4. Train Linear Regression, Random Forest, XGBoost; evaluate and export metrics + prediction scatter plots.

## Files
- `app.py` — Streamlit results dashboard (deployed app).
- `assets/` — model comparison charts and actual-vs-predicted scatter plots per horizon.

## Reproducing
Data source (`Groundwater India Data.csv`, 34 MB) is not committed here. The training
pipeline lives in the sibling project folder: `main.py` → `src/` (preprocess → features →
split → train → evaluate), writing results to `outputs/`.
