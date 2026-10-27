"""Groundwater Level Forecasting (India) — Results Dashboard.

Station-holdout forecasts of groundwater depth at 30/60/90-day horizons using
Linear Regression, Random Forest, and XGBoost on 27 years of CGWB-style
observations (416,951 rows, 6,440 stations, 1994-2025).
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

st.set_page_config(page_title="Groundwater Level Forecasting", page_icon="💧", layout="wide")


@st.cache_data
def load_summary():
    return json.loads((ASSETS / "dataset_summary.json").read_text())


@st.cache_data
def load_metrics():
    df = pd.read_csv(ASSETS / "phase1_model_metrics.csv")
    df["MAE"] = df["MAE"].round(3)
    df["RMSE"] = df["RMSE"].round(3)
    df["R2"] = df["R2"].round(4)
    return df


def avp_path(horizon: int, model: str) -> Path:
    return ASSETS / f"actual_vs_pred_{horizon}d_{model}.png"


summary = load_summary()
metrics_df = load_metrics()

st.title("💧 Groundwater Level Forecasting — India")
st.markdown(
    "**Long-horizon forecasting of groundwater level** from 27 years of station "
    "observations (1994–2025). Models are evaluated on a strict **station holdout**: "
    "test stations are geographically unseen during training, so scores measure "
    "generalization to new regions — not memorized time series."
)

# ---- Headline metrics ----
best = metrics_df[metrics_df.horizon_days == 30].set_index("model")
rf30 = best.loc["Random Forest"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset size", f"{summary['rows']:,} rows")
c2.metric("Stations", f"{summary['stations']:,}")
c3.metric("Best model (30d)", "Random Forest")
c4.metric("R² @ 30 days", f"{rf30.R2:.2f}", f"MAE {rf30.MAE:.2f} m")

st.divider()

# ---- Metrics table ----
st.header("📊 Benchmark — 3 models × 3 horizons")
st.markdown(
    "Every model is evaluated on held-out stations at the exact target date "
    "(30/60/90 days ahead). Depth-to-water in metres."
)
pivot_r2 = metrics_df.pivot(index="model", columns="horizon_days", values="R2")[[30, 60, 90]]
pivot_mae = metrics_df.pivot(index="model", columns="horizon_days", values="MAE")[[30, 60, 90]]
c1, c2 = st.columns(2)
c1.subheader("R² (higher is better)")
c1.dataframe(pivot_r2, use_container_width=True)
c2.subheader("MAE in metres (lower is better)")
c2.dataframe(pivot_mae, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.image(str(ASSETS / "comparison_r2.png"), use_container_width=True)
c2.image(str(ASSETS / "comparison_mae.png"), use_container_width=True)
c3.image(str(ASSETS / "comparison_rmse.png"), use_container_width=True)

st.divider()

# ---- Per-horizon scatter plots ----
st.header("🔮 Predictions vs. observations")
st.markdown(
    "Each panel plots predicted vs. actual depth for exactly one horizon and "
    "model on unseen stations. The diagonal is perfect prediction; points below "
    "the diagonal are overestimates of decline."
)
horizon = st.radio("Forecast horizon", [30, 60, 90], horizontal=True)
c1, c2, c3 = st.columns(3)
c1.image(str(avp_path(horizon, "linear_regression")), caption="Linear Regression", use_container_width=True)
c2.image(str(avp_path(horizon, "xgboost")), caption="XGBoost", use_container_width=True)
c3.image(str(avp_path(horizon, "random_forest")), caption="Random Forest", use_container_width=True)

st.divider()

# ---- Dataset ----
st.header("🗂 Data & pipeline")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{summary['rows']:,}")
c2.metric("Stations", f"{summary['stations']:,}")
c3.metric("Date range", "1994–2025")
c4.metric("Missing values", "0")

st.markdown(
    """
1. **Clean** — parse observations, remove zero-variance records, validate date coverage.
2. **Feature build** — exact-horizon targets (same station observed exactly *n* days after the cutoff).
3. **Split** — station holdout: ~80% of stations train, held-out stations never appear in training.
4. **Train & evaluate** — Linear Regression / Random Forest / XGBoost, MAE–RMSE–R² reporting per horizon.
"""
)

st.divider()
st.caption(
    "Built with Python · pandas · scikit-learn · XGBoost · Streamlit. "
    "Source code: see linked GitHub repository."
)
