"""Interactive demo of precomputed held-out station forecasts.

Upgraded dashboard: KPI cards, Safe/Semi-critical/Critical tier badges,
an India unseen-station map, and diverging SHAP contribution charts.
All artifacts are precomputed by phase2.py; nothing retrains here.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import altair as alt

ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "outputs" / "metrics"
MODELS = ROOT / "outputs" / "models"
DATA = ROOT / "data" / "Groundwater India Data.csv"
HORIZONS = [30, 60, 90]

# Problem-statement tier names rendered from the internal movement labels.
DISPLAY_LABELS = {
    "Stable": "Safe (Stable)",
    "Watch": "Semi-critical (Watch)",
    "Large change": "Critical (Large change)",
}
TIER_NOTE = (
    "Movement-magnitude proxy: tiers reuse the Safe/Semi-critical/Critical naming "
    "requested in the project statement, but are derived from the 50th and 85th "
    "percentiles of |target(t+h)-target(t)| on development stations only. They are "
    "not the official CGWB resource categories, which require extraction-to-recharge "
    "data absent from this dataset."
)

st.set_page_config(page_title="Groundwater Forecast & Risk Dashboard", page_icon="💧", layout="wide")

st.markdown("""
<style>
.risk-badge { padding: 8px 16px; border-radius: 6px; font-weight: 700; text-align: center; display: inline-block; font-size: 1.05rem; }
.risk-safe { background-color: #d1fae5; color: #065f46; border: 1px solid #34d399; }
.risk-semi { background-color: #fef3c7; color: #92400e; border: 1px solid #fbbf24; }
.risk-crit { background-color: #fee2e2; color: #991b1b; border: 1px solid #f87171; }
</style>
""", unsafe_allow_html=True)

st.title("💧 Groundwater Depletion Forecasting & Early-Warning Risk Dashboard")
st.caption(
    "Spatial-generalization benchmark on wholly unseen monitoring stations across India (1994-2025) | "
    "Random Forest, XGBoost and Linear Regression with SHAP explainability."
)


def render_badge(tier: str) -> str:
    cls = "risk-safe" if "Safe" in tier else ("risk-semi" if "Semi-critical" in tier else "risk-crit")
    return f'<div class="risk-badge {cls}">{tier}</div>'


@st.cache_data
def predictions(h):
    frame = pd.read_csv(METRICS / f"heldout_predictions_{h}d.csv", parse_dates=["datetime"])
    frame["target_date"] = frame["datetime"] + pd.Timedelta(days=h)
    frame["pred_change"] = frame["predicted_target"] - frame["target"]
    frame["actual_change"] = frame["future_target"] - frame["target"]
    frame["pred_risk_tier"] = frame["predicted_alert"].map(DISPLAY_LABELS).fillna(frame["predicted_alert"])
    frame["true_risk_tier"] = frame["observed_alert"].map(DISPLAY_LABELS).fillna(frame["observed_alert"])
    return frame


@st.cache_resource
def model_data(h):
    return joblib.load(MODELS / f"horizon_{h}.joblib")


@st.cache_data
def summary_tables():
    metrics = pd.read_csv(METRICS / "phase2_final_test_metrics.csv")
    alerts = json.loads((METRICS / "phase2_alert_metrics.json").read_text())
    tuning = pd.read_csv(METRICS / "phase2_validation_tuning.csv")
    robust = pd.read_csv(METRICS / "phase2_outlier_robustness.csv")
    return metrics, alerts, tuning, robust


@st.cache_data
def full_history():
    if not DATA.exists():
        return None
    df = pd.read_csv(DATA, usecols=["station_id", "datetime", "target"])
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df.dropna()


if not all(
    (METRICS / f"heldout_predictions_{h}d.csv").exists() and (MODELS / f"horizon_{h}.joblib").exists()
    for h in HORIZONS
):
    st.error("Missing Phase 2 artifacts in outputs/. Run `python phase2.py` first, then reload.")
    st.stop()

try:
    bundle = model_data(st.sidebar.selectbox("Forecast horizon (days)", HORIZONS, index=0))
except Exception as exc:
    bundle = None
    st.warning(f"Saved model could not load in this environment ({exc}). Saved forecasts still work; SHAP is unavailable.")
    st.stop()
h = bundle["horizon_days"]
data = predictions(h)
metrics, alerts, tuning, robust = summary_tables()
history = full_history()

stations = sorted(data["station_id"].unique())
station = st.sidebar.selectbox("Unseen test station", stations)
s = data[data["station_id"] == station].sort_values("datetime").reset_index(drop=True)
if len(s) == 1:
    st.sidebar.caption("This station has one eligible historical observation for this horizon.")
date_labels = s["datetime"].dt.strftime("%Y-%m-%d").tolist()
chosen = st.sidebar.selectbox("Observation date (t)", date_labels, index=len(date_labels) - 1)
row = s[s["datetime"].dt.strftime("%Y-%m-%d") == chosen].iloc[0]

with st.sidebar:
    st.subheader("Selected station")
    st.write(f"**Station ID:** `{station}`")
    st.write(f"**Coordinates:** {row['latitude']:.3f} N, {row['longitude']:.3f} E")
    st.write(f"**Well depth:** {row['wellDepth']:.1f} m")
    st.write(f"**Model:** {bundle['selected_model']} ({bundle['configuration']})")
    st.caption(TIER_NOTE)

# --- KPI cards -------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(f"Current target y(t)", f"{row['target']:.2f}")
k2.metric(f"Forecast y(t+{h}d)", f"{row['predicted_target']:.2f}", delta=f"{row['pred_change']:+.2f} predicted shift")
k3.metric(f"Observed y(t+{h}d)", f"{row['future_target']:.2f}", delta=f"{row['actual_change']:+.2f} actual shift", delta_color="off")
with k4:
    st.markdown("**Predicted tier**")
    st.markdown(render_badge(row["pred_risk_tier"]), unsafe_allow_html=True)
with k5:
    st.markdown("**Observed tier**")
    st.markdown(render_badge(row["true_risk_tier"]), unsafe_allow_html=True)
st.caption(f"Observation date {row['datetime'].date()} · target date {row['target_date'].date()} · "
           "target direction and units are not established by the source schema, so shift signs are not interpreted.")

tab1, tab2, tab3 = st.tabs(["📈 Station trajectory & SHAP drivers", "🗺️ National unseen-station tiers", "📊 Benchmark & robustness"])

with tab1:
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader(f"Trajectory — station `{station}`")
        series_name = "Observed target"
        if history is not None and history[history.station_id == station].shape[0] > 2:
            hist = history[history.station_id == station].sort_values("datetime")
            line = alt.Chart(hist).mark_line(color="#64748b", opacity=0.7).encode(
                x=alt.X("datetime:T", title="Date"),
                y=alt.Y("target:Q", title="Groundwater target", scale=alt.Scale(zero=False)),
                tooltip=["datetime:T", "target:Q"],
            )
        else:
            hist = s[["datetime", "target"]].rename(columns={"target": "Observation"})
            line = alt.Chart(hist).mark_point(color="#64748b").encode(
                x=alt.X("datetime:T", title="Date"), y=alt.Y("Observation:Q", title="Groundwater target"),
                tooltip=["datetime:T", "Observation:Q"],
            )
        points = pd.DataFrame({
            "datetime": [row["datetime"], row["target_date"], row["target_date"]],
            "value": [row["target"], row["predicted_target"], row["future_target"]],
            "Series": ["Current y(t)", f"Forecast y(t+{h}d)", "Observed y(t+h)"],
        })
        order = ["Current y(t)", f"Forecast y(t+{h}d)", "Observed y(t+h)"]
        dot = alt.Chart(points).mark_circle(size=140).encode(
            x="datetime:T", y=alt.Y("value:Q", scale=alt.Scale(zero=False)), color=alt.Color(
                "Series:N", scale=alt.Scale(domain=order, range=["#0284c7", "#dc2626", "#16a34a"]), sort=order),
            tooltip=["Series:N", "datetime:T", "value:Q"],
        )
        st.altair_chart((line + dot).interactive(), width="stretch")
        st.caption("The station is absent from model training and validation; y(t+h) is a held-out future target.")
        st.dataframe(
            s[["datetime", "target", "predicted_target", "future_target", "pred_risk_tier", "true_risk_tier"]],
            hide_index=True, width="stretch",
        )
    with right:
        st.subheader("What drove this forecast? (SHAP)")
        fitted = bundle["model"]
        if not hasattr(fitted, "feature_importances_"):
            st.info("Feature attributions are unavailable for this model type.")
        else:
            try:
                features = bundle["features"]
                x_one = pd.DataFrame([{c: row[c] for c in features}])[features]
                sv = np.asarray(shap.TreeExplainer(fitted).shap_values(x_one)).reshape(-1)
                shap_df = pd.DataFrame({
                    "Feature": features,
                    "Value": [float(x_one.iloc[0][c]) for c in features],
                    "SHAP": sv,
                    "Direction": np.where(sv >= 0, "Pushes target up", "Pushes target down"),
                }).sort_values("SHAP", key=np.abs, ascending=False).head(8)
                chart = alt.Chart(shap_df).mark_bar().encode(
                    x=alt.X("SHAP:Q", title="SHAP contribution"),
                    y=alt.Y("Feature:N", sort="-x"),
                    color=alt.Color("Direction:N", scale=alt.Scale(
                        domain=["Pushes target up", "Pushes target down"], range=["#ef4444", "#3b82f6"])),
                    tooltip=["Feature", "Value", "SHAP"],
                ).properties(height=320)
                st.altair_chart(chart, width="stretch")
                st.caption("Associative, not causal: SHAP only describes how this tree-model forecast was computed.")
            except Exception as exc:
                st.info(f"Attribution could not be computed: {exc}")

with tab2:
    st.subheader(f"Predicted tiers across unseen test stations — {h}-day horizon")
    latest = data.sort_values("datetime").groupby("station_id", as_index=False).tail(1)
    m1, m2, m3 = st.columns(3)
    m1.metric("Unseen test stations", f"{latest.station_id.nunique():,}")
    m2.metric("Safe (Stable)", f"{(latest.pred_risk_tier.str.startswith('Safe')).sum():,}")
    m3.metric("Semi-critical + Critical", f"{(~latest.pred_risk_tier.str.startswith('Safe')).sum():,}")
    st.map(latest[["latitude", "longitude"]], zoom=4)
    st.dataframe(
        latest[["station_id", "latitude", "longitude", "wellDepth", "target", "predicted_target",
                "pred_risk_tier", "true_risk_tier"]].head(60),
        hide_index=True, width="stretch",
    )
    st.caption("Latest eligible observation per station; each row is a held-out historical example, not a live reading.")

with tab3:
    st.subheader("Regression metrics on wholly unseen test stations")
    st.dataframe(metrics, hide_index=True, width="stretch")
    st.subheader("Tier alert metrics (validation-selected model per horizon)")
    selected_cfgs = {}
    for hh in HORIZONS:
        winner = tuning[tuning.horizon_days == hh].sort_values("validation_mae").iloc[0]
        selected_cfgs[hh] = (winner["model"], winner["configuration"])
    alert_rows = []
    for hh, (model, cfg) in selected_cfgs.items():
        for a in alerts:
            if (a["horizon_days"], a["model"], a["configuration"]) == (hh, model, cfg):
                alert_rows.append({k: a[k] for k in (
                    "horizon_days", "model", "configuration", "threshold_50", "threshold_85",
                    "accuracy", "balanced_accuracy", "macro_f1")})
    st.dataframe(pd.DataFrame(alert_rows), hide_index=True, width="stretch")
    st.subheader("Hyperparameter validation search")
    st.dataframe(tuning, hide_index=True, width="stretch")
    st.subheader("Extreme-target outlier sensitivity")
    st.dataframe(robust, hide_index=True, width="stretch")
    st.caption("Selection was by validation MAE on disjoint stations; the 20% test stations were never used to compare models.")
