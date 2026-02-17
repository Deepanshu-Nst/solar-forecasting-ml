# ============================================================
#  SOLAR POWER FORECASTING SYSTEM — STABLE VERSION (UNCERTAINTY ENABLED)
# ============================================================

import plotly.graph_objects as go
import plotly.express as px

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.data.cleaner import clean_solar_data
from src.features.engineer import create_features
from src.pipeline.forecast import run_forecast_pipeline
from src.models.uncertainty import predict_with_uncertainty   # ⭐ NEW


# ============================================================
# UI STYLE — GLOBAL THEME
# ============================================================

st.markdown("""
<style>
.stApp { background-color:#0E1117; }
.block-container { max-width:1200px; padding-top:2rem; }

.card {
    background:#161B22;
    padding:22px;
    border-radius:14px;
    border:1px solid #30363D;
    margin-bottom:22px;
}

.metric-card {
    background:linear-gradient(145deg,#1f2630,#141a22);
    padding:18px;
    border-radius:12px;
    text-align:center;
    border:1px solid #2f3542;
}

section[data-testid="stSidebar"] {
    background:#0B0F14;
    border-right:1px solid #2f3542;
}

[data-testid="stDataFrame"] {
    border-radius:12px;
    overflow:hidden;
}

.stButton>button {
    border-radius:8px;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(page_title="Solar Power Forecasting", layout="wide")

st.markdown("""
<h1 style='text-align:center;margin-bottom:0;'> Solar Power Forecasting System</h1>
<p style='text-align:center;color:#8b949e;margin-top:4px;'>
AI-powered solar generation prediction & monitoring dashboard
</p>
""", unsafe_allow_html=True)

MODEL_PATH = Path("models/random_forest.pkl")


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("""
<h2 style='text-align:center;'> Solar AI</h2>
<p style='text-align:center;color:#8b949e;'>Prediction Console</p>
<hr>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["Upload Data", "Forecast Dashboard", "Model Insights"]
)


# ============================================================
# PAGE 1 — UPLOAD DATA
# ============================================================

if page == "Upload Data":

    st.header("Upload Solar Data")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:

        try:
            raw_df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])

            from src.data.validator import run_all_validations
            raw_df, missing_report = run_all_validations(raw_df)

            st.success("Data validation passed")

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Missing Values Report")
            st.dataframe(missing_report)
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error("Invalid input data")
            st.exception(e)
            st.stop()

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Raw Data Preview")
        st.dataframe(raw_df.head())
        st.markdown('</div>', unsafe_allow_html=True)

        cleaned_df, report = clean_solar_data(raw_df)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Cleaned Data Preview")
        st.dataframe(cleaned_df.head())
        st.markdown('</div>', unsafe_allow_html=True)

        feature_df = create_features(cleaned_df)
        forecast_df = run_forecast_pipeline(feature_df, model)

        st.session_state["raw_df"] = raw_df
        st.session_state["cleaned_df"] = cleaned_df
        st.session_state["X"] = feature_df
        st.session_state["forecast_df"] = forecast_df

        st.success("Forecast generated successfully. Go to Dashboard.")


# ============================================================
# PAGE 2 — FORECAST DASHBOARD
# ============================================================

elif page == "Forecast Dashboard":

    st.header("Forecast Dashboard")

    if model is None:
        st.error("Model file not found: models/random_forest.pkl")
        st.stop()

    if "cleaned_df" not in st.session_state:
        st.warning("Upload data first")
        st.stop()

    cleaned_df = st.session_state["cleaned_df"]
    feature_df = create_features(cleaned_df.copy())

    y_true = feature_df["power"]
    X = feature_df.drop(columns=["power", "timestamp"], errors="ignore")
    st.session_state["X"] = X

    # ⭐ UNCERTAINTY PREDICTION
    mean_pred, lower_bound, upper_bound = predict_with_uncertainty(model, X)

    forecast_df = feature_df.copy()
    forecast_df["actual_power"] = y_true
    forecast_df["predicted_power"] = mean_pred
    forecast_df["lower_bound"] = lower_bound
    forecast_df["upper_bound"] = upper_bound
    forecast_df["error"] = y_true - mean_pred
    st.session_state["forecast_df"] = forecast_df

    mae_value = mean_absolute_error(y_true, mean_pred)
    rmse_value = np.sqrt(mean_squared_error(y_true, mean_pred))
    mape_value = np.mean(np.abs((y_true - mean_pred) / (y_true + 1e-6))) * 100
    r2_value = r2_score(y_true, mean_pred)

    # ===== METRICS =====
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Performance")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("MAE", f"{mae_value:.3f}")
    c2.metric("RMSE", f"{rmse_value:.3f}")
    c3.metric("MAPE", f"{mape_value:.2f}%")
    c4.metric("R²", f"{r2_value:.3f}")
    st.markdown('</div>', unsafe_allow_html=True)

    # ===== ACTUAL VS PREDICTED WITH CONFIDENCE BAND =====
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Actual vs Predicted Power (Confidence Interval)")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecast_df["timestamp"],
        y=forecast_df["upper_bound"],
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["timestamp"],
        y=forecast_df["lower_bound"],
        fill="tonexty",
        fillcolor="rgba(0,176,246,0.2)",
        line=dict(width=0),
        name="Prediction Range"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["timestamp"],
        y=forecast_df["predicted_power"],
        mode="lines",
        name="Predicted"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["timestamp"],
        y=forecast_df["actual_power"],
        mode="lines",
        name="Actual"
    ))

    fig.update_layout(template="plotly_dark", height=420, hovermode="x unified")
    fig.update_xaxes(rangeslider_visible=True)

    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ===== ERROR OVER TIME =====
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction Error Over Time")
    fig = px.line(forecast_df, x="timestamp", y="error", template="plotly_dark")
    fig.add_hline(y=0)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ===== ERROR DISTRIBUTION =====
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Error Distribution")
    fig = px.histogram(forecast_df, x="error", nbins=40, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ===== TABLE =====
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Forecast Results")
    st.dataframe(
        forecast_df[["timestamp","actual_power","predicted_power","lower_bound","upper_bound","error"]],
        use_container_width=True
    )
    st.download_button(
        "Download Forecast CSV",
        forecast_df.to_csv(index=False),
        "forecast.csv"
    )
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# PAGE 3 — MODEL INSIGHTS
# ============================================================

elif page == "Model Insights":

    st.header("Model Insights")

    if model is None:
        st.warning("Model not loaded")
        st.stop()

    if "X" not in st.session_state:
        st.warning("Run forecast first")
        st.stop()

    X = st.session_state["X"]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Feature Importance (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    fig = plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Feature Correlation")
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(X.corr())
    plt.colorbar(im)
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)
