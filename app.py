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
from src.models.uncertainty import predict_with_uncertainty

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Solar Power Forecasting API", layout="wide", page_icon="⚡")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0b1220 0%, #111827 100%);
    color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1200px;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.5) !important;
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,255,255,0.05);
}

/* Metrics row styling */
div[data-testid="metric-container"] {
    background: rgba(30, 41, 59, 0.4);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border 0.2s ease;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    border: 1px solid rgba(56, 189, 248, 0.3);
    box-shadow: 0 8px 40px rgba(0, 0, 0, 0.3);
}
[data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #38bdf8 !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
    color: white;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    box-shadow: 0 0 20px rgba(56, 189, 248, 0.5);
    transform: translateY(-2px);
    color: white;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.05);
    background: rgba(15, 23, 42, 0.5);
}

/* Header Text */
.gradient-header {
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0px;
}
.subtitle {
    color: #94a3b8;
    font-size: 1.2rem;
    margin-top: 5px;
}

/* Subheaders */
h2, h3 {
    font-weight: 600 !important;
    color: #f1f5f9;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================
MODEL_PATH = Path("models/random_forest.pkl")

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
<div style='text-align:center;'>
    <h2 style='background: linear-gradient(90deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2rem; margin-top: 2rem;'>⚡ Solar AI</h2>
</div>
<hr style='border-color: rgba(255,255,255,0.1); margin-bottom: 20px;'>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    ["📤 Upload Data", "📊 Forecast Dashboard", "🧠 Model Insights", "⚙️ Settings"]
)

st.sidebar.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
st.sidebar.info("Enterprise ML Model: v2.4.1\n\nStatus: 🟢 Online")


# ============================================================
# MAIN HEADER
# ============================================================
st.markdown("""
<div style='text-align: center; margin-bottom: 3rem;'>
    <h1 class='gradient-header'>Solar Power Forecasting System</h1>
    <p class='subtitle'>AI-powered solar generation prediction</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# PAGE 1 — UPLOAD DATA
# ============================================================
if page == "📤 Upload Data":

    with st.container():
        st.subheader("Data Intake & Validation")
        st.markdown("<p style='color:#94a3b8;'>Injest your telemetry payload to generate forecasts or test our architecture with sample data.</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload CSV Payload", type=["csv"])

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            use_sample = st.button("🚀 Use Sample Data Instead", use_container_width=True)
        with col2:
            with open("data/sample.csv", "rb") as f:
                st.download_button("📥 Download sample.csv", f, file_name="sample.csv", use_container_width=True)

        if uploaded_file is not None or use_sample:
            with st.spinner("Running forecasting pipeline..."):
                try:
                    if use_sample:
                        raw_df = pd.read_csv("data/sample.csv", parse_dates=["timestamp"])
                    else:
                        raw_df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])

                    from src.data.validator import run_all_validations
                    raw_df, missing_report = run_all_validations(raw_df)

                    st.success("✅ Data validation passed successfully")

                except Exception as e:
                    st.error("Invalid input data format")
                    st.exception(e)
                    st.stop()

                st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
                
                with st.container():
                    st.markdown("#### Raw Telemetry Payload")
                    st.dataframe(raw_df.head(), use_container_width=True, height=200)

                cleaned_df, report = clean_solar_data(raw_df)

                with st.container():
                    st.markdown("#### Cleaned Features Preview")
                    st.dataframe(cleaned_df.head(), use_container_width=True, height=200)

                feature_df = create_features(cleaned_df)
                forecast_df = run_forecast_pipeline(feature_df, model)

                st.session_state["raw_df"] = raw_df
                st.session_state["cleaned_df"] = cleaned_df
                st.session_state["X"] = feature_df
                st.session_state["forecast_df"] = forecast_df

                st.success("🎉 Forecast generated successfully. Please navigate to the Dashboard to view results.")


# ============================================================
# PAGE 2 — FORECAST DASHBOARD
# ============================================================
elif page == "📊 Forecast Dashboard":

    if model is None:
        st.error("Model file not found: models/random_forest.pkl")
        st.stop()

    if "cleaned_df" not in st.session_state:
        st.warning("Please upload data on the Upload Data page first.")
        st.stop()

    with st.spinner("Compiling UI metrics..."):
        cleaned_df = st.session_state["cleaned_df"]
        feature_df = create_features(cleaned_df.copy())
        
        y_true = feature_df["power"]
        X = feature_df.drop(columns=["power", "timestamp"], errors="ignore")
        st.session_state["X"] = X

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

    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{mae_value:.3f}")
        col2.metric("RMSE", f"{rmse_value:.3f}")
        col3.metric("R² Score", f"{r2_value:.3f}")
        col4.metric("Samples", f"{len(forecast_df)}")

    st.markdown("<br><br>", unsafe_allow_html=True)

    with st.container():
        st.subheader("⚡ Actual vs Predicted Power (Confidence Interval)")
        st.markdown("<p style='color:#94a3b8; margin-top:-10px; margin-bottom:15px;'>Real-time generation overlay against AI estimates.</p>", unsafe_allow_html=True)
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=forecast_df["timestamp"], y=forecast_df["upper_bound"],
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["timestamp"], y=forecast_df["lower_bound"],
            fill="tonexty", fillcolor="rgba(56, 189, 248, 0.15)",
            line=dict(width=0), name="95% Confidence Base"
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["timestamp"], y=forecast_df["predicted_power"],
            mode="lines", name="Predicted Generation",
            line=dict(color="#38bdf8", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["timestamp"], y=forecast_df["actual_power"],
            mode="lines", name="Actual Telemetry",
            line=dict(color="#f43f5e", width=2, dash="dot")
        ))
        fig.update_layout(
            template="plotly_dark", 
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=450, 
            hovermode="x unified",
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    colA, colB = st.columns(2)
    
    with colA:
        with st.container():
            st.subheader("📉 Prediction Error Over Time")
            fig2 = px.line(forecast_df, x="timestamp", y="error", template="plotly_dark")
            fig2.add_hline(y=0, line_dash="dash", line_color="#f43f5e")
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=20))
            st.plotly_chart(fig2, use_container_width=True)

    with colB:
        with st.container():
            st.subheader("📊 Error Distribution")
            fig3 = px.histogram(forecast_df, x="error", nbins=40, template="plotly_dark", color_discrete_sequence=['#818cf8'])
            fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(t=20))
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container():
        st.subheader("📋 Forecast Operations Table")
        st.dataframe(
            forecast_df[["timestamp","actual_power","predicted_power","lower_bound","upper_bound","error"]],
            use_container_width=True,
            height=300
        )
        st.download_button(
            "⬇️ Export Forecast Results CSV",
            forecast_df.to_csv(index=False),
            "forecast.csv",
            use_container_width=True
        )


# ============================================================
# PAGE 3 — MODEL INSIGHTS
# ============================================================
elif page == "🧠 Model Insights":

    st.header("Deep Model Diagnostics")
    st.markdown("<p style='color:#94a3b8; margin-bottom: 2rem;'>Understand the drivers behind AI predictions.</p>", unsafe_allow_html=True)

    if model is None:
        st.warning("Model not loaded in environment")
        st.stop()
    if "X" not in st.session_state:
        st.warning("Please run a forecast first to generate spatial features.")
        st.stop()

    X = st.session_state["X"]

    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.subheader("🎯 Feature Importance Vectors (SHAP)")
            with st.spinner("Generating SHAP values..."):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                fig_shap = plt.figure(figsize=(10, 8))
                plt.style.use('dark_background')
                fig_shap.patch.set_facecolor('#111827')
                ax_shap = fig_shap.gca()
                ax_shap.set_facecolor('#111827')
                
                shap.summary_plot(shap_values, X, show=False)
                st.pyplot(fig_shap)

    with col2:
        with st.container():
            st.subheader("🔗 Feature Correlation Matrix")
            fig_corr, ax_corr = plt.subplots(figsize=(10,8))
            fig_corr.patch.set_facecolor('#111827')
            ax_corr.set_facecolor('#111827')
            corr = X.corr()
            im = ax_corr.imshow(corr, cmap="coolwarm")
            plt.colorbar(im)
            ax_corr.set_xticks(np.arange(len(corr.columns)))
            ax_corr.set_yticks(np.arange(len(corr.columns)))
            ax_corr.set_xticklabels(corr.columns, rotation=45, ha="right", color='white')
            ax_corr.set_yticklabels(corr.columns, color='white')
            st.pyplot(fig_corr)


# ============================================================
# PAGE 4 — SETTINGS
# ============================================================
elif page == "⚙️ Settings":
    
    with st.container():
        st.subheader("Platform Preferences")
        st.markdown("<p style='color:#94a3b8;'>Adjust application parameters.</p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.toggle("Enable Live Telemetry Mode", value=False)
        st.toggle("Deep Uncertainty Sampling", value=True)
        st.selectbox("Forecast Horizon", ["24 Hours", "48 Hours", "7 Days", "30 Days"])
        st.slider("Anomaly Detection Threshold (%)", 5, 50, 15)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("💾 Save Configuration", use_container_width=True)
