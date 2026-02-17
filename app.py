# ============================================================
# ☀️ SOLAR POWER FORECASTING SYSTEM — STABLE VERSION
# ============================================================

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


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(page_title="Solar Power Forecasting", layout="wide")
st.title("☀️ Solar Power Forecasting System")

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

page = st.sidebar.radio(
    "Navigation",
    ["Upload Data", "Forecast Dashboard", "Model Insights"]
)


# ============================================================
# PAGE 1 — UPLOAD DATA
# ============================================================

if page == "Upload Data":

    st.header("Upload Solar Data")

    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:

        raw_df = pd.read_csv(uploaded_file, parse_dates=["timestamp"])

        st.subheader("Raw Data")
        st.dataframe(raw_df.head())

        # ---- SAFE CLEAN ----
        cleaned = clean_solar_data(raw_df)

        if isinstance(cleaned, tuple):
            cleaned_df = cleaned[0]
        else:
            cleaned_df = cleaned

        st.subheader("Cleaned Data")
        st.dataframe(cleaned_df.head())

        st.session_state["cleaned_df"] = cleaned_df
        st.success("Data uploaded successfully. Go to Forecast Dashboard.")



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

    # ---------------- FEATURE ENGINEERING ----------------
    feature_df = create_features(cleaned_df.copy())

    # TARGET
    y_true = feature_df["power"]

    # FEATURES FOR MODEL
    X = feature_df.drop(columns=["power", "timestamp"], errors="ignore")

    # STORE FOR SHAP
    st.session_state["X"] = X

    # ---------------- PREDICT ----------------
    y_pred = model.predict(X)

    forecast_df = feature_df.copy()
    forecast_df["actual_power"] = y_true
    forecast_df["predicted_power"] = y_pred
    forecast_df["error"] = y_true - y_pred

    st.session_state["forecast_df"] = forecast_df

    # =====================================================
    # METRICS
    # =====================================================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("MAE", f"{mean_absolute_error(y_true, y_pred):.3f}")
    col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_true, y_pred)):.3f}")
    col3.metric("MAPE", f"{np.mean(np.abs((y_true-y_pred)/(y_true+1e-6)))*100:.2f}%")
    col4.metric("R²", f"{r2_score(y_true, y_pred):.3f}")

    st.divider()

    # =====================================================
    # ACTUAL VS PREDICTED
    # =====================================================
    st.subheader("Actual vs Predicted")

    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(forecast_df["timestamp"], y_true, label="Actual")
    ax.plot(forecast_df["timestamp"], y_pred, label="Predicted")
    ax.legend()
    st.pyplot(fig)

    # =====================================================
    # ERROR OVER TIME
    # =====================================================
    st.subheader("Prediction Error Over Time")

    fig, ax = plt.subplots(figsize=(12,3))
    ax.plot(forecast_df["timestamp"], forecast_df["error"])
    ax.axhline(0, linestyle="--")
    st.pyplot(fig)

    # =====================================================
    # ERROR DISTRIBUTION
    # =====================================================
    st.subheader("Error Distribution")

    fig, ax = plt.subplots()
    ax.hist(forecast_df["error"], bins=40)
    st.pyplot(fig)

    # =====================================================
    # TABLE
    # =====================================================
    st.subheader("Forecast Table")
    st.dataframe(
        forecast_df[["timestamp","actual_power","predicted_power","error"]]
    )

    st.download_button(
        "Download Forecast CSV",
        forecast_df.to_csv(index=False),
        "forecast.csv"
    )



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

    st.subheader("Feature Importance (SHAP)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    fig = plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig)

    st.subheader("Feature Correlation")

    fig, ax = plt.subplots(figsize=(8,6))
    corr = X.corr()
    im = ax.imshow(corr)
    plt.colorbar(im)
    st.pyplot(fig)
