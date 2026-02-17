import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =====================================================
# IMPORT YOUR PROJECT MODULES
# =====================================================
from src.data.cleaner import clean_solar_data


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Solar Power Forecasting System",
    layout="wide"
)


# =====================================================
# LOAD TRAINED MODEL (ROBUST PATH)
# =====================================================
def load_trained_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "random_forest.pkl")

    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


model = load_trained_model()


# =====================================================
# FEATURE ENGINEERING (SAFE MINIMAL)
# =====================================================
def create_time_features(df):
    df = df.copy()

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_year"] = df["timestamp"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    return df


# =====================================================
# FORECAST FUNCTION
# =====================================================
def generate_forecast(df, model):
    df = df.copy()

    df = create_time_features(df)

    feature_cols = [c for c in df.columns if c not in ["timestamp", "power"]]
    X = df[feature_cols]

    # Align with trained model features
    if model is not None and hasattr(model, "feature_names_in_"):
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    if model is None:
        # naive forecast
        preds = df["power"].shift(1).fillna(method="bfill")
    else:
        preds = model.predict(X)

    df["predicted_power"] = preds
    df["actual_power"] = df["power"]

    residuals = df["actual_power"] - df["predicted_power"]

    df["lower_bound"] = df["predicted_power"] - residuals.std()
    df["upper_bound"] = df["predicted_power"] + residuals.std()

    return df


# =====================================================
# SESSION STATE
# =====================================================
if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = None


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Upload Data", "Forecast Dashboard", "Model Insights"]
)


# =====================================================
# TITLE
# =====================================================
st.title("☀ Solar Power Forecasting System")


# =====================================================
# PAGE 1 — UPLOAD DATA
# =====================================================
if page == "Upload Data":
    st.header("Upload Solar Data")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        cleaned_df, report = clean_solar_data(df)

        st.success("Data cleaned successfully")
        st.dataframe(cleaned_df.head())

        # Generate forecast and store
        forecast_df = generate_forecast(cleaned_df, model)
        st.session_state.forecast_df = forecast_df

        st.success("Forecast generated. Go to dashboard.")


# =====================================================
# PAGE 2 — FORECAST DASHBOARD
# =====================================================
if page == "Forecast Dashboard":
    st.header("Forecast Dashboard")

    if model is None:
        st.warning("No trained model found. Using naive forecast.")
    else:
        st.success("Trained model loaded successfully.")

    forecast_df = st.session_state.forecast_df

    if forecast_df is None:
        st.info("Upload data first.")
    else:
        mae = np.mean(np.abs(forecast_df["actual_power"] - forecast_df["predicted_power"]))
        rmse = np.sqrt(np.mean((forecast_df["actual_power"] - forecast_df["predicted_power"])**2))

        col1, col2 = st.columns(2)
        col1.metric("MAE", round(mae, 3))
        col2.metric("RMSE", round(rmse, 3))

        st.subheader("Actual vs Predicted")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(forecast_df["timestamp"], forecast_df["actual_power"], label="Actual")
        ax.plot(forecast_df["timestamp"], forecast_df["predicted_power"], label="Predicted")
        ax.legend()
        ax.set_ylabel("Power")
        st.pyplot(fig)

        st.download_button(
            "Download Forecast CSV",
            forecast_df.to_csv(index=False),
            file_name="forecast.csv"
        )


# =====================================================
# PAGE 3 — MODEL INSIGHTS
# =====================================================
if page == "Model Insights":
    st.header("Model Insights")

    forecast_df = st.session_state.forecast_df

    if forecast_df is None:
        st.warning("Run forecast first.")
    else:
        residuals = forecast_df["actual_power"] - forecast_df["predicted_power"]

        st.subheader("Prediction Error Distribution")
        fig, ax = plt.subplots()
        ax.hist(residuals, bins=30)
        st.pyplot(fig)

        st.subheader("Residuals Over Time")
        fig, ax = plt.subplots()
        ax.plot(forecast_df["timestamp"], residuals)
        ax.axhline(0, linestyle="--")
        st.pyplot(fig)

        st.subheader("Explain Individual Prediction")

        idx = st.slider(
            "Select row",
            0,
            len(forecast_df) - 1,
            0
        )

        row = forecast_df.iloc[idx]
        st.write("Prediction:", row["predicted_power"])

        # SHAP (optional)
        try:
            import shap

            if model is not None:
                feature_cols = [c for c in forecast_df.columns if c not in [
                    "timestamp",
                    "actual_power",
                    "predicted_power",
                    "lower_bound",
                    "upper_bound"
                ]]

                X = forecast_df[feature_cols]

                if hasattr(model, "feature_names_in_"):
                    X = X.reindex(columns=model.feature_names_in_, fill_value=0)

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X.iloc[[idx]])

                st.subheader("SHAP Explanation")
                shap.initjs()
                shap.force_plot(
                    explainer.expected_value,
                    shap_values[0],
                    X.iloc[idx],
                    matplotlib=True
                )
                st.pyplot(plt.gcf())

        except Exception:
            st.info("SHAP not available. Install with: pip install shap ipython")
