import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data.cleaner import clean_solar_data
from src.features.engineer import create_features
from src.models.model_io import load_model
from src.models.uncertainty import predict_with_uncertainty
from src.visualization.forecast_plot import plot_forecast_with_uncertainty


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Solar Forecast AI",
    layout="wide"
)

st.title("🌞 Solar Power Forecasting Dashboard")
st.caption("Predict solar output with uncertainty and model insights")


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
model = load_model("models/random_forest.pkl")


# ---------------------------------------------------
# FILE UPLOAD SECTION
# ---------------------------------------------------
st.header("📤 Upload Solar Data")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])


if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    st.divider()

    # ---------------------------------------------------
    # PREPROCESSING PIPELINE
    # ---------------------------------------------------
    clean_df, _ = clean_solar_data(df)
    feature_df = create_features(clean_df)

    X = feature_df.drop(columns=["timestamp", "power"])
    y = feature_df["power"]
    timestamps = feature_df["timestamp"]

    mean, lower, upper = predict_with_uncertainty(model, X)

    # ---------------------------------------------------
    # MODEL PERFORMANCE METRICS
    # ---------------------------------------------------
    st.header("📊 Model Performance")

    mae = mean_absolute_error(y, mean)
    rmse = np.sqrt(mean_squared_error(y, mean))

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("Avg Prediction", f"{mean.mean():.2f}")

    st.divider()

    # ---------------------------------------------------
    # FORECAST VISUALIZATION
    # ---------------------------------------------------
    st.header("📈 Solar Forecast")

    plot_forecast_with_uncertainty(
        timestamps,
        y,
        mean,
        lower,
        upper
    )

    st.divider()

    # ---------------------------------------------------
    # FEATURE IMPORTANCE
    # ---------------------------------------------------
    st.header("⭐ What influences solar output most?")

    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values("importance", ascending=False).head(10)

    fig, ax = plt.subplots()
    ax.barh(feat_imp["feature"], feat_imp["importance"])
    ax.invert_yaxis()
    ax.set_title("Top Influential Features")

    st.pyplot(fig, use_container_width=True)

    st.divider()

    # ---------------------------------------------------
    # SHAP EXPLANATION
    # ---------------------------------------------------
    st.header("🧠 Model Explanation")

    sample_X = X.sample(min(100, len(X)))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_X)

    fig_shap = plt.figure()
    shap.summary_plot(shap_values, sample_X, show=False)
    st.pyplot(fig_shap, use_container_width=True)

    st.divider()

    # ---------------------------------------------------
    # DOWNLOAD RESULTS
    # ---------------------------------------------------
    st.header("⬇ Download Forecast Results")

    results = pd.DataFrame({
        "timestamp": timestamps,
        "actual": y,
        "predicted": mean,
        "lower": lower,
        "upper": upper
    })

    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Forecast CSV",
        csv,
        "forecast_results.csv",
        "text/csv"
    )
