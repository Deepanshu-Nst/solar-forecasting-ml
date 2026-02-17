import streamlit as st
import pandas as pd

from src.data.cleaner import clean_solar_data
from src.features.engineer import create_features
from src.models.model_io import load_model
from src.models.uncertainty import predict_with_uncertainty
from src.visualization.forecast_plot import plot_forecast_with_uncertainty

import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


st.set_page_config(page_title="Solar Forecast AI", layout="wide")

st.title("Solar Power Forecasting System")
st.write("Upload solar data to generate predictions with uncertainty.")

# -------------------------
# Load trained model
# -------------------------
model = load_model("models/random_forest.pkl")

# -------------------------
# File upload
# -------------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    st.subheader("Raw Data")
    st.dataframe(df.head())

    # -------------------------
    # Pipeline
    # -------------------------
    clean_df, _ = clean_solar_data(df)
    feature_df = create_features(clean_df)

    X = feature_df.drop(columns=["timestamp", "power"])
    y = feature_df["power"]
    timestamps = feature_df["timestamp"]

    mean, lower, upper = predict_with_uncertainty(model, X)

    mae = mean_absolute_error(y, mean)
    rmse = np.sqrt(mean_squared_error(y, mean))

    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")

    st.subheader("Top Feature Importance")

    importances = model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values("importance", ascending=False).head(10)

    fig, ax = plt.subplots()
    ax.barh(feat_imp["feature"], feat_imp["importance"])
    ax.invert_yaxis()
    ax.set_title("Top Influential Features")

    st.pyplot(fig)

    st.subheader("Model Explanation (SHAP)")

    sample_X = X.sample(min(100, len(X)))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_X)

    fig_shap = plt.figure()
    shap.summary_plot(shap_values, sample_X, show=False)

    st.pyplot(fig_shap)



    # -------------------------
    # Results table
    # -------------------------
    results = pd.DataFrame({
        "timestamp": timestamps,
        "actual": y,
        "predicted": mean,
        "lower": lower,
        "upper": upper
    })

    st.subheader("Forecast Results")
    st.dataframe(results.head())

    # -------------------------
    # Plot
    # -------------------------
    st.subheader("Forecast Visualization")
    plot_forecast_with_uncertainty(
        timestamps,
        y,
        mean,
        lower,
        upper
    )

    # -------------------------
    # Download
    # -------------------------
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Forecast CSV",
        csv,
        "forecast_results.csv",
        "text/csv"
    )
