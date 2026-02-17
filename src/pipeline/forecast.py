import pandas as pd

from src.data.loader import load_csv
from src.data.cleaner import clean_solar_data
from src.features.engineer import create_features
from src.models.uncertainty import predict_with_uncertainty


def run_forecast_pipeline(csv_path, trained_model):
    """
    Full end-to-end forecasting pipeline.
    """

    # ------------------------
    # Load and preprocess
    # ------------------------
    df = load_csv(csv_path)
    clean_df, _ = clean_solar_data(df)
    feature_df = create_features(clean_df)

    # ------------------------
    # Prepare prediction data
    # ------------------------
    X = feature_df.drop(columns=["timestamp", "power"])
    timestamps = feature_df["timestamp"]
    actual = feature_df["power"]

    # ------------------------
    # Predict with uncertainty
    # ------------------------
    mean, lower, upper = predict_with_uncertainty(trained_model, X)

    # ------------------------
    # Build result dataframe
    # ------------------------
    results = pd.DataFrame({
        "timestamp": timestamps,
        "actual_power": actual,
        "predicted_power": mean,
        "lower_bound": lower,
        "upper_bound": upper
    })

    return results
