import pandas as pd
from src.models.uncertainty import predict_with_uncertainty


def run_forecast_pipeline(feature_df, trained_model):
    """
    Forecast pipeline using preprocessed feature dataframe.
    Assumes cleaning + feature engineering already done.
    """

    df = feature_df.copy()

    # ------------------------
    # Prepare prediction data
    # ------------------------
    X = df.drop(columns=["timestamp", "power"], errors="ignore")
    timestamps = df["timestamp"]
    actual = df["power"]

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

    return results.reset_index(drop=True)
