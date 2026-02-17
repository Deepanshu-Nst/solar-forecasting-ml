import pandas as pd
from src.models.uncertainty import predict_with_uncertainty


def run_forecast_pipeline(feature_df, trained_model):
    """
    Forecast pipeline using preprocessed feature dataframe.
    Enforces model feature schema.
    """

    df = feature_df.copy()

    # ------------------------
    # Prepare prediction data
    # ------------------------
    X = df.drop(columns=["timestamp", "power"], errors="ignore")

    # ==============================
    # FEATURE SCHEMA ENFORCEMENT ⭐
    # ==============================
    if hasattr(trained_model, "feature_names_in_"):

        expected = list(trained_model.feature_names_in_)

        missing = [c for c in expected if c not in X.columns]
        extra = [c for c in X.columns if c not in expected]

        if missing:
            raise ValueError(f"Missing required model features: {missing}")

        # reorder and drop extras
        X = X[expected]

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
