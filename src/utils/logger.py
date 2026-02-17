import pandas as pd
from pathlib import Path
from datetime import datetime


LOG_PATH = Path("logs/prediction_log.csv")


def log_forecast_run(
    forecast_df,
    model_version="v1",
    mae=None,
    rmse=None
):
    """
    Appends metadata about a forecast run to log file.
    """

    LOG_PATH.parent.mkdir(exist_ok=True)

    row = {
        "prediction_time": datetime.utcnow(),
        "model_version": model_version,
        "forecast_start": forecast_df["timestamp"].min(),
        "forecast_end": forecast_df["timestamp"].max(),
        "num_predictions": len(forecast_df),
        "mae": mae,
        "rmse": rmse
    }

    row_df = pd.DataFrame([row])

    if LOG_PATH.exists():
        existing = pd.read_csv(LOG_PATH)
        updated = pd.concat([existing, row_df], ignore_index=True)
    else:
        updated = row_df

    updated.to_csv(LOG_PATH, index=False)
