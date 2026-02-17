import pandas as pd
import numpy as np


# ------------------------------------------------
# 1. TIME INDEX SETUP
# ------------------------------------------------
def ensure_regular_time_index(df, freq="1h"):
    """
    Ensures continuous time index.
    Missing timestamps are inserted.
    """
    df = df.set_index("timestamp")
    full_range = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(full_range)
    df.index.name = "timestamp"
    return df.reset_index()


# ------------------------------------------------
# 2. MISSING VALUE IMPUTATION
# ------------------------------------------------
def impute_missing(df):
    """
    Time-based interpolation for numeric columns.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="time")
    return df


# ------------------------------------------------
# 3. OUTLIER DETECTION (ROBUST Z SCORE)
# ------------------------------------------------
def remove_outliers(df, col="power", threshold=4):
    """
    Removes extreme spikes using MAD-based z-score.
    """
    series = df[col]
    median = np.median(series)
    mad = np.median(np.abs(series - median))

    if mad == 0:
        return df

    z_score = 0.6745 * (series - median) / mad
    df.loc[np.abs(z_score) > threshold, col] = np.nan
    return df


# ------------------------------------------------
# 4. NIGHTTIME SOLAR PHYSICS RULE
# ------------------------------------------------
def enforce_night_zero(df):
    """
    Solar generation must be zero at night.
    We approximate night = 7 PM to 5 AM.
    """
    hours = df["timestamp"].dt.hour
    night_mask = (hours >= 19) | (hours <= 5)
    df.loc[night_mask, "power"] = 0
    return df


# ------------------------------------------------
# 5. OPTIONAL SMOOTHING
# ------------------------------------------------
def smooth_signal(df, col="power", window=3):
    """
    Rolling median smoothing.
    """
    df[col] = df[col].rolling(window, center=True, min_periods=1).median()
    return df


# ------------------------------------------------
# MAIN CLEANING PIPELINE
# ------------------------------------------------
def clean_solar_data(df):
    report = {}

    original_rows = len(df)

    # Ensure regular timestamps
    df = ensure_regular_time_index(df)
    report["rows_after_reindex"] = len(df)

    # Remove extreme outliers
    df = remove_outliers(df)

    # Impute missing
    df = df.set_index("timestamp")
    df = impute_missing(df)
    df = df.reset_index()

    # Night physics rule
    df = enforce_night_zero(df)

    # Smooth noise
    df = smooth_signal(df)

    report["rows_original"] = original_rows
    report["rows_final"] = len(df)

    return df, report

# Alias for compatibility
clean_data = clean_solar_data