import pandas as pd
import numpy as np


# ------------------------------------------------
# 1. CYCLICAL TIME ENCODING
# ------------------------------------------------
def add_time_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_year"] = df["timestamp"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    return df


# ------------------------------------------------
# 2. LAG FEATURES
# ------------------------------------------------
def add_lag_features(df, lags):
    for lag in lags:
        df[f"power_lag_{lag}"] = df["power"].shift(lag)
    return df


# ------------------------------------------------
# 3. ROLLING STATISTICS
# ------------------------------------------------
def add_rolling_features(df, rolling_windows):
    for w in rolling_windows:
        df[f"power_roll_mean_{w}"] = df["power"].shift(1).rolling(w).mean()
        df[f"power_roll_std_{w}"] = df["power"].shift(1).rolling(w).std()

    return df


# ------------------------------------------------
# 4. WEATHER INTERACTION FEATURES
# ------------------------------------------------
def add_weather_interactions(df):
    df["cloud_effect"] = 1 - (df["cloud_cover"] / 100)
    df["temp_cloud_interaction"] = df["temperature"] * df["cloud_effect"]
    return df


# ------------------------------------------------
# 5. CLEAR SKY BASELINE PROXY
# ------------------------------------------------
def add_clear_sky_proxy(df):
    solar_peak = np.maximum(0, np.sin(np.pi * (df["hour"] - 6) / 12))
    df["clear_sky_estimate"] = solar_peak
    return df


# ------------------------------------------------
# MAIN FEATURE PIPELINE
# ------------------------------------------------
def create_features(
    df,
    lags=[1, 2, 3],
    rolling_windows=[3, 6]
):

    df = add_time_features(df)
    df = add_lag_features(df, lags)
    df = add_rolling_features(df, rolling_windows)
    df = add_weather_interactions(df)
    df = add_clear_sky_proxy(df)

    df = df.dropna().reset_index(drop=True)
    return df
