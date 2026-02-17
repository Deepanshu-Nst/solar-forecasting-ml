import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

from src.data.cleaner import clean_solar_data


# =========================================
# LOAD DATA
# =========================================
df = pd.read_csv("data/sample.csv", parse_dates=["timestamp"])

df, _ = clean_solar_data(df)


# =========================================
# FEATURE ENGINEERING (SAME AS APP)
# =========================================
df["hour"] = df["timestamp"].dt.hour
df["day_of_year"] = df["timestamp"].dt.dayofyear

df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

df["doy_sin"] = np.sin(2*np.pi*df["day_of_year"]/365)
df["doy_cos"] = np.cos(2*np.pi*df["day_of_year"]/365)

df["power_lag_1"] = df["power"].shift(1)
df["power_lag_24"] = df["power"].shift(24)

df.fillna(0, inplace=True)


# =========================================
# FEATURES
# =========================================
features = [
    "temperature",
    "cloud_cover",
    "humidity",
    "wind_speed",
    "hour",
    "day_of_year",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
    "power_lag_1",
    "power_lag_24"
]

X = df[features]
y = df["power"]


# =========================================
# TRAIN MODEL
# =========================================
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)


# =========================================
# SAVE MODEL
# =========================================
joblib.dump(model, "models/random_forest.pkl")

print("✅ Model retrained and saved successfully")
