import pandas as pd
import numpy as np

np.random.seed(42)

# Generate 7 days hourly data
timestamps = pd.date_range("2024-01-01", periods=24*7, freq="h")

df = pd.DataFrame({"timestamp": timestamps})

# Simulate solar pattern
hour = df["timestamp"].dt.hour

solar_curve = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))

df["power"] = solar_curve * 100 + np.random.normal(0, 5, len(df))
df["power"] = df["power"].clip(lower=0)

df["temperature"] = 20 + 5 * np.sin(2*np.pi*hour/24) + np.random.normal(0,1,len(df))
df["cloud_cover"] = np.random.uniform(20, 80, len(df))
df["humidity"] = np.random.uniform(40, 90, len(df))
df["wind_speed"] = np.random.uniform(1, 6, len(df))

df.to_csv("data/raw/sample.csv", index=False)

print("Synthetic solar dataset created")
print(df.head())
