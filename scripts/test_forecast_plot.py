import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_csv
from src.data.cleaner import clean_solar_data
from src.features.engineer import create_features
from src.models.train_advanced import train_advanced_models
from src.models.uncertainty import predict_with_uncertainty
from src.visualization.forecast_plot import plot_forecast_with_uncertainty

df = load_csv("data/raw/sample.csv")
clean_df, _ = clean_solar_data(df)
feature_df = create_features(clean_df)

results, models, _ = train_advanced_models(feature_df)
rf_model = models["RandomForest"]

X = feature_df.drop(columns=["timestamp", "power"])
y = feature_df["power"]
timestamps = feature_df["timestamp"]

mean, lower, upper = predict_with_uncertainty(rf_model, X)

plot_forecast_with_uncertainty(
    timestamps,
    y,
    mean,
    lower,
    upper
)
