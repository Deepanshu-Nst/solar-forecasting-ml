import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_csv
from src.data.cleaner import clean_solar_data
from src.features.engineer import create_features
from src.models.train_advanced import train_advanced_models
from src.models.uncertainty import predict_with_uncertainty

df = load_csv("data/raw/sample.csv")
clean_df, _ = clean_solar_data(df)
feature_df = create_features(clean_df)

results, models, feature_names = train_advanced_models(feature_df)

rf_model = models["RandomForest"]
X = feature_df.drop(columns=["timestamp", "power"])

mean, lower, upper = predict_with_uncertainty(rf_model, X)

print("\nPrediction interval example:")
for i in range(5):
    print(f"Prediction: {mean[i]:.2f} | Range: [{lower[i]:.2f}, {upper[i]:.2f}]")
