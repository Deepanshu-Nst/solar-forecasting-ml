import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_csv
from src.data.cleaner import clean_solar_data
from src.features.engineer import create_features
from src.models.train_advanced import train_advanced_models
from src.models.model_io import save_model
from src.pipeline.forecast import run_forecast_pipeline

# train model once
df = load_csv("data/raw/sample.csv")
clean_df, _ = clean_solar_data(df)
feature_df = create_features(clean_df)

results, models, _ = train_advanced_models(feature_df)
rf_model = models["RandomForest"]

# save model
save_model(rf_model, "models/random_forest.pkl")

# run pipeline
forecast_results = run_forecast_pipeline(
    "data/raw/sample.csv",
    rf_model
)

print("\nFORECAST PIPELINE OUTPUT")
print(forecast_results.head())
