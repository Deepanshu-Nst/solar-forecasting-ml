import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_csv
from src.data.cleaner import clean_solar_data
from src.features.engineer import create_features
from src.models.train_baseline import train_baseline_models

# pipeline
df = load_csv("data/raw/sample.csv")
clean_df, _ = clean_solar_data(df)
feature_df = create_features(clean_df)

results, lr_model, ridge_model = train_baseline_models(feature_df)

print("\nMODEL PERFORMANCE")
for model, metrics in results.items():
    print(model, metrics)
