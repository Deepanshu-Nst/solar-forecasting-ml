import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_csv
from src.data.cleaner import clean_solar_data
from src.features.engineer import create_features
from src.models.train_advanced import train_advanced_models
from src.evaluation.explainability import plot_feature_importance, compute_shap_values

df = load_csv("data/raw/sample.csv")
clean_df, _ = clean_solar_data(df)
feature_df = create_features(clean_df)

results, models, feature_names = train_advanced_models(feature_df)

rf_model = models["RandomForest"]

X = feature_df.drop(columns=["timestamp", "power"])

print("Showing feature importance...")
plot_feature_importance(rf_model, feature_names)

print("Computing SHAP values...")
compute_shap_values(rf_model, X.sample(50))
