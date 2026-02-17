import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_csv
from src.data.cleaner import clean_solar_data
from src.features.engineer import create_features

df = load_csv("data/raw/sample.csv")
clean_df, report = clean_solar_data(df)
feature_df = create_features(clean_df)

print("\nFeatures created:")
print(feature_df.columns)
print(feature_df.head())
print("Feature engineering working")
