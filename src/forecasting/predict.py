import pandas as pd
import joblib


# ------------------------------------------------
# LOAD TRAINED MODEL
# ------------------------------------------------
def load_model(model_path="models/random_forest.pkl"):
    return joblib.load(model_path)


# ------------------------------------------------
# GENERATE FORECAST
# ------------------------------------------------
def generate_forecast(df, model):

    feature_cols = [
        col for col in df.columns
        if col not in ["timestamp", "power"]
    ]

    X = df[feature_cols]
    preds = model.predict(X)

    # Simple uncertainty (tree spread approximation)
    if hasattr(model, "estimators_"):
        all_preds = []
        for tree in model.estimators_:
            all_preds.append(tree.predict(X))

        all_preds = pd.DataFrame(all_preds)
        lower = all_preds.quantile(0.1)
        upper = all_preds.quantile(0.9)
    else:
        lower = preds * 0.9
        upper = preds * 1.1

    result = pd.DataFrame({
        "timestamp": df["timestamp"],
        "actual": df.get("power"),
        "predicted": preds,
        "lower": lower.values,
        "upper": upper.values
    })

    return result
