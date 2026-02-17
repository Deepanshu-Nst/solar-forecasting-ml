import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor


# ------------------------------------------------
# METRICS
# ------------------------------------------------
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse}


# ------------------------------------------------
# TIME SPLIT
# ------------------------------------------------
def time_split(df, test_ratio=0.2):
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


# ------------------------------------------------
# PREPARE DATA
# ------------------------------------------------
def prepare_xy(df):
    X = df.drop(columns=["timestamp", "power"])
    y = df["power"]
    return X, y


# ------------------------------------------------
# TRAIN ADVANCED MODELS
# ------------------------------------------------
def train_advanced_models(df):

    train_df, test_df = time_split(df)

    X_train, y_train = prepare_xy(train_df)
    X_test, y_test = prepare_xy(test_df)

    results = {}
    models = {}

    # -----------------------
    # Random Forest
    # -----------------------
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results["RandomForest"] = evaluate(y_test, y_pred_rf)
    models["RandomForest"] = rf

    # -----------------------
    # XGBoost
    # -----------------------
    xgb = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    results["XGBoost"] = evaluate(y_test, y_pred_xgb)
    models["XGBoost"] = xgb

    return results, models, X_train.columns
