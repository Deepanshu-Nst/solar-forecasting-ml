import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ------------------------------------------------
# 1. TIME SERIES TRAIN TEST SPLIT
# ------------------------------------------------
def time_split(df, test_ratio=0.2):
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test


# ------------------------------------------------
# 2. PERSISTENCE BASELINE
# ------------------------------------------------
def persistence_forecast(test_df):
    """
    Predict next value = previous observed value
    """
    return test_df["power_lag_1"].values


# ------------------------------------------------
# 3. PREPARE FEATURE MATRIX
# ------------------------------------------------
def prepare_xy(df):
    X = df.drop(columns=["timestamp", "power"])
    y = df["power"]
    return X, y


# ------------------------------------------------
# 4. METRICS
# ------------------------------------------------
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse}


# ------------------------------------------------
# 5. MAIN TRAINING PIPELINE
# ------------------------------------------------
def train_baseline_models(df):

    train_df, test_df = time_split(df)

    X_train, y_train = prepare_xy(train_df)
    X_test, y_test = prepare_xy(test_df)

    results = {}

    # -----------------------
    # Persistence baseline
    # -----------------------
    y_pred_persist = persistence_forecast(test_df)
    results["Persistence"] = evaluate(y_test, y_pred_persist)

    # -----------------------
    # Linear Regression
    # -----------------------
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results["LinearRegression"] = evaluate(y_test, y_pred_lr)

    # -----------------------
    # Ridge Regression
    # -----------------------
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    results["Ridge"] = evaluate(y_test, y_pred_ridge)

    return results, lr, ridge
