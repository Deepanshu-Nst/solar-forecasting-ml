import numpy as np


def predict_with_uncertainty(rf_model, X, lower=5, upper=95):

    all_preds = np.array([
        tree.predict(X.values) for tree in rf_model.estimators_
    ])

    mean_pred = np.mean(all_preds, axis=0)
    lower_bound = np.percentile(all_preds, lower, axis=0)
    upper_bound = np.percentile(all_preds, upper, axis=0)

    return mean_pred, lower_bound, upper_bound

