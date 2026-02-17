import shap
import matplotlib.pyplot as plt
import pandas as pd


# ------------------------------------------------
# FEATURE IMPORTANCE (TREE MODELS)
# ------------------------------------------------
def plot_feature_importance(model, feature_names, top_n=10):
    importances = model.feature_importances_
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(8,5))
    plt.barh(imp_df["feature"], imp_df["importance"])
    plt.gca().invert_yaxis()
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.show()


# ------------------------------------------------
# SHAP VALUES
# ------------------------------------------------
def compute_shap_values(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample)
