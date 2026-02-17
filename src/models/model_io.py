import joblib


def save_model(model, path):
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path):
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model
