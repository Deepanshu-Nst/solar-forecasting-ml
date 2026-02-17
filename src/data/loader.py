import pandas as pd
from pathlib import Path
from .validator import run_all_validations


def load_csv(filepath: str, verbose=True):
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.suffix != ".csv":
        raise ValueError("Only CSV files are supported")

    df = pd.read_csv(filepath)

    if verbose:
        print(f"Loaded data shape: {df.shape}")

    df, missing = run_all_validations(df)

    if verbose:
        print("Missing values:")
        print(missing)

    return df
