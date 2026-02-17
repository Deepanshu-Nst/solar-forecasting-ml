import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp",
    "power",
    "temperature",
    "cloud_cover",
    "humidity",
    "wind_speed"
]


def validate_columns(df: pd.DataFrame):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_timestamp(df: pd.DataFrame):
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception:
            raise ValueError("Timestamp column cannot be converted to datetime")


def validate_sorted(df: pd.DataFrame):
    if not df["timestamp"].is_monotonic_increasing:
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)


def validate_duplicates(df: pd.DataFrame):
    if df.duplicated(subset=["timestamp"]).sum() > 0:
        df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)


def validate_missing(df: pd.DataFrame):
    missing_report = df.isna().sum()
    return missing_report


def run_all_validations(df: pd.DataFrame):
    validate_columns(df)
    validate_timestamp(df)
    validate_sorted(df)
    validate_duplicates(df)
    missing = validate_missing(df)
    return df, missing
