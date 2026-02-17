import pandas as pd


def load_csv(uploaded_file):
    """
    Loads CSV from Streamlit UploadedFile or file path.
    """

    # Streamlit UploadedFile
    if hasattr(uploaded_file, "read"):
        df = pd.read_csv(uploaded_file)

    # Normal path
    else:
        df = pd.read_csv(uploaded_file)

    # Ensure timestamp exists
    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain 'timestamp' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df
