from pathlib import Path

import pandas as pd


# Google Data is CSV with headers Date, Open, High, Low, Close, Volume


def load_google_stock_data(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["Volume"] = df["Volume"].str.replace(",", "")  # Remove commas
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )  # Convert to numeric, invalid data becomes NaN
    df = df.dropna()  # Drop rows with NaN values

    normalization_params = {
        "Open": (df["Open"].min(), df["Open"].max()),
        "High": (df["High"].min(), df["High"].max()),
        "Low": (df["Low"].min(), df["Low"].max()),
        "Close": (df["Close"].min(), df["Close"].max()),
        "Volume": (df["Volume"].min(), df["Volume"].max()),
    }

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        min_val, max_val = normalization_params[col]
        df[col] = (df[col] - min_val) / (max_val - min_val)

    data = df[["Open", "High", "Low", "Close", "Volume"]].values
    return data
