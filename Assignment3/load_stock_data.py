from pathlib import Path

import pandas as pd


def normalise(value: float, min_val: float, max_val: float) -> float:
    return (value - min_val) / (max_val - min_val)


def unnormalise(value: float, min_val: float, max_val: float) -> float:
    return value * (max_val - min_val) + min_val


# Google Data is CSV with headers Date, Open, High, Low, Close, Volume
def load_stock_data(
    filepath: Path,
    filename: str,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    df = pd.read_csv(filepath / f"{filename}.csv")
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
        df[col] = normalise(df[col], min_val, max_val)

    return (df, normalization_params)
