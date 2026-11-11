"""Loading the raw Kaggle CSVs into tidy, memory-efficient DataFrames."""
import pandas as pd

from .config import RAW_DIR, PROCESSED_DIR

RENAME = {
    "trans_date_trans_time": "trans_time",
    "cc_num": "card_id",
    "amt": "amount",
    "trans_num": "trans_id",
}

CATEGORICAL = ["merchant", "category", "gender", "city", "state", "job"]


def load_raw(split: str = "train") -> pd.DataFrame:
    """Load fraudTrain.csv (split='train') or fraudTest.csv (split='test').

    - drops the CSV index column
    - renames a few columns to readable names
    - parses datetimes, stores low-cardinality text as `category` dtype
    """
    fname = {"train": "fraudTrain.csv", "test": "fraudTest.csv"}[split]
    df = pd.read_csv(RAW_DIR / fname, index_col=0)
    df = df.rename(columns=RENAME)
    df["trans_time"] = pd.to_datetime(df["trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])
    df["card_id"] = df["card_id"].astype("int64")
    for col in CATEGORICAL:
        df[col] = df[col].astype("category")
    df["is_fraud"] = df["is_fraud"].astype("int8")
    df["source"] = split
    return df


def load_all() -> pd.DataFrame:
    """Train + test stacked and sorted by time (needed for causal per-card features)."""
    df = pd.concat([load_raw("train"), load_raw("test")], ignore_index=True)
    for col in CATEGORICAL:  # concat can drop categorical dtype when categories differ
        df[col] = df[col].astype("category")
    return df.sort_values("trans_time", kind="stable").reset_index(drop=True)


def save_processed(df: pd.DataFrame, name: str) -> None:
    df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)


def load_processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
