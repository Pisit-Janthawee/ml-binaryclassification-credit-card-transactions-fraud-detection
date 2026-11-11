"""Leak-free feature engineering.

Rule: every feature for a transaction may only use information available at the
moment that transaction happens (the transaction itself + the card's PAST
transactions). Nothing from the future, nothing from the label.
"""
import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points (vectorised)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features computable from a single row."""
    out = df.copy()
    t = out["trans_time"]
    out["hour"] = t.dt.hour.astype("int8")
    out["day_of_week"] = t.dt.dayofweek.astype("int8")
    out["is_night"] = out["hour"].isin([22, 23, 0, 1, 2, 3]).astype("int8")
    out["age"] = ((t - out["dob"]).dt.days / 365.25).astype("float32")  # age AT transaction time
    out["log_amount"] = np.log1p(out["amount"]).astype("float32")
    out["log_city_pop"] = np.log1p(out["city_pop"]).astype("float32")
    out["distance_km"] = haversine_km(out["lat"], out["long"], out["merch_lat"], out["merch_long"]).astype("float32")
    out["is_male"] = (out["gender"] == "M").astype("int8")
    return out


def add_card_history_features(df: pd.DataFrame, windows=("1h", "24h", "7d")) -> pd.DataFrame:
    """Per-card behavioural features that look only BACKWARDS in time.

    For each window W:
      card_txn_count_W  - number of the card's previous transactions in the last W
      card_amount_sum_W - total amount of those transactions
    Plus:
      secs_since_prev_txn - seconds since the card's previous transaction
      amount_vs_card_mean - amount / mean of the card's previous amounts (expanding)
      card_txn_index      - how many transactions the card has made before this one
    """
    out = df.sort_values(["card_id", "trans_time"], kind="stable").copy()
    g = out.groupby("card_id", sort=False, observed=True)

    out["secs_since_prev_txn"] = g["trans_time"].diff().dt.total_seconds().astype("float32")
    out["card_txn_index"] = g.cumcount().astype("int32")

    prev_mean = g["amount"].transform(lambda s: s.shift().expanding().mean())
    out["amount_vs_card_mean"] = (out["amount"] / prev_mean).astype("float32")

    indexed = out.set_index("trans_time")[["card_id", "amount"]]
    for w in windows:
        # closed="left" -> window [t - W, t): excludes the current transaction itself
        roll = indexed.groupby("card_id", sort=False)["amount"].rolling(w, closed="left")
        out[f"card_txn_count_{w}"] = roll.count().fillna(0).to_numpy().astype("float32")
        out[f"card_amount_sum_{w}"] = roll.sum().fillna(0).to_numpy().astype("float32")

    return out.sort_values("trans_time", kind="stable").reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_card_history_features(add_transaction_features(df))


# Final model inputs (identifiers, raw text and leaky columns excluded on purpose)
NUMERIC_FEATURES = [
    "amount", "log_amount", "hour", "day_of_week", "is_night", "age", "is_male",
    "log_city_pop", "distance_km",
    "secs_since_prev_txn", "card_txn_index", "amount_vs_card_mean",
    "card_txn_count_1h", "card_amount_sum_1h",
    "card_txn_count_24h", "card_amount_sum_24h",
    "card_txn_count_7d", "card_amount_sum_7d",
]
CATEGORICAL_FEATURES = ["category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
