"""Project paths and constants.

Works both locally (repo layout) and on Kaggle (dataset mounted under /kaggle/input).
"""
from pathlib import Path

RANDOM_STATE = 42
TARGET = "is_fraud"

# Repo root = parent of src/
ROOT = Path(__file__).resolve().parents[1]

_KAGGLE_INPUT = Path("/kaggle/input/fraud-detection")
RAW_DIR = _KAGGLE_INPUT if _KAGGLE_INPUT.exists() else ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

for _d in (PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Time-based split boundaries (fraudTrain.csv ends 2020-06-21, fraudTest.csv starts after it)
VALID_START = "2020-03-21"  # last ~3 months of fraudTrain become the validation set

# Colours used in every plot: legit = grey, fraud = red
COLORS = {0: "#6b7280", 1: "#dc2626"}
LABELS = {0: "Legit", 1: "Fraud"}
