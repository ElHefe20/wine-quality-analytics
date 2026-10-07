"""Shared helpers used by both the training script and the Streamlit app."""
from pathlib import Path

import pandas as pd

RANDOM_STATE = 42

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

UNITS = {
    "fixed acidity": "g/L",
    "volatile acidity": "g/L",
    "citric acid": "g/L",
    "residual sugar": "g/L",
    "chlorides": "g/L",
    "free sulfur dioxide": "mg/L",
    "total sulfur dioxide": "mg/L",
    "density": "g/cm³",
    "pH": "",
    "sulphates": "g/L",
    "alcohol": "% vol",
}

STEPS = {
    "fixed acidity": 0.1,
    "volatile acidity": 0.01,
    "citric acid": 0.01,
    "residual sugar": 0.1,
    "chlorides": 0.001,
    "free sulfur dioxide": 1.0,
    "total sulfur dioxide": 1.0,
    "density": 0.0001,
    "pH": 0.01,
    "sulphates": 0.01,
    "alcohol": 0.1,
}

FORMATS = {
    "fixed acidity": "%.1f",
    "volatile acidity": "%.2f",
    "citric acid": "%.2f",
    "residual sugar": "%.1f",
    "chlorides": "%.3f",
    "free sulfur dioxide": "%.0f",
    "total sulfur dioxide": "%.0f",
    "density": "%.4f",
    "pH": "%.2f",
    "sulphates": "%.2f",
    "alcohol": "%.1f",
}

DATA_DIRS = [Path("data"), Path("../data"), Path(".")]


def find_file(filename):
    for folder in DATA_DIRS:
        if (folder / filename).exists():
            return folder / filename
    raise FileNotFoundError(
        f"'{filename}' not found in {[str(d) for d in DATA_DIRS]}. "
        "Place the CSV files in the data/ folder."
    )


def read_wine_csv(path):
    """Read a wine CSV, detecting whether it is ';' (UCI original) or ',' separated."""
    with open(path, encoding="utf-8") as f:
        sep = ";" if ";" in f.readline() else ","
    return pd.read_csv(path, sep=sep)


def load_clean_data():
    """Load red + white wines and remove exact duplicate records (same as the notebook)."""
    frames = []
    for wine_type in ("red", "white"):
        df = read_wine_csv(find_file(f"winequality-{wine_type}.csv"))
        df["type"] = wine_type
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    return raw.drop_duplicates().reset_index(drop=True)


def molecular_so2(free_so2, ph, pka1=1.81):
    """Molecular (active) SO2 from free SO2 and pH (Henderson-Hasselbalch)."""
    return free_so2 / (1 + 10 ** (ph - pka1))


def validate(values):
    """Basic chemical consistency checks on user-entered values."""
    problems = []
    if values["free sulfur dioxide"] > values["total sulfur dioxide"]:
        problems.append(
            "Free SO₂ cannot exceed total SO₂ (free SO₂ is a fraction of the total). "
            "Please correct one of the two values."
        )
    return problems
