"""Preprocessing for the student performance dataset.

All column definitions and the shared ColumnTransformer live here so that
training and prediction use identical preprocessing. The transformer is only
fitted inside sklearn Pipelines on training data, which prevents leakage.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# G1 and G2 are excluded: they are earlier period grades in the same course and
# almost directly reveal the target G3, which would be data leakage.
LEAKAGE_COLUMNS = ["G1", "G2"]
TARGET_RAW = "G3"

# Binary categoricals use simple 0/1 encoding; multi-valued ones use one-hot.
BINARY_CATEGORICAL = [
    "school", "sex", "address", "famsize", "Pstatus",
    "schoolsup", "famsup", "paid", "activities", "nursery",
    "higher", "internet", "romantic",
]
MULTI_CATEGORICAL = ["Mjob", "Fjob", "reason", "guardian"]
NUMERIC = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
]
FEATURE_COLUMNS = BINARY_CATEGORICAL + MULTI_CATEGORICAL + NUMERIC


def grade_to_class(g3: float) -> str:
    """Map final grade G3 (0-20) to a performance class.

    Low: <= 9 (failing), Medium: 10-13 (passing), High: >= 14 (strong).
    """
    if g3 <= 9:
        return "Low"
    if g3 <= 13:
        return "Medium"
    return "High"


def load_data(path: str = "data/student-mat.csv") -> pd.DataFrame:
    """Load the raw UCI student-mat dataset (semicolon-separated)."""
    return pd.read_csv(path, sep=";")


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with leakage columns dropped and the 3-class target."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_RAW].map(grade_to_class)
    return X, y


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Build the shared ColumnTransformer.

    SimpleImputer is included to handle missing values robustly even though
    this dataset has none. scale_numeric should be True for models that are
    sensitive to feature scale (e.g. logistic regression).
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    # Binary columns hold exactly two categories, so one-hot with drop="if_binary"
    # yields a single 0/1 column each and avoids redundant features.
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, BINARY_CATEGORICAL + MULTI_CATEGORICAL),
    ])
