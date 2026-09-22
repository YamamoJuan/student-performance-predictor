"""Load the trained model and make predictions from a single student record."""

from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import FEATURE_COLUMNS

MODEL_PATH = Path("models") / "model.joblib"


def load_model():
    """Load the persisted sklearn Pipeline. Call once per process."""
    return joblib.load(MODEL_PATH)


def predict_student(model, inputs: dict) -> dict:
    """Predict a performance class and confidence for one student.

    inputs must contain keys matching FEATURE_COLUMNS. Returns the predicted
    class and a per-class probability dictionary.
    """
    row = pd.DataFrame([inputs], columns=FEATURE_COLUMNS)
    predicted_class = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0]
    return {
        "prediction": predicted_class,
        "probabilities": dict(zip(model.classes_, probabilities)),
    }
