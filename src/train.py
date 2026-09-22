"""Train, evaluate, and save the student performance classifier.

Run from the project root:
    python -m src.train
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import build_features, build_preprocessor, load_data

RANDOM_STATE = 42
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "model.joblib"
CLASSES = ["Low", "Medium", "High"]


def build_pipelines() -> dict[str, Pipeline]:
    """Each pipeline bundles preprocessing with the model so the exact same
    transformations are applied at training and inference time."""
    return {
        "logistic_regression": Pipeline([
            ("preprocess", build_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("preprocess", build_preprocessor(scale_numeric=False)),
            ("model", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
        ]),
    }


def evaluate(pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=CLASSES),
        "report": classification_report(y_test, y_pred, labels=CLASSES,
                                        zero_division=0),
    }


def main() -> None:
    df = load_data()
    X, y = build_features(df)

    # Stratified split keeps the class proportions similar in train and test,
    # which matters here because the classes are imbalanced.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print("Class distribution (train):")
    print(y_train.value_counts().reindex(CLASSES).to_string())

    results = {}
    for name, pipeline in build_pipelines().items():
        pipeline.fit(X_train, y_train)
        results[name] = evaluate(pipeline, X_test, y_test)

        print(f"\n===== {name} =====")
        print(f"Accuracy : {results[name]['accuracy']:.3f}")
        print(f"F1 macro : {results[name]['f1_macro']:.3f}")
        print(results[name]["report"])
        print("Confusion matrix (rows=true, cols=pred; order Low/Medium/High):")
        print(results[name]["confusion_matrix"])

    # Select on F1-macro: it weights every class equally, so a model cannot
    # win by only doing well on the majority class.
    best_name = max(results, key=lambda n: results[n]["f1_macro"])
    print(f"\nSelected model: {best_name} "
          f"(F1-macro {results[best_name]['f1_macro']:.3f})")

    MODELS_DIR.mkdir(exist_ok=True)
    # Rebuild and refit the winning pipeline, then persist it with joblib.
    best_pipeline = build_pipelines()[best_name]
    best_pipeline.fit(X_train, y_train)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Saved to {MODEL_PATH}")

    # Sanity check: reload and confirm predictions match.
    reloaded = joblib.load(MODEL_PATH)
    check = reloaded.predict(X_test.head(5))
    assert np.array_equal(check, best_pipeline.predict(X_test.head(5)))
    print("Reload check: OK")


if __name__ == "__main__":
    main()
