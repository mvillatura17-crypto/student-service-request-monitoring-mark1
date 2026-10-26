"""Train the midterm Decision Tree pipeline and save it with joblib.

Same settings as the midterm notebook: DecisionTreeClassifier(max_depth=5,
min_samples_leaf=5, random_state=42), 80/20 stratified split.
Run it through train_model.py (not from the Streamlit app).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from modules.data_processing import (
    CATEGORY_COLUMNS, MODEL_CATEGORICAL, MODEL_FEATURES, MODEL_NUMERICAL,
    TARGET, clean_data, prepare_features, validate_schema,
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_VERSION = "1.0"


def build_pipeline(max_depth: int = 5, min_leaf: int = 5, seed: int = 42) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]), MODEL_CATEGORICAL),
            ("numerical", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]), MODEL_NUMERICAL),
        ]
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_leaf, random_state=seed)),
    ])


def original_feature(encoded_name: str) -> str:
    name = encoded_name.replace("categorical__", "").replace("numerical__", "")
    for prefix in MODEL_CATEGORICAL:
        if name.startswith(prefix + "_"):
            return prefix
    return name


def _metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision (breach)": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "Recall (breach)": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "F1-score (breach)": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def train_and_save(csv_path, data_source: str, model_dir=None,
                   max_depth: int = 5, min_leaf: int = 5,
                   seed: int = 42, test_size: float = 0.20) -> dict:
    model_dir = Path(model_dir) if model_dir else BASE_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    problems = validate_schema(df)
    if problems["missing"]:
        raise ValueError("Missing required columns: " + ", ".join(problems["missing"]))

    clean, model_df, _ = clean_data(df)
    data = model_df.dropna(subset=[TARGET]).copy()
    X = prepare_features(data)
    y = data[TARGET].astype(int)
    if y.nunique() < 2:
        raise ValueError("The target has only one class; cannot train a classifier.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    model = build_pipeline(max_depth, min_leaf, seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, list(model.classes_).index(1)]

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    b_pred = baseline.predict(X_test)

    names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    grouped = (
        pd.DataFrame({"feature": [original_feature(n) for n in names], "importance": importances})
        .groupby("feature")["importance"].sum().sort_values(ascending=False)
    )

    # Dropdown options for the Prediction page come from the cleaned data
    options = {}
    for col in MODEL_CATEGORICAL:
        if col == "Submission_Weekday":
            continue
        options[col] = sorted(clean[col].dropna().astype(str).unique().tolist())

    report = classification_report(
        y_test, y_pred, labels=[0, 1], target_names=["No Breach", "Breach"],
        zero_division=0, output_dict=True)

    info = {
        "model_name": "Decision Tree (preprocessing + classifier pipeline)",
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": data_source,
        "sklearn_version": sklearn.__version__,
        "target": TARGET,
        "target_definition": "1 if Processing_Days > SLA_Days (completed requests only)",
        "features": MODEL_FEATURES,
        "hyperparameters": {"max_depth": max_depth, "min_samples_leaf": min_leaf,
                            "random_state": seed, "test_size": test_size},
        "n_rows_raw": int(len(df)),
        "n_rows_model": int(len(data)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "metrics": _metrics(y_test, y_pred),
        "baseline_metrics": _metrics(y_test, b_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "classification_report": report,
        "feature_importance": {k: float(v) for k, v in grouped.items()},
        "category_options": options,
    }

    joblib.dump(model, model_dir / "trained_model.joblib", compress=3)
    (model_dir / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    # Held-out test cases + the predictions the saved model made at training time.
    # The Prediction page re-runs them to prove the saved model is consistent.
    cases = X_test.copy()
    cases["Actual_SLA_Breached"] = y_test.values
    cases["Saved_Prediction"] = y_pred
    cases["Saved_Probability"] = np.round(y_proba, 6)
    cases.to_csv(model_dir / "test_cases.csv", index=False)

    return info
