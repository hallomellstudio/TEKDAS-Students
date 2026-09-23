"""Train and evaluate a calibrated semester-1 student dropout-risk model.

Run after ``data_prep.py``:

    python train.py
    python retention/train.py

The saved ``artifacts/model.joblib`` accepts a pandas DataFrame containing
``FEATURE_COLUMNS`` and returns the probability that the row belongs to the
``Dropout`` class.  It is a decision-support score for supportive outreach,
not an automated academic decision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_prep import (
    AUDIT_ONLY_COLUMNS,
    BASE_DIR,
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    OUTCOME_COLUMN,
    PROCESSED_DIR,
    RANDOM_STATE,
    ROW_ID,
    SECOND_SEMESTER_COLUMNS,
    TARGET,
)


ARTIFACT_DIR = BASE_DIR / "artifacts"
TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"

# This is an operational flag threshold, not a statement that probabilities
# above it are certain.  Campuses should set it with their support capacity and
# the relative cost of missed students versus unnecessary outreach in mind.
DECISION_THRESHOLD = 0.50
MEDIUM_RISK_CUTOFF = 0.30
HIGH_RISK_CUTOFF = 0.60
CALIBRATION_FOLDS = 5


def build_preprocessor() -> ColumnTransformer:
    """Create leak-safe transformations fitted only inside cross-validation/train."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )


def build_model() -> CalibratedClassifierCV:
    """Build a calibrated, regularized logistic-regression risk model.

    Logistic regression keeps the educational example interpretable and works
    well with one-hot category codes.  Sigmoid calibration is fitted with
    folds within the training partition, leaving the final test partition
    untouched until evaluation.
    """
    base_model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=3_000,
                ),
            ),
        ]
    )
    folds = StratifiedKFold(
        n_splits=CALIBRATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=folds,
        n_jobs=-1,
    )


def validate_prepared_split(frame: pd.DataFrame, name: str) -> None:
    """Fail with a helpful error if an app or script has altered the contract."""
    required = set(FEATURE_COLUMNS + [ROW_ID, OUTCOME_COLUMN, TARGET])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    if frame[TARGET].isna().any() or not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"{name} has an invalid binary {TARGET!r} column")


def dropout_probabilities(model: CalibratedClassifierCV, features: pd.DataFrame) -> np.ndarray:
    """Return probabilities for the explicit positive class, Dropout = 1."""
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("The fitted model does not contain Dropout class 1")
    return model.predict_proba(features)[:, classes.index(1)]


def risk_level_from_probability(probability: float) -> str:
    """Map a probability to a UI-friendly, deliberately coarse risk band."""
    if probability >= HIGH_RISK_CUTOFF:
        return "High"
    if probability >= MEDIUM_RISK_CUTOFF:
        return "Medium"
    return "Low"


def calibration_table(
    y_true: pd.Series, probabilities: np.ndarray, bins: int = 10
) -> tuple[list[dict[str, float | int | str]], float]:
    """Create a holdout calibration table and expected calibration error.

    ECE is a descriptive diagnostic, not a universal calibration test; it
    changes with the number and placement of bins.  Fixed bins make the saved
    table easy for the Streamlit app to render consistently.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    # np.digitize puts a probability of exactly 1.0 into index ``bins``;
    # clipping keeps every score in one of the requested intervals.
    bin_index = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, bins - 1)
    frame = pd.DataFrame(
        {
            "y_true": np.asarray(y_true, dtype=int),
            "probability": probabilities,
            "bin_index": bin_index,
        }
    )

    rows: list[dict[str, float | int | str]] = []
    ece = 0.0
    total = len(frame)
    for index in range(bins):
        group = frame.loc[frame["bin_index"] == index]
        if group.empty:
            continue
        observed = float(group["y_true"].mean())
        predicted = float(group["probability"].mean())
        count = int(len(group))
        ece += (count / total) * abs(observed - predicted)
        rows.append(
            {
                "bin": f"{edges[index]:.1f}-{edges[index + 1]:.1f}",
                "count": count,
                "mean_predicted_dropout_probability": predicted,
                "observed_dropout_rate": observed,
            }
        )
    return rows, float(ece)


def evaluate(
    y_true: pd.Series, probabilities: np.ndarray, threshold: float = DECISION_THRESHOLD
) -> dict[str, Any]:
    """Compute discrimination, operational, and calibration diagnostics."""
    prediction = (probabilities >= threshold).astype(int)
    calibration_bins, ece = calibration_table(y_true, probabilities)
    return {
        "positive_class": {"label": "Dropout", "value": 1},
        "decision_threshold": threshold,
        "test_rows": int(len(y_true)),
        "test_dropout_rate": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision_dropout": float(precision_score(y_true, prediction, zero_division=0)),
        "recall_dropout": float(recall_score(y_true, prediction, zero_division=0)),
        "f1_dropout": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "pr_auc_no_skill_baseline": float(y_true.mean()),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "expected_calibration_error_10_bins": ece,
        "calibration_bins": calibration_bins,
        "confusion_matrix_labels": ["Graduate (0)", "Dropout (1)"],
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            prediction,
            labels=[0, 1],
            target_names=["Graduate", "Dropout"],
            output_dict=True,
            zero_division=0,
        ),
    }


def build_feature_importance(
    model: CalibratedClassifierCV, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """Estimate whole-feature importance without confusing category codes.

    Permuting raw columns evaluates the deployed calibrated model as a whole;
    this is more useful to the dashboard than presenting one coefficient for
    each arbitrary category code.  It is descriptive, not causal evidence.
    """
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "scoring": "decrease in held-out average precision when permuted",
        }
    )
    return importance.sort_values("importance_mean", ascending=False, ignore_index=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON while converting NumPy scalar values emitted by sklearn."""

    def default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=default),
        encoding="utf-8",
    )


def main() -> None:
    if not TRAIN_FILE.exists() or not TEST_FILE.exists():
        raise FileNotFoundError(
            "Prepared files are missing. Run `python data_prep.py` before `python train.py`."
        )

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    validate_prepared_split(train, "processed/train.csv")
    validate_prepared_split(test, "processed/test.csv")

    leakage = sorted(set(FEATURE_COLUMNS) & set(SECOND_SEMESTER_COLUMNS))
    if leakage:
        raise ValueError(f"Temporal leakage detected in model features: {leakage}")

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET].astype(int)
    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET].astype(int)

    print("Training calibrated logistic-regression dropout-risk model...")
    model = build_model()
    model.fit(X_train, y_train)
    probabilities = dropout_probabilities(model, X_test)
    metrics = evaluate(y_test, probabilities)

    print("Computing permutation feature importance...")
    feature_importance = build_feature_importance(model, X_test, y_test)

    # Store enough metadata beside the fitted estimator for a Streamlit app to
    # score frames consistently without making row_id or audit-only fields
    # model inputs.  Joblib preserves these attributes.
    model.feature_columns_ = list(FEATURE_COLUMNS)
    model.audit_only_columns_ = list(AUDIT_ONLY_COLUMNS)
    model.decision_threshold_ = DECISION_THRESHOLD
    model.risk_cutoffs_ = {
        "medium": MEDIUM_RISK_CUTOFF,
        "high": HIGH_RISK_CUTOFF,
    }
    model.model_purpose_ = "Semester-1 binary dropout-risk decision support"

    prediction = test[[ROW_ID, OUTCOME_COLUMN, TARGET]].copy()
    prediction["dropout_probability"] = probabilities
    prediction["prediction"] = (probabilities >= DECISION_THRESHOLD).astype(int)
    prediction["prediction_label"] = np.where(
        prediction["prediction"].eq(1), "Dropout risk", "Not flagged"
    )
    prediction["risk_level"] = [
        risk_level_from_probability(value) for value in prediction["dropout_probability"]
    ]
    prediction = prediction.sort_values("dropout_probability", ascending=False, ignore_index=True)

    metrics["model"] = {
        "algorithm": "L2-regularized logistic regression with sigmoid calibration",
        "calibration_method": (
            f"{CALIBRATION_FOLDS}-fold sigmoid calibration fitted only within the training partition"
        ),
        "categorical_handling": "median/mode imputation and one-hot encoding; unknown app codes are ignored",
        "numeric_handling": "median imputation and standardization",
        "feature_count_before_one_hot": len(FEATURE_COLUMNS),
        "audit_only_columns_excluded_from_scoring": AUDIT_ONLY_COLUMNS,
        "temporal_leakage_excluded_columns": SECOND_SEMESTER_COLUMNS,
    }
    metrics["interpretation_note"] = (
        "Feature importance is permutation-based association on this held-out split; "
        "it is not a causal explanation for an individual student."
    )
    metrics["calibration_caveat"] = (
        "Brier score and ECE are estimated on one random holdout. They do not establish "
        "calibration for another institution, cohort, or future academic year."
    )
    metrics["use_caveat"] = (
        "Use risk bands to prioritize supportive human outreach. Do not make punitive, "
        "admissions, scholarship, or enrollment decisions automatically from this score."
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    prediction.to_csv(ARTIFACT_DIR / "predictions.csv", index=False)
    feature_importance.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)
    write_json(ARTIFACT_DIR / "metrics.json", metrics)

    print("\nHoldout metrics")
    print(f"PR-AUC    : {metrics['pr_auc']:.3f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.3f}")
    print(f"Precision : {metrics['precision_dropout']:.3f}")
    print(f"Recall    : {metrics['recall_dropout']:.3f}")
    print(f"F1        : {metrics['f1_dropout']:.3f}")
    print(f"Brier     : {metrics['brier_score']:.3f}")
    print("Saved: artifacts/model.joblib, metrics.json, predictions.csv, feature_importance.csv")


if __name__ == "__main__":
    main()
