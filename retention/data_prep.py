"""Prepare a leakage-aware dataset for student dropout-risk modelling.

The supplied data represents a final academic outcome.  For an intervention
model that is used immediately after semester 1, this script deliberately:

* keeps only resolved outcomes (``Dropout`` and ``Graduate``),
* maps ``Dropout`` to 1 and ``Graduate`` to 0, and
* removes every second-semester curricular-unit field.

``Enrolled`` is not treated as a successful outcome because it has not yet
resolved to graduation.  It is therefore excluded from this binary supervised
learning dataset, rather than being silently labelled as retained.

Run from this directory or the repository root:

    python data_prep.py
    python retention/data_prep.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_FILE = BASE_DIR / "dataset.csv"
PROCESSED_DIR = BASE_DIR / "processed"

SOURCE_TARGET = "Target"
ROW_ID = "row_id"
OUTCOME_COLUMN = "outcome"
TARGET = "dropout_target"

POSITIVE_OUTCOME = "Dropout"
NEGATIVE_OUTCOME = "Graduate"
UNRESOLVED_OUTCOME = "Enrolled"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# These fields are present only after semester 2 and would make a semester-1
# intervention score misleadingly optimistic.  Keep this explicit rather than
# relying only on a name pattern, so an unexpected schema change fails early.
SECOND_SEMESTER_COLUMNS = [
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]

# These columns are retained in the prepared data for aggregate fairness audits
# only.  They are deliberately excluded from direct scoring, so that a risk
# score does not turn protected status into a reason for different treatment.
# The source codes have no local data dictionary, so any audit must first map
# codes to their documented categories.
AUDIT_ONLY_COLUMNS = [
    "Nacionality",
    "Educational special needs",
    "Gender",
    "International",
]

# The source stores category codes as integers.  They must be one-hot encoded
# downstream, not interpreted as continuous distances (for example, course 2
# is not "closer" to course 3 than course 11).  The spelling "Nacionality" is
# retained because it is the exact header in the source data; it is audit-only.
CATEGORICAL_COLUMNS = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
]

# Application order is preserved as an ordinal number.  The remaining values
# are counts, grades, age, or macroeconomic measurements and are treated as
# numeric.  Missing values, if any future data contains them, are imputed only
# inside the training pipeline.
NUMERIC_COLUMNS = [
    "Application order",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
PREPARED_COLUMNS = AUDIT_ONLY_COLUMNS + FEATURE_COLUMNS


def validate_source(df: pd.DataFrame) -> None:
    """Validate the minimum data contract before creating any artifacts."""
    if df.empty:
        raise ValueError("dataset.csv is empty")

    required = set(PREPARED_COLUMNS + SECOND_SEMESTER_COLUMNS + [SOURCE_TARGET])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"dataset.csv is missing required columns: {missing}")

    if set(CATEGORICAL_COLUMNS) & set(NUMERIC_COLUMNS):
        raise ValueError("Categorical and numeric feature lists overlap")

    leaked = [name for name in FEATURE_COLUMNS if "2nd sem" in name.lower()]
    if leaked:
        raise ValueError(f"Second-semester fields leaked into FEATURES: {leaked}")


def prepare_model_data(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return binary-outcome rows and a transparent data-quality report.

    Rows are not de-duplicated: this dataset has no student identifier, and
    equal feature values can still represent different students.  ``row_id``
    is a synthetic, source-row traceability key only; it is never a feature.
    """
    validate_source(source)

    raw_outcome = source[SOURCE_TARGET].astype("string").str.strip()
    accepted_outcomes = {POSITIVE_OUTCOME, NEGATIVE_OUTCOME}
    eligible = raw_outcome.isin(accepted_outcomes)

    # Audit-only columns travel with the processed records, but do not appear
    # in FEATURE_COLUMNS and therefore cannot enter the fitted pipeline.
    model_data = source.loc[eligible, PREPARED_COLUMNS].copy()
    model_data.insert(0, ROW_ID, source.index[eligible].to_numpy() + 1)
    model_data.insert(1, OUTCOME_COLUMN, raw_outcome.loc[eligible].astype(str).to_numpy())
    model_data.insert(
        2,
        TARGET,
        (raw_outcome.loc[eligible] == POSITIVE_OUTCOME).astype(int).to_numpy(),
    )

    if model_data.empty or model_data[TARGET].nunique() != 2:
        raise ValueError(
            "Need both Dropout and Graduate rows after excluding unresolved outcomes"
        )

    # The source does not include a reliable student ID.  We report exact
    # duplicate feature/outcome rows but retain them as potentially distinct
    # student records.
    duplicate_rows = int(source.duplicated().sum())
    missing_features = model_data[FEATURE_COLUMNS].isna().sum()
    source_outcomes = raw_outcome.value_counts(dropna=False)
    unknown_outcomes = sorted(
        str(value)
        for value in raw_outcome.dropna().unique()
        if value not in accepted_outcomes | {UNRESOLVED_OUTCOME}
    )

    report: dict[str, Any] = {
        "model_purpose": "Binary dropout-risk score available after semester 1",
        "source_file": RAW_DATA_FILE.name,
        "source_rows": int(len(source)),
        "eligible_rows": int(len(model_data)),
        "excluded_unresolved_enrolled_rows": int((raw_outcome == UNRESOLVED_OUTCOME).sum()),
        "excluded_unknown_or_missing_outcome_rows": int((~eligible).sum())
        - int((raw_outcome == UNRESOLVED_OUTCOME).sum()),
        "source_outcome_distribution": {
            str(key): int(value) for key, value in source_outcomes.items()
        },
        "unknown_outcome_values": unknown_outcomes,
        "binary_target_mapping": {
            POSITIVE_OUTCOME: 1,
            NEGATIVE_OUTCOME: 0,
        },
        "model_target_distribution": {
            str(key): int(value)
            for key, value in model_data[OUTCOME_COLUMN].value_counts().items()
        },
        "model_dropout_rate": float(model_data[TARGET].mean()),
        "synthetic_row_id": (
            "row_id is the one-based source row position for traceability only; "
            "it is not a student identifier and is never used as a feature."
        ),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "categorical_code_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "audit_only_columns": AUDIT_ONLY_COLUMNS,
        "fairness_note": (
            "Gender, Nacionality, Educational special needs, and International "
            "are retained for aggregate audit only and excluded from model features. "
            "Other inputs can still act as proxies; evaluate disparities before use."
        ),
        "excluded_for_temporal_leakage": SECOND_SEMESTER_COLUMNS,
        "missing_by_feature_before_imputation": {
            column: int(value) for column, value in missing_features.items()
        },
        "exact_duplicate_source_rows_retained": duplicate_rows,
        "limitations": [
            "No student identifier, cohort, or timestamp is available; the split is stratified random rather than temporal or group-based.",
            "The macroeconomic fields may proxy cohort conditions, so holdout performance is not evidence of future-cohort generalization.",
            "Category-code labels are not supplied; applications should not display raw code numbers as meaningful human labels.",
            "A risk score is for supportive outreach and must not be the sole basis for punitive or high-impact decisions.",
        ],
    }
    return model_data, report


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic UTF-8 JSON that remains readable by the app."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA_FILE}")

    source = pd.read_csv(RAW_DATA_FILE)
    model_data, report = prepare_model_data(source)

    train, test = train_test_split(
        model_data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=model_data[TARGET],
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    model_data.to_csv(PROCESSED_DIR / "model_data.csv", index=False)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)

    report["split"] = {
        "method": "stratified_random_holdout",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_dropout_rate": float(train[TARGET].mean()),
        "test_dropout_rate": float(test[TARGET].mean()),
    }
    write_json(PROCESSED_DIR / "data_quality.json", report)

    print("Data preparation complete.")
    print(f"Eligible resolved outcomes: {len(model_data):,}")
    print(f"Excluded Enrolled rows    : {report['excluded_unresolved_enrolled_rows']:,}")
    print(f"Train / test              : {len(train):,} / {len(test):,}")
    print(f"Dropout rate              : {model_data[TARGET].mean() * 100:.2f}%")
    print("Created: processed/model_data.csv, train.csv, test.csv, data_quality.json")


if __name__ == "__main__":
    main()
