"""Data loading and preprocessing utilities for stroke risk modeling."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

GLUCOSE_HISTORY_COLS = ("glucose_mean", "glucose_sd", "glucose_cv")


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the stroke dataset from CSV."""
    return pd.read_csv(csv_path)


def add_glucose_variability_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible wrapper that no longer simulates random glucose history.

    If real glucose history features exist in the dataset they are used directly.
    """
    return enrich_with_available_glucose_history(df)


def enrich_with_available_glucose_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare optional real glucose-history features if present.

    This function performs numeric coercion only; it does not create synthetic
    history features.
    """
    updated = df.copy()

    for col in GLUCOSE_HISTORY_COLS:
        if col in updated.columns:
            updated[col] = pd.to_numeric(updated[col], errors="coerce")

    if "is_diabetic" not in updated.columns:
        if "glucose_mean" in updated.columns:
            updated["is_diabetic"] = (updated["glucose_mean"] >= 126).astype(int)
        else:
            updated["is_diabetic"] = (updated["avg_glucose_level"] >= 126).astype(int)

    return updated


def filter_diabetic_patients(df: pd.DataFrame) -> pd.DataFrame:
    """Keep diabetic patients using deterministic criteria."""
    if "is_diabetic" in df.columns:
        return df[df["is_diabetic"] == 1]

    if "glucose_mean" in df.columns:
        return df[df["glucose_mean"] >= 126]

    return df[df["avg_glucose_level"] >= 126]


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnecessary columns."""
    columns_to_drop = ["id"]
    return df.drop(columns=[c for c in columns_to_drop if c in df.columns])


def split_features_target(
    df: pd.DataFrame, target_col: str = "stroke"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create transformer that imputes, one-hot encodes, and scales."""
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    return preprocessor


def glucose_history_stats(readings: Iterable[float]) -> Tuple[float, float, float]:
    """
    Compute glucose mean, SD, and CV from user-provided historical readings.
    """
    arr = np.asarray(list(readings), dtype=float)
    mean_glucose = float(np.mean(arr))
    sd_glucose = float(np.std(arr))
    cv_glucose = float(sd_glucose / mean_glucose) if mean_glucose != 0 else 0.0
    return mean_glucose, sd_glucose, cv_glucose
