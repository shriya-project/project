"""Data loading and preprocessing utilities aligned with glucose variability study."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RANDOM_STATE


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the stroke dataset from CSV."""
    return pd.read_csv(csv_path)


def add_glucose_variability_features(
    df: pd.DataFrame, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """
    Simulate retrospective glucose readings and compute:
    - Mean glucose
    - Standard deviation (SD)
    - Coefficient of variation (CV)

    This aligns with the base paper focus on glucose variability.
    """

    rng = np.random.default_rng(seed=random_state)
    updated = df.copy()

    # Simulate 10 retrospective glucose readings per patient
    simulated_readings = []

    for avg_glucose in updated["avg_glucose_level"]:
        readings = rng.normal(
            loc=avg_glucose,
            scale=rng.uniform(10, 40),
            size=10,
        )
        simulated_readings.append(readings)

    updated["glucose_mean"] = [np.mean(r) for r in simulated_readings]
    updated["glucose_sd"] = [np.std(r) for r in simulated_readings]
    updated["glucose_cv"] = (
        updated["glucose_sd"] / updated["glucose_mean"]
    )

    # Diabetic cohort (retrospective diabetic study)
    updated["is_diabetic"] = (
        updated["glucose_mean"] >= 126
    ).astype(int)

    return updated


def filter_diabetic_patients(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only diabetic patients to align with base paper.
    """
    return df[df["is_diabetic"] == 1]


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