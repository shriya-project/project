"""Data loading and preprocessing utilities."""

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


def add_synthetic_features(df: pd.DataFrame, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Add synthetic feature `family_history` with 30% probability of 1.
    Then increase stroke probability by 5% for family_history=1 by flipping
    a small random subset of stroke=0 rows to stroke=1.
    """
    rng = np.random.default_rng(seed=random_state)
    updated = df.copy()

    updated["family_history"] = rng.binomial(1, 0.30, size=len(updated))

    eligible_mask = (updated["family_history"] == 1) & (updated["stroke"] == 0)
    flip_mask = rng.random(len(updated)) < 0.05
    updated.loc[eligible_mask & flip_mask, "stroke"] = 1

    # Proxy diabetic status using glucose level to align with project context.
    updated["is_diabetic_proxy"] = (updated["avg_glucose_level"] >= 125).astype(int)
    return updated


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are not useful for predictive training."""
    columns_to_drop = [col for col in ["id"] if col in df.columns]
    return df.drop(columns=columns_to_drop)


def split_features_target(df: pd.DataFrame, target_col: str = "stroke") -> Tuple[pd.DataFrame, pd.Series]:
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

