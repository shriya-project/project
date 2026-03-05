"""Model training, tuning, evaluation, interpretability, and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

from .config import BEST_MODEL_PATH, COMPARISON_PATH, RANDOM_STATE


@dataclass
class ModelArtifacts:
    name: str
    best_estimator: ImbPipeline
    metrics: Dict[str, float]


def get_models_and_grids() -> Dict[str, Tuple[object, Dict[str, list]]]:
    """Define models and GridSearch parameter grids."""

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__solver": ["liblinear", "lbfgs"],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 8, 16],
                "model__min_samples_split": [2, 10],
                "model__class_weight": [None, "balanced"],
            },
        ),
    }

    return models


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, object]:
    """Compute classification and ranking metrics."""

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def extract_feature_importance(best_pipeline: ImbPipeline) -> pd.DataFrame:
    """
    Extract feature importance correctly from transformed features.
    Fixes length mismatch issue.
    """

    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]

    # Get transformed feature names AFTER preprocessing
    feature_names = preprocessor.get_feature_names_out()

    # Get importance values
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    # Safety check to avoid length mismatch
    if len(feature_names) != len(importance):
        print("Warning: Feature names and importance length mismatch.")
        return pd.DataFrame()

    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    return feature_importance_df


def train_and_tune_models(preprocessor, X_train, y_train, X_test, y_test) -> Tuple[ModelArtifacts, pd.DataFrame]:
    """Train all models with SMOTE + GridSearch and return best by recall."""

    model_defs = get_models_and_grids()
    smote = SMOTE(random_state=RANDOM_STATE)
    all_results = []
    best_artifact = None
    best_recall = -np.inf

    for model_name, (estimator, param_grid) in model_defs.items():

        pipeline = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", smote),
                ("model", estimator),
            ]
        )

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="recall",
            cv=5,
            n_jobs=-1,
            verbose=0,
        )

        grid.fit(X_train, y_train)

        metrics = evaluate_model(grid.best_estimator_, X_test, y_test)

        row = {
            "model": model_name,
            "best_params": str(grid.best_params_),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
        }

        all_results.append(row)

        if metrics["recall"] > best_recall:
            best_recall = metrics["recall"]
            best_artifact = ModelArtifacts(
                name=model_name,
                best_estimator=grid.best_estimator_,
                metrics=metrics,
            )

    comparison_df = pd.DataFrame(all_results).sort_values(
        "recall", ascending=False
    ).reset_index(drop=True)

    comparison_df.to_csv(COMPARISON_PATH, index=False)

    # Extract feature importance safely
    if best_artifact is not None:
        feature_importance_df = extract_feature_importance(
            best_artifact.best_estimator
        )
        if not feature_importance_df.empty:
            feature_importance_df.to_csv("models/feature_importance.csv", index=False)

    return best_artifact, comparison_df


def save_best_model(best_estimator) -> None:
    """Persist the best trained model pipeline."""
    joblib.dump(best_estimator, BEST_MODEL_PATH)