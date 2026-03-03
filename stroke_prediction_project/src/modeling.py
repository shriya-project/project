"""Model training, tuning, evaluation, and persistence."""

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

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


@dataclass
class ModelArtifacts:
    name: str
    best_estimator: GridSearchCV
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

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
            ),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [3, 6],
                "model__learning_rate": [0.05, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            },
        )

    return models


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, object]:
    """Compute classification and ranking metrics."""
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback for estimators without predict_proba.
        y_proba = y_pred

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


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
            "confusion_matrix": str(metrics["confusion_matrix"]),
        }
        all_results.append(row)

        if metrics["recall"] > best_recall:
            best_recall = metrics["recall"]
            best_artifact = ModelArtifacts(
                name=model_name,
                best_estimator=grid,
                metrics=metrics,
            )

    comparison_df = pd.DataFrame(all_results).sort_values("recall", ascending=False).reset_index(drop=True)
    comparison_df.to_csv(COMPARISON_PATH, index=False)
    return best_artifact, comparison_df


def save_best_model(best_estimator) -> None:
    """Persist the best trained model pipeline."""
    joblib.dump(best_estimator, BEST_MODEL_PATH)

