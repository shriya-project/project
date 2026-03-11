"""Model training, threshold tuning, evaluation, interpretability, and persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

from .config import (
    BEST_MODEL_PATH,
    COMPARISON_PATH,
    CV_RESULTS_PATH,
    HOLDOUT_METRICS_PATH,
    METADATA_PATH,
    RANDOM_STATE,
)


@dataclass
class ModelArtifacts:
    name: str
    best_estimator: ImbPipeline
    threshold: float
    threshold_strategy: str
    min_precision: float
    holdout_metrics: Dict[str, float]
    holdout_proba: np.ndarray
    holdout_pred: np.ndarray
    cv_summary: Dict[str, Dict[str, float]]
    best_params: Dict[str, object]


def get_models_and_grids() -> Dict[str, Tuple[object, Dict[str, list]]]:
    """Define models and GridSearch parameter grids."""
    models: Dict[str, Tuple[object, Dict[str, list]]] = {
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

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
            ),
            {
                "model__n_estimators": [200, 300],
                "model__max_depth": [3, 4, 6],
                "model__learning_rate": [0.03, 0.05, 0.1],
            },
        )
    except ImportError:
        pass

    return models


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def evaluate_probabilities(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> Dict[str, object]:
    """Compute metrics at a custom decision threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(pr_recall, pr_precision)

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_auc(y_true, y_proba),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def tune_decision_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    strategy: str = "max_f1",
    min_precision: float = 0.20,
) -> float:
    """Tune threshold by F1 or recall at minimum precision."""
    candidate_thresholds = np.linspace(0.05, 0.95, 181)
    candidates: List[Dict[str, float]] = []

    for threshold in candidate_thresholds:
        metrics = evaluate_probabilities(y_true, y_proba, float(threshold))
        candidates.append(metrics)

    if strategy == "recall_at_precision":
        valid = [c for c in candidates if c["precision"] >= min_precision]
        if valid:
            valid.sort(key=lambda x: (x["recall"], x["f1_score"]), reverse=True)
            return float(valid[0]["threshold"])

    candidates.sort(key=lambda x: (x["f1_score"], x["balanced_accuracy"]), reverse=True)
    return float(candidates[0]["threshold"])


def _generate_oof_probabilities(
    estimator: ImbPipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> np.ndarray:
    """Generate out-of-fold predicted probabilities."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros(len(X), dtype=float)

    for train_idx, valid_idx in skf.split(X, y):
        estimator_fold = clone(estimator)
        estimator_fold.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_proba[valid_idx] = estimator_fold.predict_proba(X.iloc[valid_idx])[:, 1]

    return oof_proba


def repeated_cv_with_confidence_intervals(
    estimator: ImbPipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
    n_splits: int = 5,
    n_repeats: int = 3,
) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """Run repeated stratified CV and compute 95% confidence intervals."""
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
    )

    fold_metrics: List[Dict[str, float]] = []
    for fold_id, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        estimator_fold = clone(estimator)
        estimator_fold.fit(X.iloc[train_idx], y.iloc[train_idx])

        y_proba = estimator_fold.predict_proba(X.iloc[valid_idx])[:, 1]
        metrics = evaluate_probabilities(y.iloc[valid_idx].to_numpy(), y_proba, threshold)
        metrics["fold"] = fold_id
        fold_metrics.append(metrics)

    fold_df = pd.DataFrame(fold_metrics)

    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
        "specificity",
        "brier_score",
    ]

    summary: Dict[str, Dict[str, float]] = {}
    for metric in metric_cols:
        values = fold_df[metric].dropna().to_numpy(dtype=float)
        mean_val = float(np.mean(values))
        std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        margin = 1.96 * std_val / np.sqrt(len(values)) if len(values) > 1 else 0.0
        summary[metric] = {
            "mean": mean_val,
            "std": std_val,
            "ci_low": float(mean_val - margin),
            "ci_high": float(mean_val + margin),
        }

    return summary, fold_df


def extract_feature_importance(best_pipeline: ImbPipeline) -> pd.DataFrame:
    """Extract feature importance from transformed features."""
    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    if len(feature_names) != len(importance):
        return pd.DataFrame()

    return (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _to_native(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    return value


def save_model_metadata(
    artifact: ModelArtifacts,
    training_rows: int,
    holdout_rows: int,
    holdout_positive_rate: float,
) -> None:
    """Persist metadata for app inference and reporting."""
    payload = {
        "model_name": artifact.name,
        "best_params": artifact.best_params,
        "decision_threshold": artifact.threshold,
        "threshold_strategy": artifact.threshold_strategy,
        "min_precision": artifact.min_precision,
        "training_rows": training_rows,
        "holdout_rows": holdout_rows,
        "holdout_positive_rate": holdout_positive_rate,
        "holdout_metrics": artifact.holdout_metrics,
        "cv_summary": artifact.cv_summary,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as fp:
        json.dump(_to_native(payload), fp, indent=2)


def train_and_tune_models(
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    threshold_strategy: str = "max_f1",
    min_precision: float = 0.20,
) -> Tuple[ModelArtifacts, pd.DataFrame]:
    """Train all models, tune thresholds, and return best by repeated-CV objective."""
    model_defs = get_models_and_grids()
    smote = SMOTE(random_state=RANDOM_STATE)

    all_results = []
    all_cv_folds = []
    best_artifact = None
    best_objective = -np.inf

    for model_name, (estimator, param_grid) in model_defs.items():
        pipeline = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", smote),
                ("model", estimator),
            ]
        )

        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)

        tuned_pipeline = grid.best_estimator_

        oof_proba = _generate_oof_probabilities(tuned_pipeline, X_train, y_train, n_splits=5)
        threshold = tune_decision_threshold(
            y_true=y_train.to_numpy(),
            y_proba=oof_proba,
            strategy=threshold_strategy,
            min_precision=min_precision,
        )

        cv_summary, cv_folds = repeated_cv_with_confidence_intervals(
            estimator=tuned_pipeline,
            X=X_train,
            y=y_train,
            threshold=threshold,
            n_splits=5,
            n_repeats=3,
        )
        cv_folds.insert(0, "model", model_name)
        all_cv_folds.append(cv_folds)

        holdout_proba = tuned_pipeline.predict_proba(X_holdout)[:, 1]
        holdout_metrics = evaluate_probabilities(
            y_true=y_holdout.to_numpy(),
            y_proba=holdout_proba,
            threshold=threshold,
        )

        objective_metric = "f1_score" if threshold_strategy == "max_f1" else "recall"
        objective_value = cv_summary[objective_metric]["mean"]

        all_results.append(
            {
                "model": model_name,
                "best_params": str(grid.best_params_),
                "threshold": threshold,
                "cv_f1_mean": cv_summary["f1_score"]["mean"],
                "cv_recall_mean": cv_summary["recall"]["mean"],
                "cv_precision_mean": cv_summary["precision"]["mean"],
                "cv_roc_auc_mean": cv_summary["roc_auc"]["mean"],
                "holdout_accuracy": holdout_metrics["accuracy"],
                "holdout_precision": holdout_metrics["precision"],
                "holdout_recall": holdout_metrics["recall"],
                "holdout_f1_score": holdout_metrics["f1_score"],
                "holdout_roc_auc": holdout_metrics["roc_auc"],
                "holdout_pr_auc": holdout_metrics["pr_auc"],
                "holdout_specificity": holdout_metrics["specificity"],
                "holdout_brier_score": holdout_metrics["brier_score"],
            }
        )

        if objective_value > best_objective:
            best_objective = objective_value
            best_artifact = ModelArtifacts(
                name=model_name,
                best_estimator=tuned_pipeline,
                threshold=threshold,
                threshold_strategy=threshold_strategy,
                min_precision=min_precision,
                holdout_metrics=holdout_metrics,
                holdout_proba=holdout_proba,
                holdout_pred=(holdout_proba >= threshold).astype(int),
                cv_summary=cv_summary,
                best_params=grid.best_params_,
            )

    comparison_df = pd.DataFrame(all_results).sort_values(
        by=["cv_f1_mean", "holdout_f1_score"],
        ascending=False,
    )
    comparison_df.to_csv(COMPARISON_PATH, index=False)

    cv_results_df = pd.concat(all_cv_folds, axis=0, ignore_index=True)
    cv_results_df.to_csv(CV_RESULTS_PATH, index=False)

    if best_artifact is None:
        raise RuntimeError("No model artifacts were generated.")

    holdout_df = pd.DataFrame(
        [{k: v for k, v in best_artifact.holdout_metrics.items() if k != "confusion_matrix"}]
    )
    holdout_df.to_csv(HOLDOUT_METRICS_PATH, index=False)

    feature_importance_df = extract_feature_importance(best_artifact.best_estimator)
    if not feature_importance_df.empty:
        feature_importance_df.to_csv(Path(BEST_MODEL_PATH).parent / "feature_importance.csv", index=False)

    save_model_metadata(
        artifact=best_artifact,
        training_rows=len(X_train),
        holdout_rows=len(X_holdout),
        holdout_positive_rate=float(y_holdout.mean()),
    )

    return best_artifact, comparison_df


def save_best_model(best_estimator) -> None:
    """Persist the best trained model pipeline."""
    joblib.dump(best_estimator, BEST_MODEL_PATH)
