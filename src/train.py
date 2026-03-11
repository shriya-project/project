"""Entry point to run full stroke prediction pipeline with robust evaluation."""

from __future__ import annotations

from pathlib import Path

from sklearn.model_selection import train_test_split

from .config import DATA_PATH, MODELS_DIR, RANDOM_STATE
from .data_preprocessing import (
    build_preprocessor,
    drop_irrelevant_columns,
    enrich_with_available_glucose_history,
    filter_diabetic_patients,
    load_dataset,
    split_features_target,
)
from .eda import (
    plot_calibration_curve,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_stroke_distribution,
)
from .modeling import save_best_model, train_and_tune_models


def _extract_feature_importance(best_pipeline):
    """Extract feature importance or coefficient magnitudes from trained pipeline."""
    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_[0])
    else:
        return None, None

    return feature_names, importances


def run_training(
    threshold_strategy: str = "max_f1",
    min_precision: float = 0.20,
) -> None:
    """Run end-to-end train, evaluate, compare, and save workflow."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place Kaggle CSV in data/healthcare-dataset-stroke-data.csv"
        )

    # 1) Load and deterministically prepare dataset
    df = load_dataset(str(DATA_PATH))
    df = enrich_with_available_glucose_history(df)
    df = filter_diabetic_patients(df)
    df = drop_irrelevant_columns(df)

    # 2) EDA plots
    plot_stroke_distribution(df)
    plot_correlation_heatmap(df)

    # 3) Split features and target
    X, y = split_features_target(df, target_col="stroke")
    preprocessor = build_preprocessor(X)

    # 4) True untouched holdout split
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 5) Train, tune threshold, and evaluate
    best_artifact, comparison_df = train_and_tune_models(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        threshold_strategy=threshold_strategy,
        min_precision=min_precision,
    )

    # 6) Save best model
    save_best_model(best_artifact.best_estimator)

    # 7) Feature importance and calibration plots
    feature_names, importances = _extract_feature_importance(best_artifact.best_estimator)
    if feature_names is not None:
        plot_feature_importance(feature_names, importances)
    plot_calibration_curve(y_holdout.to_numpy(), best_artifact.holdout_proba, n_bins=10)

    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))

    print(f"\nBest Model: {best_artifact.name}")
    print(f"Decision Threshold: {best_artifact.threshold:.3f} ({best_artifact.threshold_strategy})")

    print("\nHoldout Metrics:")
    for metric_name, metric_value in best_artifact.holdout_metrics.items():
        if metric_name == "confusion_matrix":
            print(f"- {metric_name}: {metric_value}")
        else:
            print(f"- {metric_name}: {metric_value:.6f}" if isinstance(metric_value, float) else f"- {metric_name}: {metric_value}")

    print("\nRepeated CV Summary (mean, 95% CI):")
    for metric_name, summary in best_artifact.cv_summary.items():
        print(
            f"- {metric_name}: {summary['mean']:.4f} "
            f"(95% CI {summary['ci_low']:.4f} to {summary['ci_high']:.4f})"
        )


if __name__ == "__main__":
    run_training()
