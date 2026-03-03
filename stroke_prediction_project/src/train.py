"""Entry point to run full stroke prediction pipeline."""

from __future__ import annotations

from pathlib import Path

from sklearn.model_selection import train_test_split

from .config import DATA_PATH, MODELS_DIR, RANDOM_STATE
from .data_preprocessing import (
    add_synthetic_features,
    build_preprocessor,
    drop_irrelevant_columns,
    load_dataset,
    split_features_target,
)
from .eda import plot_correlation_heatmap, plot_feature_importance, plot_stroke_distribution
from .modeling import save_best_model, train_and_tune_models


def _extract_feature_importance(best_pipeline, model_name: str):
    """Extract feature importance or coefficient magnitudes from trained pipeline."""
    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_[0])
    else:
        importances = None

    if importances is None:
        return None, None
    return feature_names, importances


def run_training() -> None:
    """Run end-to-end train, evaluate, compare, and save workflow."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place Kaggle CSV in data/healthcare-dataset-stroke-data.csv"
        )

    df = load_dataset(str(DATA_PATH))
    df = add_synthetic_features(df, random_state=RANDOM_STATE)
    df = drop_irrelevant_columns(df)

    # EDA before splitting.
    plot_stroke_distribution(df)
    plot_correlation_heatmap(df)

    X, y = split_features_target(df, target_col="stroke")
    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    best_artifact, comparison_df = train_and_tune_models(
        preprocessor=preprocessor,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    save_best_model(best_artifact.best_estimator.best_estimator_)

    feature_names, importances = _extract_feature_importance(
        best_pipeline=best_artifact.best_estimator.best_estimator_,
        model_name=best_artifact.name,
    )
    if feature_names is not None:
        plot_feature_importance(feature_names, importances)

    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    print(f"\nBest Model (by stroke-class recall): {best_artifact.name}")
    print("Best Model Metrics:")
    for metric_name, metric_value in best_artifact.metrics.items():
        print(f"- {metric_name}: {metric_value}")


if __name__ == "__main__":
    run_training()

