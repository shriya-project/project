"""Exploratory Data Analysis utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve

from .config import (
    PLOT_CALIBRATION_CURVE_PATH,
    PLOT_CORR_HEATMAP_PATH,
    PLOT_FEATURE_IMPORTANCE_PATH,
    PLOT_STROKE_DIST_PATH,
)


def plot_stroke_distribution(df: pd.DataFrame) -> None:
    """Plot and save stroke class distribution."""
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="stroke", hue="stroke", legend=False, palette="Set2")
    plt.title("Stroke Class Distribution")
    plt.xlabel("Stroke (0 = No, 1 = Yes)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOT_STROKE_DIST_PATH, dpi=200)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Plot and save correlation heatmap of numeric features."""
    numeric_df = df.select_dtypes(include=["number"])
    corr_matrix = numeric_df.corr(numeric_only=True)

    plt.figure(figsize=(10, 7))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(PLOT_CORR_HEATMAP_PATH, dpi=200)
    plt.close()


def plot_feature_importance(feature_names, importances) -> None:
    """Plot and save feature importances for the best model."""
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False).head(20)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=imp_df, x="importance", y="feature", hue="feature", legend=False, palette="viridis")
    plt.title("Top 20 Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(PLOT_FEATURE_IMPORTANCE_PATH, dpi=200)
    plt.close()


def plot_calibration_curve(y_true, y_proba, n_bins: int = 10) -> None:
    """Plot and save calibration curve for holdout predictions."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")

    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Positive Rate")
    plt.title("Calibration Curve (Holdout)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(PLOT_CALIBRATION_CURVE_PATH, dpi=200)
    plt.close()

