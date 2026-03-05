"""Configuration values for the stroke prediction project."""

from pathlib import Path

RANDOM_STATE = 42

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_PATH = DATA_DIR / "healthcare-dataset-stroke-data.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"

PLOT_STROKE_DIST_PATH = MODELS_DIR / "stroke_distribution.png"
PLOT_CORR_HEATMAP_PATH = MODELS_DIR / "correlation_heatmap.png"
PLOT_FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.png"

