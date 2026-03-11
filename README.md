# Stroke Risk Prediction (Deterministic + Threshold-Tuned)

This project predicts stroke risk in a diabetic cohort using the Kaggle Healthcare Stroke Dataset and a Streamlit interface.

The pipeline now uses:
- deterministic preprocessing (no random feature simulation)
- repeated stratified cross-validation with confidence intervals
- a true untouched holdout set for final reporting
- decision-threshold tuning (not fixed at 0.5)
- richer metrics (PR-AUC, sensitivity, specificity, Brier score, calibration curve)
- explainability in app (top factors, optional SHAP)

## Project Structure

```text
project/
  app/
    streamlit_app.py
  data/
    healthcare-dataset-stroke-data.csv
  models/
    best_model.joblib
    model_metadata.json
    model_comparison.csv
    cv_results.csv
    holdout_metrics.csv
    feature_importance.csv
    stroke_distribution.png
    correlation_heatmap.png
    feature_importance.png
    calibration_curve.png
  src/
    config.py
    data_preprocessing.py
    eda.py
    modeling.py
    train.py
  requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Place dataset at:

`data/healthcare-dataset-stroke-data.csv`

## Train

```bash
python -m src.train
```

Training flow:
1. Load dataset and deterministic preprocessing.
2. Keep diabetic cohort (`is_diabetic` based on glucose threshold).
3. Split into development set and untouched holdout set.
4. Hyperparameter tuning with CV.
5. Threshold tuning (`max_f1` or `recall_at_precision`) on training data.
6. Repeated stratified CV metrics + 95% confidence intervals.
7. Final holdout evaluation and artifact export.

## Run App

```bash
streamlit run app/streamlit_app.py
```

App highlights:
- asks user for explicit glucose history summary (mean, SD, CV)
- deterministic prediction for same input
- threshold-aware risk label
- uncertainty messaging based on distance to threshold
- local top factors and optional SHAP view
- downloadable PDF report

## Output Metrics

Artifacts in `models/` include:
- `model_comparison.csv`: model-level comparison
- `cv_results.csv`: fold-level repeated CV results
- `holdout_metrics.csv`: untouched holdout metrics
- `model_metadata.json`: threshold and metric metadata used by app

Reported metrics include:
- accuracy
- balanced accuracy
- precision
- recall (sensitivity)
- specificity
- F1-score
- ROC-AUC
- PR-AUC
- Brier score
- confusion matrix

## Clinical Limitations and Disclaimer

- This project is for educational and research use.
- It is not validated for clinical deployment.
- Dataset is retrospective and may not generalize across populations.
- Predictions must not replace clinician judgment or diagnostic workflow.
- Any practical use requires clinical validation, governance, and regulatory review.
