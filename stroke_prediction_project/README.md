# Prediction Analysis for Stroke Risk in Genetically Predisposed Diabetic Patients

Complete machine learning project in Python using the Kaggle Healthcare Stroke Dataset.

## Project Structure

```text
stroke_prediction_project/
├── app/
│   └── streamlit_app.py
├── data/
│   └── healthcare-dataset-stroke-data.csv   # place Kaggle CSV here
├── models/
│   ├── best_model.joblib                    # generated after training
│   ├── model_comparison.csv                 # generated after training
│   └── *.png                                # generated plots
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── modeling.py
│   └── train.py
└── requirements.txt
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the Kaggle Healthcare Stroke Dataset CSV and place it at:

`data/healthcare-dataset-stroke-data.csv`

## Train Pipeline

Run from project root (`stroke_prediction_project`):

```bash
python -m src.train
```

This performs:
- Missing value handling via imputers
- Dropping irrelevant column (`id`)
- One-hot encoding categorical features
- Scaling numeric features
- Synthetic `family_history` feature generation:
  - 30% probability of value 1
  - +5% synthetic stroke probability adjustment for `family_history=1`
- EDA plots:
  - Stroke class distribution
  - Correlation heatmap
  - Feature importance plot (best model)
- Train/test split (80/20)
- SMOTE balancing on training data
- Model training and GridSearchCV tuning:
  - Logistic Regression
  - Random Forest
  - XGBoost (if installed)
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - Confusion Matrix
- Best model selection by recall for stroke class (class `1`)
- Model saving with `joblib` to `models/best_model.joblib`

## Optional Streamlit App

After training:

```bash
streamlit run app/streamlit_app.py
```

The app loads `models/best_model.joblib` and predicts stroke risk for user inputs.
