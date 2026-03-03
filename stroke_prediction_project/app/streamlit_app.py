"""Optional Streamlit app for stroke risk inference."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run training first to create models/best_model.joblib")
    return joblib.load(MODEL_PATH)


def main():
    st.set_page_config(page_title="Stroke Risk Predictor", layout="centered")
    st.title("Stroke Risk Predictor")
    st.caption("Prediction Analysis for Genetically Predisposed Diabetic Patients")

    model = load_model()

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=45.0, step=1.0)
    hypertension = st.selectbox("Hypertension", [0, 1], index=0)
    heart_disease = st.selectbox("Heart Disease", [0, 1], index=0)
    ever_married = st.selectbox("Ever Married", ["No", "Yes"])
    work_type = st.selectbox(
        "Work Type",
        ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
    )
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    avg_glucose_level = st.number_input(
        "Average Glucose Level",
        min_value=40.0,
        max_value=400.0,
        value=110.0,
        step=0.5,
    )
    bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=27.0, step=0.1)
    smoking_status = st.selectbox(
        "Smoking Status",
        ["formerly smoked", "never smoked", "smokes", "Unknown"],
    )
    family_history = st.selectbox("Family History of Stroke", [0, 1], index=0)

    if st.button("Predict Stroke Risk"):
        is_diabetic_proxy = int(avg_glucose_level >= 125.0)
        input_df = pd.DataFrame(
            [
                {
                    "gender": gender,
                    "age": age,
                    "hypertension": hypertension,
                    "heart_disease": heart_disease,
                    "ever_married": ever_married,
                    "work_type": work_type,
                    "Residence_type": residence_type,
                    "avg_glucose_level": avg_glucose_level,
                    "bmi": bmi,
                    "smoking_status": smoking_status,
                    "family_history": family_history,
                    "is_diabetic_proxy": is_diabetic_proxy,
                }
            ]
        )

        prediction = model.predict(input_df)[0]
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1]
        else:
            probability = float(prediction)

        st.subheader("Result")
        st.write(f"Predicted Class: `{int(prediction)}`")
        st.write(f"Estimated Stroke Risk Probability: `{probability:.3f}`")


if __name__ == "__main__":
    main()

