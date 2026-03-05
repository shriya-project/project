"""Streamlit app for glucose variability-based stroke risk prediction."""

from __future__ import annotations

from pathlib import Path
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run training first to create models/best_model.joblib")
    return joblib.load(MODEL_PATH)


def simulate_glucose_variability(avg_glucose):
    """
    Simulate retrospective glucose readings and compute:
    mean, standard deviation, and coefficient of variation.
    """
    readings = np.random.normal(loc=avg_glucose, scale=20, size=10)
    mean_glucose = np.mean(readings)
    sd_glucose = np.std(readings)
    cv_glucose = sd_glucose / mean_glucose
    return mean_glucose, sd_glucose, cv_glucose


def generate_pdf_report(patient_data: dict, metrics: dict, prediction: int, probability: float):
    """
    Generate downloadable PDF patient report.
    """
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setFont("Helvetica", 12)

    y = 750

    pdf.drawString(180, y, "Stroke Risk Prediction Report")
    y -= 40

    pdf.drawString(50, y, "Patient Information:")
    y -= 25

    for key, value in patient_data.items():
        pdf.drawString(60, y, f"{key}: {value}")
        y -= 18

    y -= 10
    pdf.drawString(50, y, "Glucose Variability Metrics:")
    y -= 25

    for key, value in metrics.items():
        pdf.drawString(60, y, f"{key}: {value}")
        y -= 18

    y -= 10
    pdf.drawString(50, y, "Prediction Result:")
    y -= 25

    risk_label = "High Stroke Risk" if prediction == 1 else "Low Stroke Risk"

    pdf.drawString(60, y, f"Predicted Class: {prediction}")
    y -= 18
    pdf.drawString(60, y, f"Risk Category: {risk_label}")
    y -= 18
    pdf.drawString(60, y, f"Stroke Probability: {probability:.3f}")

    pdf.save()
    buffer.seek(0)
    return buffer


def main():
    st.set_page_config(page_title="Stroke Risk Predictor", layout="centered")
    st.title("Glucose Variability-Based Stroke Risk Predictor")
    st.caption("Retrospective Predictive Analysis in Diabetic Patients")

    model = load_model()

    st.header("Patient Information")

    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=50.0)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])
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
        value=150.0,
    )
    bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=28.0)
    smoking_status = st.selectbox(
        "Smoking Status",
        ["formerly smoked", "never smoked", "smokes", "Unknown"],
    )

    if st.button("Predict Stroke Risk"):

        # Simulate glucose variability
        glucose_mean, glucose_sd, glucose_cv = simulate_glucose_variability(
            avg_glucose_level
        )

        st.subheader("Glucose Variability Metrics")
        st.write(f"Mean Glucose: `{glucose_mean:.2f}`")
        st.write(f"Glucose Standard Deviation: `{glucose_sd:.2f}`")
        st.write(f"Glucose Coefficient of Variation: `{glucose_cv:.3f}`")

        is_diabetic = int(glucose_mean >= 126)

        if is_diabetic == 0:
            st.warning("Patient does not meet diabetic threshold. Model trained on diabetic cohort.")

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
                    "glucose_mean": glucose_mean,
                    "glucose_sd": glucose_sd,
                    "glucose_cv": glucose_cv,
                    "is_diabetic": is_diabetic,
                }
            ]
        )

        prediction = model.predict(input_df)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1]
        else:
            probability = float(prediction)

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f"High Stroke Risk Detected (Probability: {probability:.3f})")
        else:
            st.success(f"Low Stroke Risk (Probability: {probability:.3f})")

        # Prepare data for PDF
        patient_data = {
            "Gender": gender,
            "Age": age,
            "Hypertension": hypertension,
            "Heart Disease": heart_disease,
            "Ever Married": ever_married,
            "Work Type": work_type,
            "Residence Type": residence_type,
            "Average Glucose Level": avg_glucose_level,
            "BMI": bmi,
            "Smoking Status": smoking_status,
        }

        metrics = {
            "Mean Glucose": f"{glucose_mean:.2f}",
            "Glucose SD": f"{glucose_sd:.2f}",
            "Glucose CV": f"{glucose_cv:.3f}",
        }

        pdf_file = generate_pdf_report(
            patient_data,
            metrics,
            prediction,
            probability,
        )

        st.download_button(
            label="Download Patient Report",
            data=pdf_file,
            file_name="stroke_prediction_report.pdf",
            mime="application/pdf",
        )


if __name__ == "__main__":
    main()