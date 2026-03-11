"""Streamlit app for stroke risk prediction with polished UI and deterministic inference."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"
FEATURE_IMPORTANCE_PATH = BASE_DIR / "models" / "feature_importance.csv"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run training first to create models/best_model.joblib")
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH, "r", encoding="utf-8") as fp:
        return json.load(fp)


@st.cache_data
def load_feature_importance():
    if not FEATURE_IMPORTANCE_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


def clean_feature_name(name: str) -> str:
    if "__" in name:
        return name.split("__", 1)[1]
    return name


def expected_input_columns(model) -> list[str]:
    preprocessor = model.named_steps["preprocessor"]
    if hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)
    return []


def local_top_factors(model, input_df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Return top local factors via linear contribution or weighted-importance proxy."""
    preprocessor = model.named_steps["preprocessor"]
    estimator = model.named_steps["model"]

    transformed = preprocessor.transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "coef_"):
        impact = transformed[0] * estimator.coef_[0]
    elif hasattr(estimator, "feature_importances_"):
        impact = transformed[0] * estimator.feature_importances_
    else:
        return pd.DataFrame()

    ranked_idx = np.argsort(np.abs(impact))[::-1][:top_n]
    factors = pd.DataFrame(
        {
            "feature": [clean_feature_name(feature_names[i]) for i in ranked_idx],
            "impact": [float(impact[i]) for i in ranked_idx],
        }
    )
    factors["direction"] = np.where(factors["impact"] >= 0, "Increases risk", "Decreases risk")
    factors["magnitude"] = factors["impact"].abs()
    return factors[["feature", "impact", "direction", "magnitude"]]


def shap_top_factors(model, input_df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame | None:
    """Try SHAP explanation if shap is installed; return None on failure."""
    try:
        import shap
    except Exception:
        return None

    try:
        preprocessor = model.named_steps["preprocessor"]
        estimator = model.named_steps["model"]
        transformed = preprocessor.transform(input_df)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        feature_names = preprocessor.get_feature_names_out()
        explainer = shap.Explainer(estimator, transformed)
        values = explainer(transformed).values[0]

        if values.ndim > 1:
            values = values[:, 0]

        ranked_idx = np.argsort(np.abs(values))[::-1][:top_n]
        result = pd.DataFrame(
            {
                "feature": [clean_feature_name(feature_names[i]) for i in ranked_idx],
                "shap_value": [float(values[i]) for i in ranked_idx],
            }
        )
        result["direction"] = np.where(result["shap_value"] >= 0, "Increases risk", "Decreases risk")
        return result
    except Exception:
        return None


def prediction_uncertainty(probability: float, threshold: float) -> tuple[str, str]:
    margin = abs(probability - threshold)
    if margin < 0.05:
        return "High uncertainty: output is very close to the decision threshold.", "uncertain-high"
    if margin < 0.15:
        return "Moderate uncertainty: use with caution and additional checks.", "uncertain-medium"
    return "Lower uncertainty: output is relatively far from decision threshold.", "uncertain-low"


def pct(value) -> str:
    if value is None:
        return "N/A"
    return f"{100 * float(value):.1f}%"


def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

        :root {
            --bg-a: #f7fbff;
            --bg-b: #e7f1fb;
            --card: #ffffff;
            --text: #000000;
            --line: rgba(0, 0, 0, 0.16);
            --line-strong: rgba(0, 0, 0, 0.34);
            --accent: #075985;
            --accent-2: #0e7490;
            --shadow-soft: 0 16px 36px rgba(0, 0, 0, 0.08);
            --shadow-strong: 0 14px 30px rgba(7, 89, 133, 0.2);
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(920px 320px at -12% -20%, rgba(56, 189, 248, 0.3), transparent 60%),
                radial-gradient(760px 340px at 108% -8%, rgba(14, 165, 233, 0.24), transparent 62%),
                linear-gradient(155deg, var(--bg-a) 0%, var(--bg-b) 100%);
            color: var(--text) !important;
        }

        .main .block-container {
            max-width: 1150px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] *,
        p, div, label, span, li, a, small {
            color: #000000 !important;
            font-family: "Space Grotesk", "Segoe UI", Tahoma, sans-serif !important;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: "Space Grotesk", "Trebuchet MS", sans-serif !important;
            letter-spacing: -0.01em;
            color: #000000 !important;
            font-weight: 700 !important;
            text-shadow: 0 1px 0 rgba(255, 255, 255, 0.7);
        }

        .hero-wrap, .surface, .risk-banner, [data-testid="stMetric"], [data-testid="stForm"] {
            animation: fadeUp 520ms cubic-bezier(.19,1,.22,1) both;
        }

        .hero-wrap {
            background: linear-gradient(130deg, rgba(255,255,255,0.95), rgba(240,249,255,0.94));
            border: 1px solid var(--line-strong);
            border-radius: 22px;
            padding: 1.15rem 1.25rem 1.15rem 1.25rem;
            box-shadow: var(--shadow-soft);
            margin-bottom: 1.1rem;
            position: relative;
            overflow: hidden;
        }

        .hero-wrap::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(110deg, transparent 10%, rgba(255,255,255,0.55) 45%, transparent 80%);
            transform: translateX(-120%);
            animation: shimmer 4s ease-in-out infinite;
        }

        .hero-kicker {
            font-family: "Space Grotesk", "Trebuchet MS", sans-serif !important;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.75rem;
            color: #000000 !important;
            margin-bottom: 0.18rem;
            position: relative;
            z-index: 2;
        }

        .hero-title {
            font-family: "Space Grotesk", "Trebuchet MS", sans-serif !important;
            font-size: clamp(1.5rem, 2vw, 2.1rem);
            font-weight: 700;
            margin: 0.1rem 0 0.45rem 0;
            color: #000000 !important;
            position: relative;
            z-index: 2;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.45rem;
            position: relative;
            z-index: 2;
        }

        .chip {
            border-radius: 999px;
            border: 1px solid var(--line-strong);
            padding: 0.3rem 0.72rem;
            background: rgba(255,255,255,0.96);
            font-size: 0.77rem;
            color: #000000 !important;
            font-family: "Space Grotesk", "Trebuchet MS", sans-serif !important;
        }

        .surface {
            background: var(--card);
            border: 1px solid var(--line-strong);
            border-radius: 18px;
            padding: 1rem 1rem 0.7rem 1rem;
            box-shadow: var(--shadow-soft);
            margin-top: 0.25rem;
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        .surface:hover {
            transform: translateY(-2px);
            box-shadow: 0 18px 30px rgba(0,0,0,0.1);
        }

        .surface-title {
            font-family: "Space Grotesk", "Trebuchet MS", sans-serif !important;
            font-size: 0.92rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: #000000 !important;
            margin-bottom: 0.45rem;
            border-bottom: 1px dashed rgba(0,0,0,0.25);
            padding-bottom: 0.35rem;
        }

        .risk-banner {
            border-radius: 16px;
            border: 1px solid var(--line-strong);
            padding: 0.85rem 1rem;
            margin-bottom: 0.8rem;
            box-shadow: var(--shadow-soft);
            animation: popIn 380ms cubic-bezier(.2,.8,.2,1) both;
        }

        .risk-high {
            background: linear-gradient(125deg, rgba(254,226,226,0.85), rgba(255,255,255,0.95));
            border-left: 6px solid #be123c;
        }

        .risk-low {
            background: linear-gradient(125deg, rgba(220,252,231,0.82), rgba(255,255,255,0.95));
            border-left: 6px solid #15803d;
        }

        .uncertain-high, .uncertain-medium, .uncertain-low {
            color: #000000 !important;
            font-weight: 600;
        }

        .info-strip {
            border-left: 5px solid var(--accent);
            background: rgba(255,255,255,0.92);
            border-radius: 10px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.9rem;
            color: #000000 !important;
            box-shadow: var(--shadow-soft);
        }

        [data-testid="stForm"] {
            border: 1px solid var(--line-strong);
            border-radius: 18px;
            padding: 0.8rem 0.8rem 0.2rem 0.8rem;
            background: rgba(255,255,255,0.88);
            box-shadow: var(--shadow-soft);
        }

        [data-testid="stMetric"] {
            border: 1px solid var(--line-strong);
            border-radius: 14px;
            padding: 0.55rem 0.75rem;
            background: #ffffff;
            box-shadow: var(--shadow-soft);
        }

        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
            color: #000000 !important;
            font-weight: 700 !important;
        }

        [data-baseweb="tab-list"] {
            gap: 0.25rem;
            border-bottom: 1px solid var(--line-strong);
            margin-bottom: 0.5rem;
        }

        button[role="tab"] {
            color: #000000 !important;
            font-weight: 700 !important;
            border-radius: 12px 12px 0 0 !important;
            border: 1px solid transparent !important;
            background: rgba(255,255,255,0.65) !important;
            transition: transform 140ms ease, background 140ms ease !important;
        }

        button[role="tab"][aria-selected="true"] {
            background: #ffffff !important;
            border-color: var(--line-strong) !important;
            border-bottom-color: #ffffff !important;
            transform: translateY(-1px);
        }

        button[role="tab"]:hover {
            background: rgba(255,255,255,0.9) !important;
        }

        .stButton > button, [data-testid="stFormSubmitButton"] button {
            border-radius: 999px !important;
            border: none !important;
            background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            letter-spacing: 0.02em !important;
            padding: 0.55rem 1.1rem !important;
            box-shadow: var(--shadow-strong) !important;
            transition: transform 180ms ease, filter 180ms ease, box-shadow 180ms ease !important;
        }

        .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
            transform: translateY(-2px) scale(1.01) !important;
            filter: brightness(1.06) !important;
            box-shadow: 0 14px 28px rgba(7,89,133,0.28) !important;
        }

        [data-baseweb="input"] input,
        [data-baseweb="select"] *,
        textarea {
            color: #000000 !important;
        }

        [data-testid="stAlert"] {
            border-radius: 12px !important;
            border: 1px solid var(--line-strong) !important;
        }

        [data-testid="stAlert"] * {
            color: #000000 !important;
        }

        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #0ea5e9, #0369a1) !important;
            animation: flowBar 1.1s ease-in-out;
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes popIn {
            0% { opacity: 0; transform: scale(0.98) translateY(8px); }
            100% { opacity: 1; transform: scale(1) translateY(0); }
        }

        @keyframes shimmer {
            0% { transform: translateX(-120%); }
            60%, 100% { transform: translateX(120%); }
        }

        @keyframes flowBar {
            from { filter: saturate(0.8); }
            to { filter: saturate(1.15); }
        }

        @media (max-width: 900px) {
            .main .block-container {
                padding-left: 0.8rem;
                padding-right: 0.8rem;
            }
            .hero-wrap { padding: 1rem; border-radius: 18px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero_panel(model_name: str, threshold: float, threshold_strategy: str):
    st.markdown(
        f"""
        <div class="hero-wrap">
          <div class="hero-kicker">Clinical Decision Support Prototype</div>
          <div class="hero-title">Stroke Risk Intelligence Dashboard</div>
          <div style="color:#000000; font-size:0.94rem; position:relative; z-index:2;">
            Deterministic prediction, tuned thresholding, explainability overlays, and uncertainty signals.
          </div>
          <div class="chip-row">
            <span class="chip">Model: {model_name}</span>
            <span class="chip">Threshold: {threshold:.3f}</span>
            <span class="chip">Strategy: {threshold_strategy}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def generate_pdf_report(
    patient_data: dict,
    history_metrics: dict,
    prediction: int,
    probability: float,
    threshold: float,
    uncertainty_note: str,
):
    """Generate downloadable PDF patient report."""
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setFont("Helvetica", 11)

    y = 770
    pdf.drawString(170, y, "Stroke Risk Prediction Report (Research Prototype)")
    y -= 30

    pdf.drawString(50, y, "Patient Information:")
    y -= 20
    for key, value in patient_data.items():
        pdf.drawString(60, y, f"{key}: {value}")
        y -= 16

    y -= 8
    pdf.drawString(50, y, "Glucose History Inputs:")
    y -= 20
    for key, value in history_metrics.items():
        pdf.drawString(60, y, f"{key}: {value}")
        y -= 16

    y -= 8
    risk_label = "High Stroke Risk" if prediction == 1 else "Low Stroke Risk"
    pdf.drawString(50, y, "Prediction:")
    y -= 20
    pdf.drawString(60, y, f"Risk Category: {risk_label}")
    y -= 16
    pdf.drawString(60, y, f"Probability: {probability:.3f}")
    y -= 16
    pdf.drawString(60, y, f"Decision Threshold: {threshold:.3f}")
    y -= 16
    pdf.drawString(60, y, f"Uncertainty: {uncertainty_note}")

    y -= 30
    pdf.setFont("Helvetica-Oblique", 9)
    pdf.drawString(50, y, "Clinical disclaimer: For educational/research use only, not medical diagnosis.")

    pdf.save()
    buffer.seek(0)
    return buffer


def main():
    st.set_page_config(page_title="Stroke Risk Predictor", layout="wide", initial_sidebar_state="collapsed")
    inject_custom_css()

    model = load_model()
    metadata = load_metadata()
    model_name = metadata.get("model_name", "Unknown Model")
    threshold = float(metadata.get("decision_threshold", 0.5))
    threshold_strategy = metadata.get("threshold_strategy", "default_0.5")
    expected_cols = expected_input_columns(model)

    hero_panel(model_name=model_name, threshold=threshold, threshold_strategy=threshold_strategy)

    st.markdown(
        """
        <div class="info-strip">
          Research tool only. Not a medical device. Use alongside clinician review and validated diagnostics.
        </div>
        """,
        unsafe_allow_html=True,
    )

    holdout_metrics = metadata.get("holdout_metrics", {})
    if holdout_metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Holdout Accuracy", pct(holdout_metrics.get("accuracy")))
        m2.metric("Holdout Recall", pct(holdout_metrics.get("recall")))
        m3.metric("Holdout Specificity", pct(holdout_metrics.get("specificity")))
        m4.metric("Holdout PR-AUC", pct(holdout_metrics.get("pr_auc")))

    with st.form("prediction_form", clear_on_submit=False):
        left, right = st.columns([1.2, 1], gap="large")

        with left:
            st.markdown('<div class="surface"><div class="surface-title">Patient Profile</div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            with a:
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                age = st.number_input("Age", min_value=0.0, max_value=120.0, value=50.0)
                hypertension = st.selectbox("Hypertension", [0, 1])
                heart_disease = st.selectbox("Heart Disease", [0, 1])
                smoking_status = st.selectbox(
                    "Smoking Status",
                    ["formerly smoked", "never smoked", "smokes", "Unknown"],
                )
            with b:
                ever_married = st.selectbox("Ever Married", ["No", "Yes"])
                work_type = st.selectbox(
                    "Work Type",
                    ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
                )
                residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
                avg_glucose_level = st.number_input(
                    "Average Glucose (mg/dL)",
                    min_value=40.0,
                    max_value=400.0,
                    value=150.0,
                )
                bmi = st.number_input("BMI", min_value=10.0, max_value=80.0, value=28.0)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="surface"><div class="surface-title">Recent Glucose History</div>', unsafe_allow_html=True)
            glucose_mean = st.number_input(
                "Mean Glucose (mg/dL)",
                min_value=40.0,
                max_value=400.0,
                value=150.0,
                help="Use your recent logged readings.",
            )
            glucose_sd = st.number_input(
                "Glucose SD (mg/dL)",
                min_value=0.0,
                max_value=200.0,
                value=20.0,
            )
            default_cv = glucose_sd / glucose_mean if glucose_mean > 0 else 0.0
            glucose_cv = st.number_input(
                "Glucose CV",
                min_value=0.0,
                max_value=2.0,
                value=float(default_cv),
                help="Coefficient of variation (SD / Mean).",
            )
            is_diabetic = int(glucose_mean >= 126.0)
            st.caption(
                f"Derived diabetic flag: `{is_diabetic}` | Decision threshold: `{threshold:.3f}`"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("Analyze Stroke Risk", use_container_width=True)

    if not submitted:
        return

    all_features = {
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

    if not expected_cols:
        st.error("Model schema unavailable; retrain model and try again.")
        st.stop()

    missing_required = [col for col in expected_cols if col not in all_features]
    if missing_required:
        st.error(f"Missing required model fields: {missing_required}")
        st.stop()

    model_input = {col: all_features[col] for col in expected_cols}
    input_df = pd.DataFrame([model_input], columns=expected_cols)

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(input_df)[0][1])
    else:
        probability = float(model.predict(input_df)[0])

    prediction = int(probability >= threshold)
    uncertainty_note, uncertainty_class = prediction_uncertainty(probability, threshold)
    risk_label = "High Stroke Risk" if prediction == 1 else "Low Stroke Risk"
    risk_class = "risk-high" if prediction == 1 else "risk-low"

    st.markdown(
        f"""
        <div class="risk-banner {risk_class}">
          <h3 style="margin:0; font-family:'Space Grotesk','Trebuchet MS',sans-serif;">{risk_label}</h3>
          <div style="margin-top:0.3rem;">Predicted probability: <strong>{probability:.3f}</strong> (threshold {threshold:.3f})</div>
          <div class="{uncertainty_class}" style="margin-top:0.28rem;">{uncertainty_note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    confidence = min(1.0, abs(probability - threshold) / 0.5)
    st.progress(confidence, text=f"Decision confidence score: {confidence:.2f}")

    if is_diabetic == 0:
        st.warning("Input does not meet diabetic threshold (mean glucose < 126 mg/dL).")

    summary_tab, explain_tab, quality_tab, report_tab = st.tabs(
        ["Risk Summary", "Explainability", "Model Quality", "Report"]
    )

    with summary_tab:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Probability", f"{100 * probability:.1f}%")
        s2.metric("Threshold", f"{100 * threshold:.1f}%")
        s3.metric("Predicted Class", str(prediction))
        s4.metric("Diabetic Flag", str(is_diabetic))

        st.markdown("**Input Snapshot**")
        snapshot = pd.DataFrame(
            [
                {
                    "Age": age,
                    "Avg Glucose": avg_glucose_level,
                    "Glucose Mean": glucose_mean,
                    "Glucose SD": glucose_sd,
                    "Glucose CV": glucose_cv,
                    "BMI": bmi,
                    "Hypertension": hypertension,
                    "Heart Disease": heart_disease,
                }
            ]
        )
        st.dataframe(snapshot, use_container_width=True, hide_index=True)

    with explain_tab:
        st.markdown("**Top Local Factors**")
        local_factors_df = local_top_factors(model, input_df)
        if not local_factors_df.empty:
            st.dataframe(local_factors_df, use_container_width=True, hide_index=True)
        else:
            st.info("Local factor view unavailable for this model.")

        show_shap = st.checkbox("Show SHAP factors (optional)", value=False)
        if show_shap:
            shap_df = shap_top_factors(model, input_df)
            if shap_df is None:
                st.info("SHAP is not available. Install `shap` to enable SHAP explanations.")
            else:
                st.dataframe(shap_df, use_container_width=True, hide_index=True)

        global_importance = load_feature_importance()
        if not global_importance.empty:
            st.markdown("**Global Feature Influence**")
            top_global = global_importance.head(10).copy()
            top_global["feature"] = top_global["feature"].apply(clean_feature_name)
            st.dataframe(top_global, use_container_width=True, hide_index=True)
            chart_df = top_global.set_index("feature")[["importance"]]
            st.bar_chart(chart_df)

    with quality_tab:
        if holdout_metrics:
            q1, q2, q3, q4, q5 = st.columns(5)
            q1.metric("Accuracy", pct(holdout_metrics.get("accuracy")))
            q2.metric("Precision", pct(holdout_metrics.get("precision")))
            q3.metric("Recall", pct(holdout_metrics.get("recall")))
            q4.metric("Specificity", pct(holdout_metrics.get("specificity")))
            q5.metric("ROC-AUC", pct(holdout_metrics.get("roc_auc")))

            st.markdown("**Calibration and Reliability Metrics**")
            c1, c2, c3 = st.columns(3)
            c1.metric("PR-AUC", pct(holdout_metrics.get("pr_auc")))
            brier = holdout_metrics.get("brier_score")
            c2.metric("Brier Score", f"{brier:.3f}" if brier is not None else "N/A")
            c3.metric("Balanced Accuracy", pct(holdout_metrics.get("balanced_accuracy")))

        cv_summary = metadata.get("cv_summary", {})
        if cv_summary:
            st.markdown("**Repeated CV Summary (Mean with 95% CI)**")
            rows = []
            for metric, values in cv_summary.items():
                rows.append(
                    {
                        "metric": metric,
                        "mean": values.get("mean"),
                        "ci_low": values.get("ci_low"),
                        "ci_high": values.get("ci_high"),
                    }
                )
            cv_df = pd.DataFrame(rows)
            st.dataframe(cv_df, use_container_width=True, hide_index=True)

    with report_tab:
        patient_data = {
            "Gender": gender,
            "Age": age,
            "Hypertension": hypertension,
            "Heart Disease": heart_disease,
            "Ever Married": ever_married,
            "Work Type": work_type,
            "Residence Type": residence_type,
            "Average Glucose": avg_glucose_level,
            "BMI": bmi,
            "Smoking Status": smoking_status,
        }
        history_data = {
            "Mean Glucose": glucose_mean,
            "Glucose SD": glucose_sd,
            "Glucose CV": glucose_cv,
            "Is Diabetic": is_diabetic,
        }

        pdf_file = generate_pdf_report(
            patient_data=patient_data,
            history_metrics=history_data,
            prediction=prediction,
            probability=probability,
            threshold=threshold,
            uncertainty_note=uncertainty_note,
        )
        st.download_button(
            label="Download Patient Report (PDF)",
            data=pdf_file,
            file_name="stroke_prediction_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
