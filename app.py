"""
Streamlit app (single-page): EMI Predictor

This variant only shows the EMI Predictor UI (no multi-page sidebar).
It supports:
- Single applicant prediction via form inputs
- Batch predictions from a CSV (uploaded or data/applicants.csv) using the 'best' available model found in models/
- Heuristic fallback if no models are available

How it chooses models (best-effort):
1. Looks for explicit filenames (priority):
   - Classifier: models/best_classifier.pkl, models/emi_classifier.pkl
   - Regressor: models/best_regressor.pkl, models/emi_regressor.pkl
2. If not present, scans models/*.pkl and picks the first file that can be loaded and used for .predict_proba (classifier) or .predict (regressor).
3. If no usable model found, uses a simple heuristic fallback so the UI still works.

Place trained model artifacts in models/ (e.g., joblib.dump(pipeline, "models/emi_classifier.pkl")).
"""
import io
import os
import glob
from typing import Tuple, Optional

import pandas as pd
import numpy as np
import streamlit as st

# optional plotting libs
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

st.set_page_config(page_title="EMI Predictor", layout="wide")

# Constants
MODEL_DIR = "models"
DEFAULT_DATA_PATH = "data/applicants.csv"
EXPECTED_COLS = [
    "monthly_salary",
    "current_emi",
    "requested_loan_amount",
    "loan_tenure_months",
    "credit_score",
    "employment_years",
    "expenses_total",
]


# -----------------------
# Model discovery & load
# -----------------------
def try_load(path: str):
    try:
        m = joblib.load(path)
        return m
    except Exception:
        return None


def find_classifier_model() -> Optional[object]:
    # priority names
    candidates = [
        os.path.join(MODEL_DIR, "best_classifier.pkl"),
        os.path.join(MODEL_DIR, "emi_classifier.pkl"),
    ]
    # add any pkl in models/
    candidates += sorted(glob.glob(os.path.join(MODEL_DIR, "*.pkl")))
    seen = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        model = try_load(c)
        if model is None:
            continue
        # check for predict_proba or predict as a fallback
        if hasattr(model, "predict_proba") or hasattr(model, "predict"):
            return model
    return None


def find_regressor_model() -> Optional[object]:
    candidates = [
        os.path.join(MODEL_DIR, "best_regressor.pkl"),
        os.path.join(MODEL_DIR, "emi_regressor.pkl"),
    ]
    candidates += sorted(glob.glob(os.path.join(MODEL_DIR, "*.pkl")))
    seen = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        model = try_load(c)
        if model is None:
            continue
        # prefer something that has predict
        if hasattr(model, "predict"):
            return model
    return None


CLF_MODEL = find_classifier_model()
REG_MODEL = find_regressor_model()


# -----------------------
# Heuristic fallback
# -----------------------
def heuristic_predict(inputs: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    credit = inputs["credit_score"].values
    salary = inputs["monthly_salary"].values
    current_emi = inputs["current_emi"].values
    requested = inputs["requested_loan_amount"].values
    tenure = inputs["loan_tenure_months"].values
    expenses = inputs["expenses_total"].values

    prob = []
    pred = []
    for c, s, cur, req, t, exp in zip(credit, salary, current_emi, requested, tenure, expenses):
        affordability = s * 0.35 - (cur + exp)
        per_month_req = req / max(1, t)
        score_factor = (c - 600) / 400  # scaled shift by credit
        raw = 0.5 * (affordability / max(1.0, per_month_req) + score_factor)
        p = float(np.clip(raw, 0.05, 0.99))
        prob.append(p)
        if p > 0.7:
            pred.append("Eligible")
        elif p > 0.4:
            pred.append("High Risk")
        else:
            pred.append("Not Eligible")

    prob = np.array(prob)
    pred = np.array(pred)
    max_emi = np.maximum(0, salary * 0.35 - expenses)
    return pred, prob, max_emi


# -----------------------
# Preprocessing helpers
# -----------------------
def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in EXPECTED_COLS:
        if c not in df.columns:
            df[c] = 0
    return df[EXPECTED_COLS]


def classify_and_regress(df: pd.DataFrame):
    df_proc = ensure_columns(df.copy())

    # classifier
    if CLF_MODEL is not None:
        try:
            if hasattr(CLF_MODEL, "predict_proba"):
                probs = CLF_MODEL.predict_proba(df_proc)
                # if binary, take column 1; if multi, attempt to map a positive class
                if probs.ndim == 2 and probs.shape[1] >= 2:
                    probs = probs[:, 1]
                else:
                    # fallback to sum over last axis / normalize
                    probs = np.mean(probs, axis=1)
            else:
                # no predict_proba: use predict and map to probability 0.9/0.1
                pred_labels = CLF_MODEL.predict(df_proc)
                probs = np.array([0.9 if p in (1, "Eligible", "eligible") else 0.1 for p in pred_labels])
            labels = np.where(probs >= 0.7, "Eligible", np.where(probs >= 0.4, "High Risk", "Not Eligible"))
        except Exception:
            labels, probs, _ = heuristic_predict(df_proc)
    else:
        labels, probs, _ = heuristic_predict(df_proc)

    # regressor
    if REG_MODEL is not None:
        try:
            max_emi = REG_MODEL.predict(df_proc)
            max_emi = np.maximum(0, np.array(max_emi).astype(float))
        except Exception:
            _, probs, max_emi = heuristic_predict(df_proc)
    else:
        _, probs, max_emi = heuristic_predict(df_proc)

    return labels, probs, max_emi


# -----------------------
# UI components
# -----------------------
def render_gauge(credit_score: float, prob: float):
    if PLOTLY_AVAILABLE:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=credit_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Credit Score (Prob: {prob:.2f})"},
            gauge={
                'axis': {'range': [300, 900]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [300, 579], 'color': "red"},
                    {'range': [580, 669], 'color': "orange"},
                    {'range': [670, 739], 'color': "yellow"},
                    {'range': [740, 799], 'color': "lightgreen"},
                    {'range': [800, 900], 'color': "green"},
                ],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    else:
        min_score, max_score = 300, 900
        frac = (credit_score - min_score) / (max_score - min_score)
        frac = float(np.clip(frac, 0.0, 1.0))
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.barh([0], [frac], color="tab:blue", height=0.5)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels([str(min_score), str(int(min_score + 0.25*(max_score-min_score))),
                            str(int(min_score + 0.5*(max_score-min_score))),
                            str(int(min_score + 0.75*(max_score-min_score))),
                            str(max_score)])
        ax.set_title(f"Credit Score: {credit_score} (Prob: {prob:.2f})")
        st.pyplot(fig)


def show_prediction_card(label: str, max_emi: float, prob: float):
    if label == "Eligible":
        emoji = "✅"
    elif label == "High Risk":
        emoji = "⚠️"
    else:
        emoji = "❌"
    st.markdown(f"### Result: {emoji}  {label}")
    st.write(f"Recommended maximum EMI:  ₹{max_emi:.2f}")
    st.write(f"Probability (model):  {prob:.2f}")


# -----------------------
# Main page (single-page app)
# -----------------------
def main():
    st.title("🏠 EMI Predictor")
    st.write("Single-page app: use the form below for one-off prediction or upload a CSV for batch predictions.")

    # single prediction form
    with st.form(key="single_form"):
        st.subheader("Single applicant prediction")
        monthly_salary = st.number_input("Monthly salary (₹)", min_value=0.0, value=30000.0, step=1000.0)
        current_emi = st.number_input("Current EMI amount (₹)", min_value=0.0, value=5000.0, step=500.0)
        requested_loan_amount = st.number_input("Requested loan amount (₹)", min_value=0.0, value=200000.0, step=10000.0)
        loan_tenure_months = st.number_input("Loan tenure (months)", min_value=1, value=36, step=1)
        credit_score = st.number_input("Credit score (300-900)", min_value=300, max_value=900, value=680, step=1)
        employment_years = st.number_input("Employment years", min_value=0.0, value=3.0, step=0.5)
        rent = st.number_input("Monthly rent (₹)", min_value=0.0, value=8000.0, step=500.0)
        travel = st.number_input("Monthly travel (₹)", min_value=0.0, value=2000.0, step=100.0)
        school = st.number_input("Monthly school/education (₹)", min_value=0.0, value=3000.0, step=100.0)
        other_expenses = st.number_input("Other monthly expenses (₹)", min_value=0.0, value=2000.0, step=100.0)

        expenses_total = rent + travel + school + other_expenses
        st.write("Total monthly expenses (entered):", int(expenses_total))

        submit_btn = st.form_submit_button("Predict Eligibility")

    if submit_btn:
        single = {
            "monthly_salary": monthly_salary,
            "current_emi": current_emi,
            "requested_loan_amount": requested_loan_amount,
            "loan_tenure_months": loan_tenure_months,
            "credit_score": credit_score,
            "employment_years": employment_years,
            "expenses_total": expenses_total,
        }
        df_in = pd.DataFrame([single])
        labels, probs, max_emi = classify_and_regress(df_in)
        label = labels[0]
        prob = float(probs[0])
        recommended = float(max_emi[0])

        c1, c2 = st.columns([1, 1])
        with c1:
            show_prediction_card(label, recommended, prob)
        with c2:
            render_gauge(credit_score, prob)

        st.subheader("Prediction details")
        st.json({
            "label": label,
            "probability": prob,
            "recommended_max_emi": recommended,
            "input_summary": single
        })

    st.markdown("---")
    st.subheader("Batch predictions from CSV")

    uploaded = st.file_uploader("Upload applicants CSV (columns matching: monthly_salary,current_emi,requested_loan_amount,loan_tenure_months,credit_score,employment_years,expenses_total)", type=["csv", "xlsx"])
    use_default = False
    if uploaded is None and os.path.exists(DEFAULT_DATA_PATH):
        if st.checkbox(f"Use default CSV at {DEFAULT_DATA_PATH}"):
            use_default = True

    df = None
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded)
            else:
                df = pd.read_csv(uploaded)
            st.success(f"Loaded {uploaded.name} with {len(df)} rows")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            df = None
    elif use_default:
        try:
            df = pd.read_csv(DEFAULT_DATA_PATH)
            st.info(f"Using {DEFAULT_DATA_PATH} with {len(df)} rows")
        except Exception as e:
            st.error(f"Failed to load default CSV: {e}")
            df = None

    if df is not None:
        st.dataframe(df.head(200))

        if st.button("Run batch predictions on this CSV"):
            # ensure features exist
            df_proc = ensure_columns(df.copy())
            labels, probs, max_emi = classify_and_regress(df_proc)
            df["predicted_label"] = labels
            df["predicted_probability"] = probs
            df["recommended_max_emi"] = max_emi

            st.success("Batch predictions completed.")
            st.dataframe(df.head(200))

            # downloads
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="predictions")
            buf.seek(0)
            st.download_button(
                label="Download predictions as Excel",
                data=buf,
                file_name="predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.download_button(
                label="Download predictions as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )

    # show which models were selected (for transparency)
    st.markdown("---")
    st.write("Model selection summary:")
    st.write(f"Classifier loaded: {'Yes' if CLF_MODEL is not None else 'No'}")
    if CLF_MODEL is not None:
        try:
            st.write(f"Classifier type: {type(CLF_MODEL)}")
            st.write(f"Has predict_proba: {hasattr(CLF_MODEL, 'predict_proba')}")
        except Exception:
            pass
    st.write(f"Regressor loaded: {'Yes' if REG_MODEL is not None else 'No'}")
    if REG_MODEL is not None:
        try:
            st.write(f"Regressor type: {type(REG_MODEL)}")
        except Exception:
            pass

    st.caption("Tip: Put trained models in the models/ folder (e.g. models/emi_classifier.pkl and models/emi_regressor.pkl). If you want the app to automatically choose a 'best' artifact, name it models/best_classifier.pkl and models/best_regressor.pkl.")

if __name__ == "__main__":
    main()
