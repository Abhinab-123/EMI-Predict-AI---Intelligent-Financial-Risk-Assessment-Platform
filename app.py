"""
Streamlit app: EMI Predictor — EMI-Predict-AI

Pages:
- EMI Predictor
- Data Exploration
- Model Metrics
- Admin / Data Upload

How to run:
- pip install streamlit pandas scikit-learn plotly seaborn openpyxl joblib
- streamlit run app.py

Notes:
- Place trained models (classification + regression) at:
    models/emi_classifier.pkl
    models/emi_regressor.pkl
  The code will gracefully fall back to a simple heuristic if not found.
- Provide a dataset at data/applicants.csv for exploration (optional).
- Optional saved metrics can live at models/metrics.json
"""

import io
import os
import json
from typing import Tuple

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)

import joblib

# ---------------------------------------------------------------------
# Utils: load models, helpers
# ---------------------------------------------------------------------
MODEL_CLASS_PATH = "models/emi_classifier.pkl"
MODEL_REG_PATH = "models/emi_regressor.pkl"
DEFAULT_DATA_PATH = "data/applicants.csv"
METRICS_PATH = "models/metrics.json"  # optional saved metrics from training / MLflow export

st.set_page_config(page_title="EMI Predictor", layout="wide")


@st.cache_resource
def load_model(path: str):
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            return model
        except Exception as e:
            st.warning(f"Failed to load model at {path}: {e}")
            return None
    return None


clf_model = load_model(MODEL_CLASS_PATH)
reg_model = load_model(MODEL_REG_PATH)


def heuristic_predict(inputs: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple fallback classifier/regressor when models are not available.
    Classifier: produce a probability based on affordability and credit score.
    Regressor: recommend max_emi = max(0, salary * 0.35 - expenses)
    """
    credit = inputs["credit_score"].values
    salary = inputs["monthly_salary"].values
    current_emi = inputs["current_emi"].values
    requested = inputs["requested_loan_amount"].values
    tenure = inputs["loan_tenure_months"].values
    expenses = inputs["expenses_total"].values

    prob = []
    pred = []
    for c, s, cur, req, t, exp in zip(credit, salary, current_emi, requested, tenure, expenses):
        affordability = s * 0.4 - (cur + exp)
        # score factor shifts probability by credit
        score_factor = (c - 600) / 400  # roughly maps 200->-1, 600->0, 900->0.75
        # compare affordability to per-month requested repayment
        per_month_req = req / max(1, t)
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

    max_emi = np.maximum(0, salary * 0.35 - expenses)  # conservative recommendation
    return pred, prob, max_emi


def preprocess_input(single_input: dict) -> pd.DataFrame:
    df = pd.DataFrame([single_input])
    expected_cols = [
        "monthly_salary",
        "current_emi",
        "requested_loan_amount",
        "loan_tenure_months",
        "credit_score",
        "employment_years",
        "expenses_total",
    ]
    for c in expected_cols:
        if c not in df:
            df[c] = 0
    return df[expected_cols]


def classify_and_regress(df: pd.DataFrame):
    # Try classifier
    if clf_model is not None:
        try:
            # predict_proba expected; fallback if fails
            probs = clf_model.predict_proba(df)[:, 1]
            labels = np.where(probs >= 0.7, "Eligible", np.where(probs >= 0.4, "High Risk", "Not Eligible"))
        except Exception:
            labels, probs, _ = heuristic_predict(df)
    else:
        labels, probs, _ = heuristic_predict(df)

    # Try regressor
    if reg_model is not None:
        try:
            max_emi = reg_model.predict(df)
            max_emi = np.maximum(0, max_emi)
        except Exception:
            _, probs, max_emi = heuristic_predict(df)
    else:
        _, probs, max_emi = heuristic_predict(df)

    return labels, probs, max_emi


# ---------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------
def render_gauge(credit_score: float, prob: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=credit_score,
        number={'suffix': ""},
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


# ---------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------
def page_predict():
    st.title("🏠 EMI Predictor")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Applicant inputs")
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
        st.write("Total monthly expenses (entered):", expenses_total)

        if st.button("Predict Eligibility"):
            single = {
                "monthly_salary": monthly_salary,
                "current_emi": current_emi,
                "requested_loan_amount": requested_loan_amount,
                "loan_tenure_months": loan_tenure_months,
                "credit_score": credit_score,
                "employment_years": employment_years,
                "expenses_total": expenses_total,
            }
            df_in = preprocess_input(single)
            labels, probs, max_emi = classify_and_regress(df_in)
            label = labels[0]
            prob = float(probs[0])
            recommended = float(max_emi[0])

            c3, c4 = st.columns([1, 1])
            with c3:
                show_prediction_card(label, recommended, prob)
            with c4:
                render_gauge(credit_score, prob)

            st.subheader("Prediction details")
            st.json({
                "label": label,
                "probability": prob,
                "recommended_max_emi": recommended,
                "input_summary": single
            })


def page_explore():
    st.title("📊 Data Exploration")

    uploaded = st.file_uploader("Upload applicants CSV (optional)", type=["csv", "xlsx"])
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
            return
    else:
        if os.path.exists(DEFAULT_DATA_PATH):
            try:
                df = pd.read_csv(DEFAULT_DATA_PATH)
                st.info(f"Using {DEFAULT_DATA_PATH} with {len(df)} rows")
            except Exception as e:
                st.warning(f"Failed to load {DEFAULT_DATA_PATH}: {e}")

    if df is None or df.empty:
        st.write("No dataset available. Upload a CSV or place one at data/applicants.csv.")
        return

    st.dataframe(df.head(50))

    if "eligibility" in df.columns:
        avg_salary = df.groupby("eligibility")["monthly_salary"].mean().reset_index()
        st.subheader("Average salary by eligibility")
        st.bar_chart(avg_salary.rename(columns={"monthly_salary": "avg_monthly_salary"}).set_index("eligibility"))
    else:
        st.info("No 'eligibility' column found in dataset; charts will use computed estimates where possible.")

    st.subheader("Credit score distribution")
    if "credit_score" in df.columns:
        st.bar_chart(df["credit_score"].value_counts().sort_index())
    else:
        st.write("No credit_score column available.")

    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] >= 2:
        st.subheader("Correlation heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")


def page_metrics():
    st.title("⚙️ Model Metrics")

    metrics = None
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH) as f:
                metrics = json.load(f)
        except Exception:
            metrics = None

    if metrics:
        st.subheader("Saved metrics")
        st.json(metrics)
        if "confusion_matrix" in metrics:
            cm = np.array(metrics["confusion_matrix"])
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
    else:
        st.info("No saved metrics found at models/metrics.json. You can upload a labeled test CSV to compute metrics.")

    uploaded = st.file_uploader("Upload labeled test CSV (must contain 'label' column)", type=["csv", "xlsx"], key="metrics_upload")
    if uploaded is None:
        return
    try:
        if uploaded.name.endswith(".xlsx"):
            test_df = pd.read_excel(uploaded)
        else:
            test_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        return

    if "label" not in test_df.columns:
        st.error("Uploaded CSV must contain a 'label' column with ground-truth labels (Eligible / Not Eligible / High Risk).")
        return

    cols = [
        "monthly_salary",
        "current_emi",
        "requested_loan_amount",
        "loan_tenure_months",
        "credit_score",
        "employment_years",
        "expenses_total",
    ]
    for c in cols:
        if c not in test_df.columns:
            test_df[c] = 0

    X = test_df[cols]
    y_true = test_df["label"].values

    labels, probs, _ = classify_and_regress(X)
    y_pred = labels

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    st.subheader("Computed metrics")
    st.write(f"Accuracy: {acc:.3f}")
    st.write(f"Precision (macro): {prec:.3f}")
    st.write(f"Recall (macro): {rec:.3f}")
    st.write(f"F1-score (macro): {f1:.3f}")

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    if clf_model is not None:
        try:
            if len(np.unique(y_true)) == 2:
                y_bin = (y_true == np.unique(y_true)[1]).astype(int)
                y_score = clf_model.predict_proba(X)[:, 1]
                fpr, tpr, _ = roc_curve(y_bin, y_score)
                roc_auc = auc(fpr, tpr)
                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                ax2.plot([0, 1], [0, 1], linestyle="--")
                ax2.set_xlabel("False Positive Rate")
                ax2.set_ylabel("True Positive Rate")
                ax2.set_title("ROC Curve")
                ax2.legend()
                st.pyplot(fig2)
        except Exception:
            st.info("ROC curve not available for current classifier or labels.")


def page_admin():
    st.title("👨‍💻 Admin / Data Upload")

    st.markdown("Upload a CSV of applicants, preview, run batch predictions & download results.")

    uploaded = st.file_uploader("Upload applicants CSV", type=["csv", "xlsx"], key="admin_upload")
    if uploaded is None:
        st.info("No file uploaded. You can place a CSV at data/applicants.csv to use as default.")
        return

    try:
        if uploaded.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded)
        else:
            df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        return

    st.write(f"Loaded {len(df)} rows")
    st.dataframe(df.head(200))

    cols = [
        "monthly_salary",
        "current_emi",
        "requested_loan_amount",
        "loan_tenure_months",
        "credit_score",
        "employment_years",
        "expenses_total",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = 0

    if st.button("Run batch predictions"):
        labels, probs, max_emi = classify_and_regress(df[cols])
        df["predicted_label"] = labels
        df["predicted_probability"] = probs
        df["recommended_max_emi"] = max_emi

        st.success("Batch predictions added to dataframe.")
        st.dataframe(df.head(200))

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

    if st.button("Save uploaded file to data/applicants.csv"):
        try:
            save_path = DEFAULT_DATA_PATH
            df.to_csv(save_path, index=False)
            st.success(f"Saved to {save_path}")
        except Exception as e:
            st.error(f"Failed to save file: {e}")


# ---------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------
def main():
    st.sidebar.title("EMI Predictor")
    page = st.sidebar.radio("Go to", ["EMI Predictor", "Data Exploration", "Model Metrics", "Admin / Data Upload"])

    if page == "EMI Predictor":
        page_predict()
    elif page == "Data Exploration":
        page_explore()
    elif page == "Model Metrics":
        page_metrics()
    elif page == "Admin / Data Upload":
        page_admin()


if __name__ == "__main__":
    main()
