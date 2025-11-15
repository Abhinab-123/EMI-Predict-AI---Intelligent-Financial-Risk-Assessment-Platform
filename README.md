```markdown
# EMI-Predict-AI — EMI Predictor (single-page)

A compact Streamlit app to predict EMI eligibility and a recommended maximum EMI for loan applicants.
This single-page variant supports:
- Single applicant prediction via a form
- Batch predictions from a CSV (uploaded or `data/applicants.csv`)
- Use of saved sklearn/joblib models from `models/` (falls back to a heuristic if none found)

Quick start
1. (Optional) create and activate a venv:
   - python -m venv .venv && source .venv/bin/activate
2. Install dependencies:
   - pip install -r requirements.txt
   - If you don't have requirements.txt, install at minimum:
     pip install streamlit pandas numpy scikit-learn plotly seaborn matplotlib openpyxl joblib
3. Run:
   - streamlit run app.py

Models
- Place trained model artifacts (joblib) in `models/`.
  Recommended names:
  - Classifier: `models/best_classifier.pkl` or `models/emi_classifier.pkl`
  - Regressor: `models/best_regressor.pkl` or `models/emi_regressor.pkl`
- Models should accept these numeric features (column names):
  `monthly_salary, current_emi, requested_loan_amount, loan_tenure_months, credit_score, employment_years, expenses_total`
- If preprocessing (scaling/encoding) is required, save a Pipeline that includes preprocessing.

Batch CSV
- Upload a CSV/XLSX with the expected columns (missing columns will be filled with zeros).
- Or check the option to use `data/applicants.csv` if present.
- After running batch predictions you can download results as CSV/Excel.

Troubleshooting
- ModuleNotFoundError for plotly: add `plotly` to requirements.txt or install it manually.
- Models not working: ensure they were saved with joblib and accept the expected feature columns.
- Admin/save to `data/applicants.csv` writes to the running server filesystem only — it does not commit to GitHub.

Security note
- Avoid uploading sensitive/PII data to public repos or public deployments.

Contact
- Repository owner: @Abhinab-123
```
