# EMI-Predict-AI — Intelligent Financial Risk Assessment Platform

A Streamlit app that predicts EMI eligibility and recommended maximum EMI for loan applicants. The app includes interactive pages for single predictions, data exploration, model metrics, and admin batch uploads.

This repository contains a ready-to-run Streamlit application (app.py) with heuristic fallbacks so the app works even when trained model artifacts are not present. For best results, add trained classifier/regressor artifacts to the `models/` folder.

---

## Contents

- `app.py` — Streamlit application (4 pages)
- `requirements.txt` — (recommended) Python dependencies
- `models/` — (optional) put trained model files here:
  - `models/emi_classifier.pkl` — classifier (predict eligibility / probability)
  - `models/emi_regressor.pkl` — regressor (predict recommended max EMI)
  - `models/metrics.json` — optional saved training/test metrics
- `data/applicants.csv` — optional default dataset for exploration
- `README.md` — this file

---

## App overview (pages)

1. EMI Predictor (default)
   - Inputs:
     - Monthly salary
     - Current EMI amount
     - Requested loan amount
     - Loan tenure (months)
     - Credit score
     - Employment years
     - Monthly expenses (rent, travel, school, other)
   - Button: "Predict Eligibility"
   - Outputs:
     - Eligibility result (Eligible / Not Eligible / High Risk)
     - Recommended Maximum EMI (from regressor or heuristic)
     - Probability score (if classifier supports it)
     - Credit-strength visualization (gauge using Plotly, or matplotlib fallback)

2. Data Exploration
   - Upload or use `data/applicants.csv`
   - Charts:
     - Average salary vs. eligibility (if `eligibility` column available)
     - Credit score distribution
     - Correlation heatmap (numeric fields)

3. Model Metrics
   - Upload labeled test CSV (`label` column expected) to compute Accuracy, Precision, Recall, F1
   - Confusion matrix and ROC curve (if applicable)
   - Optionally loads pre-saved `models/metrics.json`

4. Admin / Data Upload
   - Upload CSV/XLSX of applicants
   - Run batch predictions and download results as Excel/CSV
   - Option to save uploaded file to `data/applicants.csv` on the server (note: this writes to the running server filesystem, not to the GitHub repo)

---

## Quick start (local)

1. Create and activate a virtual environment (recommended)
   - Linux/macOS:
     ```
     python -m venv .venv
     source .venv/bin/activate
     ```
   - Windows:
     ```
     python -m venv .venv
     .venv\Scripts\activate
     ```

2. Install dependencies (recommended to use the repo's `requirements.txt`)
   ```
   pip install -r requirements.txt
   ```
   If you don't have `requirements.txt`, at minimum install:
   ```
   pip install streamlit pandas numpy scikit-learn plotly seaborn matplotlib openpyxl joblib
   ```

3. Run the app
   ```
   streamlit run app.py
   ```

---

## Deploying to Streamlit Cloud (or other hosts)

- Add `requirements.txt` at the repo root (Streamlit Cloud will install dependencies).
- If deploying on Streamlit Cloud and you see a ModuleNotFoundError (e.g., `plotly`), add `plotly` to `requirements.txt` (or add via the Cloud UI packages setting).
- Streamlit Cloud uses the repository's default branch. Ensure the files you want are in that branch or configure the app to point to the correct branch.

---

## Models & Data

- To use ML models instead of fallbacks, save trained models using `joblib.dump()`:
  - Classifier: `models/emi_classifier.pkl`
  - Regressor: `models/emi_regressor.pkl`

- Models are expected to accept the same numeric features used in the UI:
  - `monthly_salary`, `current_emi`, `requested_loan_amount`, `loan_tenure_months`, `credit_score`, `employment_years`, `expenses_total`

- Example: save models after training:
  ```python
  import joblib
  joblib.dump(trained_classifier, "models/emi_classifier.pkl")
  joblib.dump(trained_regressor, "models/emi_regressor.pkl")
  ```

- Data format for batch uploads:
  - CSV or XLSX with columns for the features above. Column names should match expected names for automatic inference; missing columns are filled with zeros.

---

## Troubleshooting

- ModuleNotFoundError for a plotting library
  - Add missing package (e.g., plotly) to `requirements.txt` and redeploy or install locally:
    ```
    pip install plotly
    ```
  - The app includes a matplotlib fallback so it will still run without Plotly for the credit gauge.

- Models not found
  - The app uses a heuristic fallback if `models/` files are missing. For accurate predictions, provide trained model artifacts.

- Saving uploaded `data/applicants.csv`
  - The "Save uploaded file" button saves to the running server file system (useful on a persistent VM). This does not commit to GitHub. To persist in GitHub, upload via the GitHub UI or commit locally and push.

---

## Adding a requirements.txt (recommended)

Create `requirements.txt` at repo root (example):
```
streamlit>=1.20
pandas>=1.3
numpy>=1.21
scikit-learn>=1.0
plotly>=5.0
seaborn>=0.11
matplotlib>=3.4
openpyxl>=3.0
joblib>=1.1
```

---

## Security & Privacy notes

- Do not upload sensitive PII or real customer data to public repositories or public deploys.
- If you plan to deploy publicly, sanitize and test sample datasets. Consider access controls for the Admin page.

---

## Contributing

- Feel free to open issues or PRs to:
  - Improve UI/UX
  - Add authentication for Admin operations
  - Integrate MLflow or automated model fetching
  - Add unit tests or CI for notebooks

---

## License & Contact

- Add your chosen license (e.g., MIT) in `LICENSE` if desired.
- Contact: @Abhinab-123 (GitHub)

---

Thank you — if you'd like, I can:
- Create this README in a new branch and open a PR (tell me branch name & commit message), or
- Add `requirements.txt` and the safe `app.py` fallback I prepared earlier and push them for you.
