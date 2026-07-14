# Credit Card Fraud Detection — End-to-End ML Project

An end-to-end machine learning pipeline that detects fraudulent credit card
transactions using real-world imbalanced data. Built from raw data to a
deployed interactive web app.

## Overview

- **Dataset:** Real Kaggle "Credit Card Fraud Detection" dataset (ULB) —
  283,726 transactions after deduplication, with only 473 fraud cases (0.17%).
- **Problem type:** Binary classification under severe class imbalance.
- **Goal:** Maximize fraud detection (recall) while keeping false alarms
  (precision) at a usable level.

## Project Structure
Credit Card Fraud Detection/
├── data/
│   └── creditcard.csv
├── models/
│   ├── fraud_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
├── notebook/
│   └── fraud_detection.ipynb
├── reports/
│   ├── class_distribution.png
│   ├── amount_by_class.png
│   ├── correlation_heatmap.png
│   ├── confusion_matrix_LogisticRegression.png
│   ├── confusion_matrix_RandomForest.png
│   ├── confusion_matrix_XGBoost.png
│   ├── roc_curves.png
│   ├── pr_curves.png
│   └── feature_importance.png
├── app/
│   └── streamlit_app.py
├── .streamlit/
│   └── config.toml
└── README.md

## Pipeline

1. **Data cleaning** — removed 1,081 duplicate rows.
2. **Preprocessing** — `StandardScaler` applied to `Time` and `Amount`
   (other features are already PCA-transformed and on a comparable scale).
3. **Train-test split** — 80/20, stratified to preserve the fraud ratio in
   both sets.
4. **Class imbalance handling** — SMOTE applied **only on the training set**
   to avoid data leakage into evaluation.
5. **Models trained and compared:**
   - Logistic Regression (baseline)
   - Random Forest
   - XGBoost
6. **Model selection metric:** PR-AUC (Precision-Recall AUC), not accuracy.
   With 99.8% of transactions being normal, accuracy is meaningless — a model
   that always predicts "not fraud" would score ~99.8% accuracy while being
   useless.

## Results

| Model                | PR-AUC |
|-----------------------|--------|
| Logistic Regression   | 0.694  |
| Random Forest         | 0.801  |
| **XGBoost (selected)**| **0.808** |

XGBoost was selected as the final model based on PR-AUC.

**Top predictive features (by XGBoost importance):** V14, V4, V10, V12, V8 —
largely consistent with the correlation analysis in EDA, though tree-based
importance also captures non-linear relationships that simple correlation misses.

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib xgboost streamlit plotly

# Open and run notebook/fraud_detection.ipynb to reproduce training

# Launch the interactive demo app
python -m streamlit run app/streamlit_app.py
```

## Demo App

A Streamlit web app (`app/streamlit_app.py`) provides:
- **Manual Check** — enter transaction details (Time, Amount, V1–V28) and get
  a real-time fraud probability.
- **Random Sample** — pull a real transaction from the dataset and compare the
  model's prediction against the actual label.

## Key Learnings

- **Accuracy is misleading under class imbalance** — precision, recall, F1,
  and PR-AUC are the metrics that matter here.
- **SMOTE trade-off** — boosts recall (catches more fraud) but can hurt
  precision (more false alarms) if applied carelessly or paired with a model
  too simple to handle the synthetic data well.
- **Tree-based models outperform Logistic Regression** on this problem due to
  non-linear feature interactions.
- **Scaling matters** — Logistic Regression failed to converge until `Time`
  and `Amount` were standardized.

## Possible Extensions

- Use the real deployment threshold tuning based on business cost of false
  negatives vs. false positives.
- Add SHAP values for per-transaction explainability.
- Serve the model via a REST API (FastAPI/Flask) for production use.