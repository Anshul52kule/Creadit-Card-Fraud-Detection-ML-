# Credit Card Fraud Detection — Machine Learning

An end-to-end machine learning project to detect fraudulent credit card transactions using classification models, with an interactive Streamlit demo app.

## 📌 Overview

Credit card fraud detection is a highly imbalanced classification problem — fraudulent transactions make up a very small fraction of all transactions. This project builds and compares multiple ML models to identify fraud while handling this class imbalance.

## 📂 Dataset

- **Source:** [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) (Kaggle)
- **Rows:** ~284,807 transactions
- **Target:** `Class` (0 = Normal, 1 = Fraud)
- **Features:** 28 anonymized PCA components (V1–V28), plus `Time` and `Amount`

> Note: The dataset is not included in this repo due to size. Download it from Kaggle and place it in the `data/` folder.

## 🧹 Data Preprocessing

- Scaled `Amount` and `Time` using `StandardScaler`
- Applied **SMOTE** (Synthetic Minority Oversampling) on the training set only, to address severe class imbalance
- Train-test split performed before any resampling, to avoid data leakage

## 🤖 Models Trained & Compared

- Logistic Regression
- Random Forest
- XGBoost (best performing — PR-AUC ~0.808)

## 📊 Evaluation

- Confusion matrices generated for each model
- ROC curves and Precision-Recall curves compared across models
- Given the class imbalance, **PR-AUC** was prioritized over accuracy as the key evaluation metric
- Feature importance analyzed for the best model

## 📈 Reports

All visualizations are saved in the `reports/` folder:
- Class distribution
- Correlation heatmap
- Amount by class
- Confusion matrices (per model)
- ROC & PR curves
- Feature importance

## 💾 Artifacts Saved

| File | Description |
|---|---|
| `models/fraud_model.pkl` | Trained best-performing model |
| `models/scaler.pkl` | Fitted StandardScaler for preprocessing |
| `models/feature_columns.pkl` | Feature column order expected by the model |

## 🖥️ Streamlit Demo App

An interactive app (`app/streamlit_app.py`) lets you input transaction details and get a real-time fraud prediction.

### Run locally:
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## 🛠️ Tech Stack

- Python, Pandas, NumPy
- Scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- Streamlit
- Matplotlib / Seaborn

## 🚀 How to Run

1. Clone the repo and install dependencies:
```bash
   pip install -r requirements.txt
```
2. Download the dataset from Kaggle and place it in `data/creditcard.csv`
3. Run the notebook (`notebook/fraud_detection.ipynb`) to reproduce the analysis and retrain models
4. Launch the Streamlit app to test predictions interactively

## 🔮 Future Improvements

- Try deep learning approaches (autoencoders for anomaly detection)
- Add real-time streaming prediction pipeline
- Deploy the Streamlit app to the cloud (Streamlit Community Cloud / Render)