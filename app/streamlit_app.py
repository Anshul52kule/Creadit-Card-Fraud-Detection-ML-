import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fraud Shield", page_icon=":shield:", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load('models/fraud_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    columns = joblib.load('models/feature_columns.pkl')
    return model, scaler, columns

model, scaler, feature_columns = load_artifacts()

def predict_fraud(row_dict):
    row = row_dict.copy()
    scaled = scaler.transform([[row.pop('Amount'), row.pop('Time')]])
    row['Amount'] = scaled[0][0]
    row['Time'] = scaled[0][1]
    X_new = pd.DataFrame([row])[feature_columns]
    return model.predict_proba(X_new)[0][1]

# ---------- Sidebar ----------
with st.sidebar:
    st.title("Fraud Shield")
    st.caption("XGBoost-powered fraud detection system")
    st.divider()
    st.metric("PR-AUC", "0.808")
    st.metric("ROC-AUC", "1.000")

# ---------- Header ----------
st.title("Transaction Risk Analyzer")
st.caption("Enter transaction details to check fraud probability in real time")
st.write("")

tab1, tab2 = st.tabs(["Manual Check", "Random Sample"])

with tab1:
    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Transaction Details")
        c1, c2 = st.columns(2)
        time_val = c1.number_input("Time (seconds)", value=50000.0)
        amount_val = c2.number_input("Amount ($)", value=100.0, min_value=0.0)

        with st.expander("Advanced: PCA Features (V1–V28)"):
            v_values = {}
            cols = st.columns(4)
            for i in range(1, 29):
                with cols[(i - 1) % 4]:
                    v_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.1, key=f"v{i}")

        st.write("")
        analyze = st.button("Analyze Transaction", use_container_width=True, type="primary")

    with right:
        st.subheader("Result")
        if analyze:
            row = {"Time": time_val, "Amount": amount_val, **v_values}
            proba = predict_fraud(row)

            if proba >= 0.5:
                st.error(f"**FRAUD DETECTED**\n\nRisk Score: {proba*100:.1f}%")
            else:
                st.success(f"**LEGITIMATE TRANSACTION**\n\nRisk Score: {proba*100:.1f}%")

            st.progress(float(min(proba, 1.0)))
        else:
            st.info("Results will appear here after analysis")

with tab2:
    df = pd.read_csv('data/creditcard.csv')
    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Sample a real transaction")
        pick = st.button("Pick Random Transaction", use_container_width=True, type="primary")

    with right:
        st.subheader("Result")
        if pick:
            sample = df.sample(1).iloc[0]
            actual = "FRAUD" if sample['Class'] == 1 else "NORMAL"
            row = sample.drop('Class').to_dict()
            proba = predict_fraud(row)

            st.caption(f"Actual label from dataset: **{actual}**")

            if proba >= 0.5:
                st.error(f"**FRAUD DETECTED**\n\nRisk Score: {proba*100:.1f}%")
            else:
                st.success(f"**LEGITIMATE TRANSACTION**\n\nRisk Score: {proba*100:.1f}%")

            st.progress(float(min(proba, 1.0)))