"""
Step 5: Streamlit App
Simple UI for entering customer details and getting a live churn prediction
from the saved pipeline (models/churn_pipeline.joblib).

Run with:  streamlit run app/app.py
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from src.predict import load_pipeline, predict_churn

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📊")

st.title("📊 Telco Customer Churn Predictor")
st.write("Enter customer details below to predict whether they are likely to churn.")

@st.cache_resource
def get_pipeline():
    return load_pipeline("models/churn_pipeline.joblib")

pipeline = get_pipeline()

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", 0, 100, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col2:
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 500.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0)

if st.button("Predict Churn", type="primary"):
    customer = {
        "Gender": gender,
        "Senior Citizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges,
    }

    result = predict_churn(pipeline, customer)

    if result["prediction"] == "Churn":
        st.error(f"⚠️ Prediction: **{result['prediction']}** (probability: {result['churn_probability']:.1%})")
    else:
        st.success(f"✅ Prediction: **{result['prediction']}** (probability: {result['churn_probability']:.1%})")

    st.progress(result["churn_probability"])
