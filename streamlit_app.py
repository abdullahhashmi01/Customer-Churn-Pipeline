import sys
import os
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predict import load_pipeline, predict_churn

# Page Setup
st.set_page_config(
    page_title="Telco Customer Intelligence Dashboard",
    page_icon="🔮",
    layout="wide",
)

# App Title & Header
st.title("🔮 Telco Customer Retention & Churn Intelligence")
st.markdown(
    "This AI-powered platform predicts customer churn risk and provides actionable "
    "retention strategies using our trained machine learning pipeline."
)
st.write("---")


# Load Saved Pipeline Securely
@st.cache_resource
def get_pipeline():
    return load_pipeline("models/churn_pipeline.joblib")


try:
    pipeline = get_pipeline()
except Exception as e:
    st.error(
        "⚠️ Could not load the model pipeline. Please ensure you have run `python main.py` first to train and export the model."
    )
    st.stop()

# Create Tabs for Sidebar/Main view separation of features
tab1, tab2 = st.tabs(["🎯 Live Churn Predictor", "📊 Model Performance & Insights"])

with tab1:
    st.subheader("👤 Customer Profile Input")
    st.markdown("Enter the customer's demographics, services, and contract details:")

    # Form structure for clean execution
    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.info("📋 Demographics & Core")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner?", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
            tenure = st.number_input(
                "Tenure (Months)", min_value=0, max_value=100, value=12
            )

        with col2:
            st.info("⚡ Services Subscribed")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines", ["No", "Yes", "No phone service"]
            )
            internet_service = st.selectbox(
                "Internet Service Provider", ["Fiber optic", "DSL", "No"]
            )
            online_security = st.selectbox(
                "Online Security", ["No", "Yes", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online Backup", ["Yes", "No", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device Protection", ["No", "Yes", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech Support", ["No", "Yes", "No internet service"]
            )

        with col3:
            st.info("💰 Billing & Engagement")
            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies", ["No", "Yes", "No internet service"]
            )
            contract = st.selectbox(
                "Contract Type", ["Month-to-month", "One year", "Two year"]
            )
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly_charges = st.number_input(
                "Monthly Charges ($)", min_value=0.0, max_value=300.0, value=70.0
            )
            total_charges = st.number_input(
                "Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0
            )

        # Submit button inside form
        submit_btn = st.form_submit_button(
            "Analyze Churn Risk", type="primary", use_container_width=True
        )

    if submit_btn:
        # Build raw feature dictionary exactly as data loader maps it
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

        # Run model engine
        result = predict_churn(pipeline, customer)
        prob = result["churn_probability"]

        st.write("---")
        st.subheader("📊 Assessment Results")

        # Metric Visual Display Columns
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            if result["prediction"] == "Churn":
                st.metric(
                    label="Risk Assessment", value="⚠️ High Risk (Churn)", delta=f"{prob:.1%} Prob"
                )
            else:
                st.metric(
                    label="Risk Assessment", value="✅ Low Risk (Retain)", delta=f"{1-prob:.1%} Safe"
                )

        with res_col2:
            st.write("**Churn Probability Distribution Bar:**")
            # Create a simple, elegant visual bar chart for probability breakdown
            prob_df = pd.DataFrame(
                {"Status": ["Loyalty Probability", "Churn Probability"], "Percentage": [1 - prob, prob]}
            )
            st.bar_chart(data=prob_df, x="Status", y="Percentage", color="#ff4b4b" if prob > 0.5 else "#00cc66")

        # Business Prescriptive Strategy Layer
        st.markdown("### 🛡️ Recommended Retention Actions")
        if prob > 0.5:
            st.error(
                f"**Critical Status:** Customer has a {prob:.1%} chance of leaving the service provider."
            )
            # Custom recommendation matrix based on inputs
            recs = []
            if contract == "Month-to-month":
                recs.append(
                    "🔹 **Contract Migration:** Offer a high-incentive upgrade to a 1-Year or 2-Year contract with an exclusive discount."
                )
            if tech_support == "No" and internet_service != "No":
                recs.append(
                    "🔹 **Value Add-on:** Provide a free 3-month trial of Tech Support to stabilize user experience."
                )
            if payment_method == "Electronic check":
                recs.append(
                    "🔹 **Payment Auto-Enrollment:** Pitch Auto-pay setup via Credit Card/Bank Transfer with a minor billing credit reward."
                )
            if not recs:
                recs.append(
                    "🔹 **Direct Outreach:** Initiate proactive support call via account management team to gauge satisfaction metrics."
                )

            for r in recs:
                st.write(r)
        else:
            st.success(
                f"**Healthy Status:** Customer is highly likely to stay ({1-prob:.1%} retention certainty)."
            )
            st.write(
                "🔹 **Surprise & Delight:** Enroll user into current loyalty benefit tiers or cross-sell premium add-ons cleanly."
            )

with tab2:
    st.subheader("📈 Core Production Model Metrics")
    st.write(
        "Performance documentation extracted from the tuned optimal pipeline configuration:"
    )

    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric(label="Best Model Architecture", value="Logistic Regression")
    with m_col2:
        st.metric(label="Area Under ROC (ROC-AUC)", value="0.85")
    with m_col3:
        st.metric(label="Target F1-Score (Churn Class)", value="0.60")

    st.write("---")
    st.markdown("### 🛠️ Structural Pipeline Highlights")
    st.markdown(
        """
    * **Automated Engineering:** Implements strict data flow isolation using scikit-learn's `ColumnTransformer`.
    * **Leakage Avoidance:** Drops business-derived markers post-facto (`Churn Reason`, `CLTV`, `Churn Score`) explicitly at loading phase.
    * **Deterministic Deployment:** Packages the structural transformer arrays and weight models inside a singular executable `.joblib` entity.
    """
    )