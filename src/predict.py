"""
Step 4: Prediction / Reusability
Loads the saved pipeline (preprocessing + model, all in one object)
and uses it to predict on new raw customer data -- no need to
re-scale or re-encode anything manually, the pipeline does it all.
"""
import joblib
import pandas as pd

PIPELINE_PATH = "models/churn_pipeline.joblib"


def load_pipeline(path: str = PIPELINE_PATH):
    return joblib.load(path)


def predict_churn(pipeline, customer: dict):
    """customer: dict with the same raw columns used in training (before target)."""
    input_df = pd.DataFrame([customer])
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]
    return {
        "prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(float(probability), 4),
    }


if __name__ == "__main__":
    pipeline = load_pipeline()

    sample_customer = {
        "Gender": "Female",
        "Senior Citizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "Tenure Months": 2,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "No",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "No",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 85.5,
        "Total Charges": 171.0,
    }

    result = predict_churn(pipeline, sample_customer)
    print(result)
