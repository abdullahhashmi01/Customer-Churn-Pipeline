# End-to-End ML Pipeline — Telco Customer Churn Prediction

A reusable, production-ready machine learning pipeline that predicts whether
a telecom customer will churn, built entirely with scikit-learn's `Pipeline`
and `ColumnTransformer` APIs.

## Objective
Build a pipeline that handles preprocessing, model training, and
hyperparameter tuning as a single exportable object, so it can be reused
in any application (like the included Streamlit demo) without re-writing
preprocessing logic.

## Dataset
`Telco_customer_churn.xlsx` — 7043 customers, 33 raw columns.

Dropped columns (IDs, location data, and data-leakage columns such as
`Churn Label`, `Churn Reason`, `Churn Score`, `CLTV` which are only known
*after* churn happens or are derived from another model).

Target: `Churn Value` (0 = stayed, 1 = churned)

## Project Structure
```
telco-churn-pipeline/
├── data/raw/                   # source dataset
├── src/
│   ├── data_loader.py          # load + clean data
│   ├── preprocessing.py        # ColumnTransformer (scaling + encoding)
│   ├── train.py                # Pipeline + GridSearchCV + joblib export
│   └── predict.py              # load saved pipeline, predict on new data
├── models/churn_pipeline.joblib
├── streamlit_app.py                # Streamlit demo app
├── main.py                     # entry point
└── requirements.txt

```

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train the pipeline (loads data, runs GridSearchCV over Logistic
   Regression and Random Forest, saves the best pipeline):
   ```
   python main.py
   ```

3. Try a prediction from the command line:
   ```
   python -m src.predict
   ```

4. Launch the interactive Streamlit app:
   ```
   streamlit run app/app.py
   ```

## Approach
- **Preprocessing** (`ColumnTransformer`): numeric columns are median-imputed
  and standard-scaled; categorical columns are mode-imputed and one-hot
  encoded. Wrapping this in a `Pipeline` means the exact same
  transformations are automatically applied at prediction time — no manual
  re-encoding needed.
- **Model selection**: `GridSearchCV` (5-fold CV, scoring = F1) compares
  Logistic Regression and Random Forest together using a list of parameter
  grids, so each model is tuned with its own relevant hyperparameters.
- **Export**: the entire best pipeline (preprocessing + trained model) is
  saved as one `.joblib` file — reusable anywhere without needing the
  original training code.

## Results
Best model: Logistic Regression (tuned via GridSearchCV)
- Test set ROC-AUC: ~0.85
- Test set F1 (Churn class): ~0.60

## Skills Demonstrated
- ML pipeline construction with scikit-learn `Pipeline` / `ColumnTransformer`
- Hyperparameter tuning with `GridSearchCV` across multiple model types
- Model export & reusability via `joblib`
- Production-readiness: modular code, reusable preprocessing, deployable
  demo app
