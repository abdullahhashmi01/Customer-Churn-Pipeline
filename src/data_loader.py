"""
Step 1: Data Loading
Loads the Telco churn dataset and drops columns that are irrelevant
or cause data leakage (i.e. columns that "give away" the answer).
"""
import pandas as pd

# Columns not useful for prediction (IDs, location, or leakage)
DROP_COLS = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude",
    "Churn Label",     # duplicate of target (leakage)
    "Churn Reason",    # only filled AFTER churn happens (leakage)
    "Churn Score",     # pre-computed risk score (leakage)
    "CLTV",            # derived business metric, not a raw feature
]

TARGET_COL = "Churn Value"


def load_data(path: str = "data/Telco_customer_churn.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path)

    # "Total Charges" has blank strings for brand-new customers (tenure=0)
    # Convert to numeric; blanks become NaN so the pipeline's imputer can handle them
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    df = df.drop(columns=DROP_COLS)
    return df


def get_features_and_target(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


if __name__ == "__main__":
    df = load_data()
    print("Shape after cleaning:", df.shape)
    print(df.dtypes)
