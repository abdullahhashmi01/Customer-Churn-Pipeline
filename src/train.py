import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Sahi imports jo aapki baki files se match karte hain
from src.data_loader import load_data, get_features_and_target
from src.data_preprocessing import build_preprocessor

# .joblib use karein taaki app.py isko load kar sakay
MODEL_PATH = "models/churn_pipeline.joblib"
REPORT_PATH = "reports/model_metrics.txt"


def train():  # Function ka naam 'train' rakha taaki main.py ko mil sakay
    # 1. Data Load karein
    df = load_data()
    
    # 2. Features aur Target alag karein (Sahi function name)
    X, y = get_features_and_target(df)

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4. Preprocessor call karein (jo src.data_preprocessing se aa raha hai)
    preprocessor = build_preprocessor()

    # 5. Base Pipeline banayein
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])

    # 6. Hyperparameter Tuning Grid (Dono models ke liye)
    param_grid = [
        {
            "model": [LogisticRegression(max_iter=1000, random_state=42)],
            "model__C": [0.1, 1, 10]
        },
        {
            "model": [RandomForestClassifier(random_state=42)],
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10, None]
        }
    ]

    # 7. GridSearchCV Setup
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    print("Running GridSearchCV over Logistic Regression + Random Forest...")
    grid_search.fit(X_train, y_train)

    # 8. Best Model se Predict karein
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Metrics Calculate karein
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    # Folders banayein agar nahi bane toh
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # Model save karein (.joblib format mein)
    joblib.dump(best_model, MODEL_PATH)

    # Report file likhein
    with open(REPORT_PATH, "w") as f:
        f.write("Telco Customer Churn Model Report\n")
        f.write("=" * 40)
        f.write("\n\nBest Parameters:\n")
        f.write(str(grid_search.best_params_))
        f.write("\n\nAccuracy:\n")
        f.write(str(round(accuracy, 4)))
        f.write("\n\nF1 Score:\n")
        f.write(str(round(f1, 4)))
        f.write("\n\nClassification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(matrix))

    # Terminal par print karein
    print("\n=== Training Completed Successfully ===")
    print("Best Parameters:", grid_search.best_params_)
    print("Accuracy:", round(accuracy, 4))
    print("F1 Score:", round(f1, 4))
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", matrix)
    print(f"\nModel saved at: {MODEL_PATH}")
    print(f"Report saved at: {REPORT_PATH}")


if __name__ == "__main__":
    train()