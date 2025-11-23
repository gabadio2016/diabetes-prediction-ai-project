"""
evaluate.py
Loads saved models and evaluates them using accuracy, recall,
precision, F1-score, and ROC-AUC.
"""

import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from preprocess import preprocess_pipeline


def evaluate_model(csv_path: str, model_path: str):
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(csv_path)

    model = joblib.load(model_path)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    return metrics


if __name__ == "__main__":
    model = "results/xgboost.pkl"
    results = evaluate_model("data\diabetes.csv", model)
    print(results)