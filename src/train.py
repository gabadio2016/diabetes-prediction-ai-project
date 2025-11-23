"""
train.py
Trains machine learning models (Logistic Regression and Random Forest)
using the preprocessing pipeline.
"""

import joblib
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from preprocess import preprocess_pipeline


def train_models(csv_path: str):
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(csv_path)

    models = {
        "xgboost": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, eval_metric='logloss', n_jobs=-1, random_state=0),
        "neural_network": MLPClassifier(max_iter=100, hidden_layer_sizes=(32, 16), activation="relu", alpha=0.0001, random_state=0),
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        joblib.dump(model, f"results/{name}.pkl")

    joblib.dump(scaler, "results/scaler.pkl")

    print("Models successfully trained and saved.")
    return trained_models


if __name__ == "__main__":
    train_models("diabetes_prediction_dataset.csv")