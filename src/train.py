import os
from typing import Tuple

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from preprocessing import (
    load_raw_data,
    train_val_test_split,
    fit_transform_preprocessor,
)
from utils import (
    RESULTS_DIR,
    DATA_PATH,
    ensure_directories,
    seed_everything,
)


# =============================================================================
# FILE: train.py
# PROJECT: Diabetes Prediction (FAU - Intro to AI)
# DESCRIPTION:
#   End-to-end training script for the diabetes prediction models.
#   This script:
#     • Loads the raw Kaggle dataset
#     • Splits data into train / val / test
#     • Builds and fits the preprocessing pipeline
#     • Trains two models:
#           1) XGBoost classifier
#           2) Neural Network (sklearn MLPClassifier)
#     • Evaluates both models on the validation set
#     • Saves the preprocessor and both trained models under /results
# =============================================================================


TARGET_COL = "diabetes"  # Adjust if your target column name is different


def build_xgboost_model() -> XGBClassifier:
    """
    Create an XGBoost classifier with reasonable default hyperparameters.
    """
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    return model


def build_neural_net(input_dim: int) -> MLPClassifier:
    """
    Create a simple fully-connected neural network using scikit-learn's MLPClassifier.
    """
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=200,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )
    return model


def evaluate_on_validation(
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """
    Compute standard classification metrics on the validation set.

    Returns
    -------
    accuracy, precision, recall, f1, roc_auc
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_val, y_prob)
    except ValueError:
        roc_auc = float("nan")

    return accuracy, precision, recall, f1, roc_auc


def main() -> None:
    """
    Main training entrypoint.

    This function orchestrates:
      • Data loading
      • Splits
      • Preprocessing
      • Model training
      • Validation evaluation
      • Artifact saving
    """
    seed_everything(42)
    ensure_directories()

    print(">>> Loading raw data...")
    df = load_raw_data(DATA_PATH)

    print(">>> Creating train / val / test splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        df, target_col=TARGET_COL
    )

    print(">>> Fitting preprocessing pipeline...")
    X_train_proc, X_val_proc, X_test_proc, preprocessor = fit_transform_preprocessor(
        X_train, X_val, X_test
    )

    # -------------------------------------------------------------------------
    # Train XGBoost
    # -------------------------------------------------------------------------
    print("\n>>> Training XGBoost model...")
    xgb_model = build_xgboost_model()
    xgb_model.fit(X_train_proc, y_train)

    xgb_metrics = evaluate_on_validation(xgb_model, X_val_proc, y_val)
    print(
        "XGBoost Validation Metrics:\n"
        f"  Accuracy : {xgb_metrics[0]:.4f}\n"
        f"  Precision: {xgb_metrics[1]:.4f}\n"
        f"  Recall   : {xgb_metrics[2]:.4f}\n"
        f"  F1       : {xgb_metrics[3]:.4f}\n"
        f"  ROC AUC  : {xgb_metrics[4]:.4f}\n"
    )

    # -------------------------------------------------------------------------
    # Train Neural Network
    # -------------------------------------------------------------------------
    print(">>> Training Neural Network (MLPClassifier)...")
    input_dim = X_train_proc.shape[1]
    nn_model = build_neural_net(input_dim=input_dim)
    nn_model.fit(X_train_proc, y_train)

    nn_metrics = evaluate_on_validation(nn_model, X_val_proc, y_val)
    print(
        "Neural Network Validation Metrics:\n"
        f"  Accuracy : {nn_metrics[0]:.4f}\n"
        f"  Precision: {nn_metrics[1]:.4f}\n"
        f"  Recall   : {nn_metrics[2]:.4f}\n"
        f"  F1       : {nn_metrics[3]:.4f}\n"
        f"  ROC AUC  : {nn_metrics[4]:.4f}\n"
    )

    # -------------------------------------------------------------------------
    # Save artifacts
    # -------------------------------------------------------------------------
    print(">>> Saving preprocessor and models to disk...")
    preprocessor_path = os.path.join(RESULTS_DIR, "preprocessor.pkl")
    xgb_path = os.path.join(RESULTS_DIR, "xgboost.pkl")
    nn_path = os.path.join(RESULTS_DIR, "neural_network.pkl")

    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(xgb_model, xgb_path)
    joblib.dump(nn_model, nn_path)

    print(f"  Saved preprocessor -> {preprocessor_path}")
    print(f"  Saved XGBoost      -> {xgb_path}")
    print(f"  Saved Neural Net   -> {nn_path}")

    print("\n>>> Training complete.")


if __name__ == "__main__":
    main()
