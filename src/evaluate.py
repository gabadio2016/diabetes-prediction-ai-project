import os
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

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
# FILE: evaluate.py
# PROJECT: Diabetes Prediction (FAU - Intro to AI)
# DESCRIPTION:
#   Evaluation script for the diabetes prediction models.
#   This script:
#     • Loads the trained XGBoost and Neural Network models
#     • Re-creates the exact train / val / test split
#     • Rebuilds or loads the preprocessing pipeline
#     • Evaluates models on the (held-out) test set
#     • Prints metrics and saves:
#           - Confusion matrix PNGs
#           - ROC curve PNGs
# =============================================================================


TARGET_COL = "diabetes"


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Compute standard binary classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (0/1).
    y_prob : np.ndarray
        Predicted probabilities for the positive class.

    Returns
    -------
    metrics : dict
        Dictionary of metric_name -> value.
    """
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    """
    Plot and save a confusion matrix image.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[0, 1],
    )
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    save_path: str,
) -> None:
    """
    Plot and save a ROC curve image.
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        # This happens if only one class is present in y_true
        fpr, tpr, auc = [0.0, 1.0], [0.0, 1.0], float("nan")

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main() -> None:
    """
    Main evaluation entrypoint.

    Steps:
      1) Load data and recreate train/val/test split
      2) Load preprocessor and trained models
      3) Transform test features
      4) Evaluate and save plots
    """
    seed_everything(42)
    ensure_directories()

    print(">>> Loading raw data...")
    df = load_raw_data(DATA_PATH)

    print(">>> Recreating train / val / test splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        df, target_col=TARGET_COL
    )

    # Load preprocessor and models
    preprocessor_path = os.path.join(RESULTS_DIR, "preprocessor.pkl")
    xgb_path = os.path.join(RESULTS_DIR, "xgboost.pkl")
    nn_path = os.path.join(RESULTS_DIR, "neural_network.pkl")

    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(
            f"Missing preprocessor.pkl at {preprocessor_path}. "
            "Run train.py first to generate models."
        )

    print(">>> Loading preprocessor and models from disk...")
    preprocessor = joblib.load(preprocessor_path)

    if not os.path.exists(xgb_path) or not os.path.exists(nn_path):
        raise FileNotFoundError(
            "Missing trained model files in /results. "
            "Ensure train.py has been run successfully."
        )

    xgb_model = joblib.load(xgb_path)
    nn_model = joblib.load(nn_path)

    print(">>> Transforming test features...")
    X_test_proc = preprocessor.transform(X_test)

    # -------------------------------------------------------------------------
    # Evaluate XGBoost
    # -------------------------------------------------------------------------
    print("\n>>> Evaluating XGBoost on test set...")
    xgb_prob = xgb_model.predict_proba(X_test_proc)[:, 1]
    xgb_metrics = compute_metrics(y_test.values, xgb_prob)
    print("XGBoost Test Metrics:")
    for k, v in xgb_metrics.items():
        print(f"  {k:9s}: {v:.4f}")

    xgb_cm_path = os.path.join(RESULTS_DIR, "xgboost_confusion_matrix.png")
    xgb_roc_path = os.path.join(RESULTS_DIR, "xgboost_roc_curve.png")
    plot_confusion_matrix(y_test.values, xgb_prob, "XGBoost Confusion Matrix", xgb_cm_path)
    plot_roc_curve(y_test.values, xgb_prob, "XGBoost ROC Curve", xgb_roc_path)
    print(f"  Saved XGBoost confusion matrix -> {xgb_cm_path}")
    print(f"  Saved XGBoost ROC curve        -> {xgb_roc_path}")

    # -------------------------------------------------------------------------
    # Evaluate Neural Network
    # -------------------------------------------------------------------------
    print("\n>>> Evaluating Neural Network on test set...")
    nn_prob = nn_model.predict_proba(X_test_proc)[:, 1]
    nn_metrics = compute_metrics(y_test.values, nn_prob)
    print("Neural Network Test Metrics:")
    for k, v in nn_metrics.items():
        print(f"  {k:9s}: {v:.4f}")

    nn_cm_path = os.path.join(RESULTS_DIR, "neural_net_confusion_matrix.png")
    nn_roc_path = os.path.join(RESULTS_DIR, "neural_net_roc_curve.png")
    plot_confusion_matrix(y_test.values, nn_prob, "Neural Net Confusion Matrix", nn_cm_path)
    plot_roc_curve(y_test.values, nn_prob, "Neural Net ROC Curve", nn_roc_path)
    print(f"  Saved NN confusion matrix      -> {nn_cm_path}")
    print(f"  Saved NN ROC curve             -> {nn_roc_path}")

    # Summary as a small DataFrame for copy/paste into report
    print("\n>>> Summary of Test Metrics (XGBoost vs Neural Net)")
    summary_df = pd.DataFrame(
        {
            "metric": list(xgb_metrics.keys()),
            "xgboost": list(xgb_metrics.values()),
            "neural_net": [nn_metrics[m] for m in xgb_metrics.keys()],
        }
    )
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n>>> Evaluation complete.")


if __name__ == "__main__":
    main()
