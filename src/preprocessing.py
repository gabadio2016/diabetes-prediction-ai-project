import os
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =============================================================================
# FILE: preprocessing.py
# PROJECT: Diabetes Prediction (FAU - Intro to AI)
# DESCRIPTION:
#   This module handles all data loading and preprocessing, including:
#     • Reading the raw CSV dataset
#     • Handling missing values
#     • Splitting into train / validation / test (70 / 15 / 15)
#     • Building a sklearn ColumnTransformer (numeric + categorical)
#     • Returning ready-to-train NumPy arrays
# =============================================================================


RANDOM_SEED: int = 42


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """
    Load the raw diabetes dataset from a CSV file.

    Parameters
    ----------
    csv_path : str
        Relative or absolute path to the CSV file.

    Returns
    -------
    df : pd.DataFrame
        Loaded dataset.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find dataset at: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def train_val_test_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split the dataset into train, validation, and test sets using a 70/15/15 split.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including features and target.
    target_col : str
        Name of the target column.
    test_size : float, optional
        Proportion of the data to reserve for the test set.
    val_size : float, optional
        Proportion of the data to reserve for the validation set.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First, split off the test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Now split the remaining into train and validation
    val_relative_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_size,
        stratify=y_train_val,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(
    X: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build a ColumnTransformer that:
      • Imputes missing values (mean for numeric, 'Unknown' for categorical)
      • Scales numeric features
      • One-hot encodes categorical features

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    preprocessor : ColumnTransformer
        Fitted scikit-learn ColumnTransformer pipeline.
    numeric_features : list of str
        Names of numeric feature columns.
    categorical_features : list of str
        Names of categorical feature columns.
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    # Manual numeric imputation (fill with mean)
    X[numeric_features] = X[numeric_features].fillna(X[numeric_features].mean())

    # Manual categorical imputation (fill with "Unknown")
    for col in categorical_features:
        X[col] = X[col].fillna("Unknown")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def fit_transform_preprocessor(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
    """
    Fit the ColumnTransformer on the training data and transform train/val/test.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Raw feature splits.

    Returns
    -------
    X_train_proc, X_val_proc, X_test_proc : np.ndarray
        Transformed feature arrays ready for modeling.
    preprocessor : ColumnTransformer
        Fitted preprocessor that can be saved and reused at inference time.
    """
    # Build the preprocessor based on training columns (copy to avoid in-place edits)
    preprocessor, _, _ = build_preprocessor(X_train.copy())

    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    return X_train_proc, X_val_proc, X_test_proc, preprocessor
