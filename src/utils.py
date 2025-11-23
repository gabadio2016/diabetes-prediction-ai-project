"""
utils.py
Helper functions for reproducibility, loading models, and printing metrics.
"""

import joblib
from tabulate import tabulate


def load_model(path: str):
    """Loads a saved model from a .pkl file."""
    return joblib.load(path)


def print_metrics(metrics: dict):
    """Formats evaluation metrics into a clean table."""
    table = [(k, v) for k, v in metrics.items()]
    print(tabulate(table, headers=["Metric", "Value"], tablefmt="pretty"))