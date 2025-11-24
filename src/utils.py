import os
import random
from typing import Any

import numpy as np


# =============================================================================
# FILE: utils.py
# PROJECT: Diabetes Prediction (FAU - Intro to AI)
# DESCRIPTION:
#   Utility helpers shared across the training and evaluation scripts, including:
#     • Global path constants (data, results)
#     • Directory creation helpers
#     • Reproducibility utilities (random seeds)
# =============================================================================


# Project-relative paths (assumes scripts are run from the repo root)
DATA_PATH: str = os.path.join("data", "diabetes_prediction_dataset.csv")
RESULTS_DIR: str = os.path.join("results")


def ensure_directories() -> None:
    """
    Create any required output directories if they do not already exist.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)


def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and (optionally) other libraries
    to promote reproducible experiments.

    Parameters
    ----------
    seed : int, optional
        Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch is optional; this is just a bonus if installed.
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # Ignore if PyTorch is not installed.
        pass
