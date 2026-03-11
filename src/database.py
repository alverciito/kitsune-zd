"""Load KITSUNE dataset CSV files."""
import os
import numpy as np
import pandas as pd
from .config import DATA_ROOT, ATTACK_DIRS, EPSILON


def load_attack(attack_name: str, regularize: bool = False):
    """
    Load dataset and labels for a given attack from the KITSUNE dataset.

    Args:
        attack_name: One of the attack names from config.ATTACKS.
        regularize: If True, apply z-score normalization to features.

    Returns:
        (X, y) where X is (N, 115) float32 and y is (N,) int {0, 1}.
    """
    subdir = ATTACK_DIRS[attack_name]
    base = os.path.join(DATA_ROOT, subdir)
    dataset_path = os.path.join(base, f"{attack_name}_dataset.csv")
    labels_path = os.path.join(base, f"{attack_name}_labels.csv")

    # Load features
    x_df = pd.read_csv(dataset_path)
    X = x_df.to_numpy(dtype=np.float32)

    # Load labels - format: "","x" header, then "row_idx",label
    y_df = pd.read_csv(labels_path)
    y = y_df.iloc[:, -1].to_numpy(dtype=np.int32)

    # Ensure same length
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]

    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)

    if regularize:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X = (X - mean) / (std + EPSILON)

    return X, y
