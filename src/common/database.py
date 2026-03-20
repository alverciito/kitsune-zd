"""Load KITSUNE and other IDS dataset CSV files."""
import os
import glob
import numpy as np
import pandas as pd
from .config import (DATA_ROOT, ATTACK_DIRS, EPSILON,
                     CIC2017_ROOT, CIC2017_DAYS,
                     CIC2018_ROOT, CIC2018_DAYS,
                     ACI_IOT_ROOT)


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


def _clean_dataframe(df: pd.DataFrame) -> np.ndarray:
    """Clean a CIC-style dataframe: drop non-numeric, handle NaN/Inf."""
    # Keep only numeric columns
    numeric = df.select_dtypes(include=[np.number])
    X = numeric.to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)
    return X


def load_cic2017(day: str = None, regularize: bool = False):
    """
    Load CIC-IDS-2017 dataset.

    Args:
        day: One of CIC2017_DAYS, or None to load all days concatenated.
        regularize: If True, z-score normalize features.

    Returns:
        (X, y) where y=1 for attack rows (Label != 'BENIGN').
    """
    if day is not None:
        pattern = os.path.join(CIC2017_ROOT, f"*{day}*.csv")
        files = sorted(glob.glob(pattern))
    else:
        files = sorted(glob.glob(os.path.join(CIC2017_ROOT, "*.csv")))

    if not files:
        raise FileNotFoundError(f"No CIC-2017 files found for day={day} in {CIC2017_ROOT}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip()
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)

    # Extract labels: 'Label' column, BENIGN=0, else=1
    label_col = 'Label' if 'Label' in combined.columns else combined.columns[-1]
    y = (combined[label_col].str.strip() != 'BENIGN').astype(np.int32).to_numpy()

    # Drop label column and keep numeric features
    features = combined.drop(columns=[label_col], errors='ignore')
    X = _clean_dataframe(features)

    if regularize:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X = (X - mean) / (std + EPSILON)

    return X, y


def load_cic2018(day: str = None, regularize: bool = False):
    """
    Load CIC-IDS-2018 dataset.

    Args:
        day: One of CIC2018_DAYS, or None to load all days.
        regularize: If True, z-score normalize features.

    Returns:
        (X, y) where y=1 for attack rows (Label != 'Benign').
    """
    if day is not None:
        pattern = os.path.join(CIC2018_ROOT, f"*{day}*.csv")
        files = sorted(glob.glob(pattern))
    else:
        files = sorted(glob.glob(os.path.join(CIC2018_ROOT, "*.csv")))

    if not files:
        raise FileNotFoundError(f"No CIC-2018 files found for day={day} in {CIC2018_ROOT}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip()
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)

    label_col = 'Label' if 'Label' in combined.columns else combined.columns[-1]
    y = (combined[label_col].str.strip() != 'Benign').astype(np.int32).to_numpy()

    features = combined.drop(columns=[label_col], errors='ignore')
    X = _clean_dataframe(features)

    if regularize:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X = (X - mean) / (std + EPSILON)

    return X, y


def load_aci_iot(regularize: bool = False):
    """
    Load ACI-IoT-2023 dataset.

    Returns:
        (X, y) where y=1 for attack rows (Label != 'Benign').
    """
    files = sorted(glob.glob(os.path.join(ACI_IOT_ROOT, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No ACI-IoT files found in {ACI_IOT_ROOT}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = df.columns.str.strip()
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)

    label_col = 'Label' if 'Label' in combined.columns else combined.columns[-1]
    y = (combined[label_col].str.strip() != 'Benign').astype(np.int32).to_numpy()

    features = combined.drop(columns=[label_col], errors='ignore')
    X = _clean_dataframe(features)

    if regularize:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X = (X - mean) / (std + EPSILON)

    return X, y
