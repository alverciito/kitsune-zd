"""
Loader functions for IDS (Intrusion Detection System) datasets.

Supports the following datasets used in the paper's evaluation:
- KITSUNE (primary): 9 attacks with AfterImage features (115 dims).
- CIC-IDS-2017: Flow-based dataset with 78 features.
- CIC-IDS-2018: Updated version of CIC-2017 with additional attack types.
- ACI-IoT-2023: IoT-specific intrusion detection dataset.

All loaders return (X, y) tuples where X is a float32 feature matrix and
y is an int32 binary label vector (0=benign, 1=attack). Optional z-score
normalization is available via the ``regularize`` parameter.
"""
import os
import glob
import numpy as np
import pandas as pd
from .config import (DATA_ROOT, ATTACK_DIRS, EPSILON,
                     CIC2017_ROOT, CIC2017_DAYS,
                     CIC2018_ROOT, CIC2018_DAYS,
                     ACI_IOT_ROOT)


def load_attack(attack_name: str, regularize: bool = False):
    """Load dataset and labels for a given attack from the KITSUNE dataset.

    Reads the AfterImage feature CSV and corresponding label CSV from the
    KITSUNE data directory. NaN and Inf values in features are replaced with
    0.0 and +/-1e9 respectively. The feature and label arrays are truncated
    to the shorter length to handle any length mismatches between files.

    Args:
        attack_name: One of the canonical attack names from config.ATTACKS
            (e.g., 'SSDP_Flood', 'Active_Wiretap'). Must be a key in
            config.ATTACK_DIRS.
        regularize: If True, apply z-score normalization (subtract mean,
            divide by std) to the feature matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: A pair (X, y) where:
            - X: Feature matrix of shape (N, 115) with dtype float32.
            - y: Binary label vector of shape (N,) with dtype int32,
              where 0=benign and 1=attack.

    Raises:
        KeyError: If attack_name is not in ATTACK_DIRS.
        FileNotFoundError: If the dataset or label CSV files are missing.
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
    """Clean a CIC-style dataframe by keeping only numeric columns and sanitizing values.

    Selects only numeric-typed columns from the DataFrame, converts to
    float32, and replaces NaN with 0.0, +Inf with 1e9, and -Inf with -1e9.

    Args:
        df: Input pandas DataFrame, potentially containing string/object
            columns (e.g., IP addresses, timestamps).

    Returns:
        np.ndarray: Cleaned float32 array of shape (N, D) where D is the
            number of numeric columns.
    """
    # Keep only numeric columns
    numeric = df.select_dtypes(include=[np.number])
    X = numeric.to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)
    return X


def load_cic2017(day: str = None, regularize: bool = False):
    """Load the CIC-IDS-2017 flow-based intrusion detection dataset.

    Reads one or more CSV files from the CIC-IDS-2017 directory. Column
    names are stripped of whitespace. The 'Label' column is binarized:
    'BENIGN' -> 0, all other labels -> 1. Non-numeric feature columns
    are dropped automatically.

    Args:
        day: One of config.CIC2017_DAYS (e.g., 'Tuesday-WorkingHours'),
            or None to load and concatenate all days.
        regularize: If True, apply z-score normalization (subtract mean,
            divide by std) to the feature matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: A pair (X, y) where:
            - X: Feature matrix of shape (N, D) with dtype float32.
            - y: Binary label vector of shape (N,) with dtype int32.

    Raises:
        FileNotFoundError: If no CSV files match the given day pattern.
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
    """Load the CIC-IDS-2018 flow-based intrusion detection dataset.

    Reads one or more CSV files from the CIC-IDS-2018 directory. Similar to
    CIC-2017 but with additional attack categories and a different benign
    label format ('Benign' with capital B only, vs. 'BENIGN').

    Args:
        day: One of config.CIC2018_DAYS (e.g., '02-14-2018'), or None to
            load and concatenate all days.
        regularize: If True, apply z-score normalization (subtract mean,
            divide by std) to the feature matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: A pair (X, y) where:
            - X: Feature matrix of shape (N, D) with dtype float32.
            - y: Binary label vector of shape (N,) with dtype int32.

    Raises:
        FileNotFoundError: If no CSV files match the given day pattern.
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
    """Load the ACI-IoT-2023 IoT intrusion detection dataset.

    Reads all CSV files from the ACI-IoT-2023 directory and concatenates
    them. The 'Label' column is binarized: 'Benign' -> 0, all other
    labels -> 1. Non-numeric feature columns are dropped automatically.

    Args:
        regularize: If True, apply z-score normalization (subtract mean,
            divide by std) to the feature matrix.

    Returns:
        tuple[np.ndarray, np.ndarray]: A pair (X, y) where:
            - X: Feature matrix of shape (N, D) with dtype float32.
            - y: Binary label vector of shape (N,) with dtype int32.

    Raises:
        FileNotFoundError: If no CSV files are found in ACI_IOT_ROOT.
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
