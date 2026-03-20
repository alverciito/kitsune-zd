"""
Anomaly detection evaluation: threshold sweep, F1, ROC, and plotting.

Provides threshold-sweeping over anomaly scores to compute precision, recall,
F1, and FPR metrics; a WindowDiff segmentation metric (Section V-B); ROC
plotting; and JSON result serialization.

BUG FIX #1: Every call to threshold_sweep returns and stores fresh results.
The original code failed to capture the return value for elm_reg, reusing
stale variables from the previous loop iteration.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def threshold_sweep(scores: np.ndarray, labels: np.ndarray,
                    n_thresholds: int = 5000) -> dict:
    """Sweep thresholds over anomaly scores and compute detection metrics.

    Applies a log transform to raw anomaly scores (matching the original
    Kitsune evaluation protocol), then evaluates ``n_thresholds`` linearly
    spaced thresholds in [-30, 200]. For each threshold, computes precision,
    recall, F1, and FPR. Returns the threshold that maximizes F1.

    This is the primary evaluation function used in Section V of the paper
    for reporting best-F1 detection results per attack.

    Args:
        scores: 1D array of raw anomaly scores from the KitNET execution
            phase. Shape (N,).
        labels: 1D binary ground-truth array (0=benign, 1=attack). Shape (N,).
        n_thresholds: Number of threshold values to evaluate (int). Higher
            values give finer granularity at the cost of compute time.

    Returns:
        dict: Dictionary containing:
            - 'best_f1' (float): Maximum F1 score across all thresholds.
            - 'best_threshold' (float): Log-space threshold achieving best F1.
            - 'best_recall' (float): Recall (TPR) at best F1 threshold.
            - 'best_fpr' (float): False positive rate at best F1 threshold.
            - 'thresholds' (np.ndarray): All evaluated threshold values.
            - 'f1_values' (np.ndarray): F1 at each threshold.
            - 'recall_values' (np.ndarray): Recall at each threshold.
            - 'fpr_values' (np.ndarray): FPR at each threshold.
            - 'precision_values' (np.ndarray): Precision at each threshold.
    """
    # Apply log transform (matches original code)
    log_scores = np.log(scores + 1e-9)

    # Use range-based thresholds like the original: from -30 to 200
    thresholds = np.linspace(-30, 200, n_thresholds)

    f1_values = np.zeros(n_thresholds)
    recall_values = np.zeros(n_thresholds)
    fpr_values = np.zeros(n_thresholds)
    precision_values = np.zeros(n_thresholds)

    for idx, th in enumerate(thresholds):
        preds = (log_scores >= th).astype(np.int32)

        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        f1_values[idx] = f1
        recall_values[idx] = recall
        fpr_values[idx] = fpr
        precision_values[idx] = precision

    best_idx = np.argmax(f1_values)
    return {
        'best_f1': float(f1_values[best_idx]),
        'best_threshold': float(thresholds[best_idx]),
        'best_recall': float(recall_values[best_idx]),
        'best_fpr': float(fpr_values[best_idx]),
        'thresholds': thresholds,
        'f1_values': f1_values,
        'recall_values': recall_values,
        'fpr_values': fpr_values,
        'precision_values': precision_values,
    }


def windowdiff(ref: np.ndarray, hyp: np.ndarray, k: int = None) -> float:
    """Compute the WindowDiff metric for segmentation evaluation.

    Implements the WindowDiff metric described in Section V-B of the paper.
    Measures the fraction of sliding windows in which the number of
    segment boundaries in the reference differs from the hypothesis. This
    captures both false alarms and missed detections in a boundary-aware
    manner, making it more suitable than point-wise metrics for evaluating
    attack region detection.

    Args:
        ref: Binary reference segmentation array of shape (N,). Values are
            1 for attack/boundary packets and 0 for normal packets.
        hyp: Binary hypothesis segmentation array of shape (N,). Same format
            as ref; typically derived by thresholding anomaly scores.
        k: Half-window size (int). If None, automatically computed as half
            the mean segment length derived from reference boundaries.

    Returns:
        float: WindowDiff score in [0.0, 1.0]. Lower values indicate better
            segmentation agreement. Returns 0.0 if there are no boundaries
            in the reference or if k >= N.
    """
    ref = ref.astype(np.int32)
    hyp = hyp.astype(np.int32)
    n = len(ref)

    if k is None:
        # Compute mean segment length from reference boundaries
        boundaries = np.diff(ref)
        n_boundaries = np.sum(boundaries != 0)
        if n_boundaries == 0:
            return 0.0
        mean_seg_len = n / (n_boundaries + 1)
        k = max(1, int(mean_seg_len / 2))

    if k >= n:
        return 0.0

    # Count boundaries in each window of size k
    ref_counts = np.cumsum(np.abs(np.diff(ref)))
    hyp_counts = np.cumsum(np.abs(np.diff(hyp)))

    # Pad with 0 at start for cumsum indexing
    ref_counts = np.concatenate([[0], ref_counts])
    hyp_counts = np.concatenate([[0], hyp_counts])

    mismatches = 0
    n_windows = n - k
    for i in range(n_windows):
        ref_b = ref_counts[i + k] - ref_counts[i]
        hyp_b = hyp_counts[i + k] - hyp_counts[i]
        if ref_b != hyp_b:
            mismatches += 1

    return mismatches / n_windows if n_windows > 0 else 0.0


def plot_roc(results_by_variant: dict, save_path: str, attack_name: str):
    """Plot ROC curves for all KitNET variants on a single figure.

    Creates a Receiver Operating Characteristic (ROC) plot with FPR on the
    x-axis and TPR (Recall) on the y-axis. Each variant is plotted in a
    distinct color with its best F1 score shown in the legend. The best
    operating point for each variant is marked with a scatter dot.

    Supported variant colors: 'elm' (blue), 'elm_reg' (green),
    'conv1d' (orange), 'transformer' (red). Unknown variants use gray.

    Args:
        results_by_variant: Dictionary mapping variant name (str) to a
            metrics dictionary as returned by threshold_sweep(). Must contain
            keys 'fpr_values', 'recall_values', 'best_f1', 'best_fpr',
            and 'best_recall'.
        save_path: Filesystem path (str) where the PNG image will be saved.
            Parent directories are created automatically.
        attack_name: Attack name (str) used in the plot title. Underscores
            are replaced with spaces for display.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {'elm': 'blue', 'elm_reg': 'green', 'conv1d': 'orange', 'transformer': 'red'}

    for variant, metrics in results_by_variant.items():
        fpr = metrics['fpr_values']
        recall = metrics['recall_values']
        best_f1 = metrics['best_f1']
        best_fpr = metrics['best_fpr']
        best_recall = metrics['best_recall']
        color = colors.get(variant, 'gray')

        ax.plot(fpr, recall, label=f'{variant} (F1={best_f1:.3f})',
                color=color, linewidth=1.2)
        ax.scatter([best_fpr], [best_recall], color=color, s=40, zorder=5)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR (Recall)')
    ax.set_title(f'ROC - {attack_name.replace("_", " ")}')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_results(results: dict, path: str):
    """Save experiment results as a JSON file, stripping numpy arrays.

    Serializes only the scalar metrics (best_f1, best_threshold,
    best_recall, best_fpr, and optionally windowdiff) for each
    attack/variant combination. Numpy arrays (thresholds, f1_values, etc.)
    are excluded since they are not JSON-serializable and are only needed
    for plotting.

    Args:
        results: Nested dictionary of shape {attack_name: {variant_name:
            metrics_dict}}. Each metrics_dict should contain at least
            'best_f1', 'best_threshold', 'best_recall', 'best_fpr'.
        path: Filesystem path (str) for the output JSON file. Parent
            directories are created automatically.
    """
    serializable = {}
    for attack, variants in results.items():
        serializable[attack] = {}
        for variant, metrics in variants.items():
            serializable[attack][variant] = {
                'best_f1': metrics['best_f1'],
                'best_threshold': metrics['best_threshold'],
                'best_recall': metrics['best_recall'],
                'best_fpr': metrics['best_fpr'],
            }
            if 'windowdiff' in metrics:
                serializable[attack][variant]['windowdiff'] = metrics['windowdiff']
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)
