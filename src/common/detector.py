"""
Anomaly detection metrics: threshold sweep, F1, ROC, and plotting.

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
    """
    Sweep thresholds over anomaly scores and compute detection metrics.

    Uses log-space scores and percentile-based thresholds for better coverage.

    Args:
        scores: Anomaly scores from KitNET execution phase.
        labels: Binary ground truth (0=benign, 1=attack).
        n_thresholds: Number of threshold values to try.

    Returns:
        Dict with keys: best_f1, best_threshold, best_recall, best_fpr,
        and arrays: thresholds, f1_values, recall_values, fpr_values.
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
    """
    WindowDiff metric for segmentation evaluation (paper Sec V-B).

    Measures the fraction of windows where the number of boundaries in the
    reference and hypothesis differ.

    Args:
        ref: Binary reference segmentation (1=boundary/attack, 0=normal).
        hyp: Binary hypothesis segmentation (same shape as ref).
        k: Half the mean segment size. If None, computed from ref.

    Returns:
        WindowDiff score in [0, 1]. Lower is better.
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
    """
    Plot ROC curves for all variants of a given attack on the same figure.

    Args:
        results_by_variant: Dict of {variant_name: metrics_dict}.
        save_path: Path to save PNG.
        attack_name: For the plot title.
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
    """Save results dict as JSON (strip numpy arrays for serialization)."""
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
