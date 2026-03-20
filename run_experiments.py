#!/usr/bin/env python3
"""
Run KitNET experiments across multiple datasets and model configurations.

Variants (ae_type x paradigm):
  TSR (time-series reconstruction): elm, conv1d, conv2d, transformer, deep_mlp, lstm, stat
  AR  (autoregressive):             conv1d_ar, conv2d_ar, transformer_ar, deep_mlp_ar, lstm_ar
  Other: elm_reg (z-score regularized ELM)

Clustering: corr (default), dbscan, kmeans
Output AE:  elm (default), stat
Detector:   threshold (default), centroid, distribution
Datasets:   knad (default), cic2017, cic2018, aci-iot

Usage:
    python run_experiments.py                                    # All KNAD attacks, default variants
    python run_experiments.py --attacks Mirai_Botnet              # One attack
    python run_experiments.py --variants lstm lstm_ar             # LSTM both paradigms
    python run_experiments.py --clustering dbscan --output-ae stat
    python run_experiments.py --detector centroid
    python run_experiments.py --dataset cic2017 --day Monday-WorkingHours
"""
import argparse
import json
import logging
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.common.config import (ATTACKS, RESULTS_DIR, FM_GRACE_PERIOD, AD_GRACE_PERIOD,
                                N_FEATURES, CIC2017_DAYS, CIC2018_DAYS,
                                DEFAULT_CLUSTERING, DEFAULT_OUTPUT_AE)
from src.common.database import load_attack, load_cic2017, load_cic2018, load_aci_iot
from src.common.detector import threshold_sweep, windowdiff, plot_roc, save_results
from src.common.detectors import CentroidDetector, DistributionDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(RESULTS_DIR, 'experiments.log'), mode='a'),
    ]
)
log = logging.getLogger('run_experiments')

# All supported variants: ae_type + ar flag + regularize
# Paper Sec V-C: 12 models per dataset = 5 DL x 2 paradigms + KitNet(PbP) + STAT(PbP)
VARIANTS = {
    # Packet-by-packet (PbP)
    'elm':            {'ae_type': 'elm',         'regularize': False, 'ar': False},
    'stat':           {'ae_type': 'stat',        'regularize': False, 'ar': False},
    # TSR (Time-Series Reconstruction)
    'conv1d':         {'ae_type': 'conv1d',      'regularize': False, 'ar': False},
    'conv2d':         {'ae_type': 'conv2d',      'regularize': False, 'ar': False},
    'transformer':    {'ae_type': 'transformer', 'regularize': False, 'ar': False},
    'deep_mlp':       {'ae_type': 'deep_mlp',    'regularize': False, 'ar': False},
    'lstm':           {'ae_type': 'lstm',        'regularize': False, 'ar': False},
    # AR (Autoregressive)
    'conv1d_ar':      {'ae_type': 'conv1d',      'regularize': False, 'ar': True},
    'conv2d_ar':      {'ae_type': 'conv2d',      'regularize': False, 'ar': True},
    'transformer_ar': {'ae_type': 'transformer', 'regularize': False, 'ar': True},
    'deep_mlp_ar':    {'ae_type': 'deep_mlp',    'regularize': False, 'ar': True},
    'lstm_ar':        {'ae_type': 'lstm',        'regularize': False, 'ar': True},
}

# Paper Fig. 7 order: MLP(AR,TSR), Conv1D(AR,TSR), Conv2D(AR,TSR), MHA(AR,TSR), LSTM(AR,TSR), PbP(KitNet,STAT)
DEFAULT_VARIANTS = [
    'deep_mlp_ar', 'deep_mlp', 'conv1d_ar', 'conv1d', 'conv2d_ar', 'conv2d',
    'transformer_ar', 'transformer', 'lstm_ar', 'lstm', 'elm', 'stat',
]


def _load_dataset(dataset: str, attack_or_day: str, regularize: bool):
    """Load dataset and labels based on dataset type."""
    if dataset == 'knad':
        return load_attack(attack_or_day, regularize=regularize)
    elif dataset == 'cic2017':
        return load_cic2017(day=attack_or_day, regularize=regularize)
    elif dataset == 'cic2018':
        return load_cic2018(day=attack_or_day, regularize=regularize)
    elif dataset == 'aci-iot':
        return load_aci_iot(regularize=regularize)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def run_single(attack_name: str, variant_name: str, variant_cfg: dict,
               clustering: str = 'corr', output_ae: str = 'elm',
               detector: str = 'threshold', dataset: str = 'knad') -> dict:
    """
    Run a single experiment (one attack/day, one variant).

    Returns:
        metrics dict from threshold_sweep + optional windowdiff.
    """
    ae_type = variant_cfg['ae_type']
    regularize = variant_cfg['regularize']
    ar = variant_cfg.get('ar', False)

    # Output paths
    attack_dir = os.path.join(RESULTS_DIR, dataset, attack_name)
    os.makedirs(attack_dir, exist_ok=True)
    scores_path = os.path.join(attack_dir, f'{variant_name}_{clustering}_{output_ae}_scores.npy')

    # Check for cached scores (resume support)
    if os.path.exists(scores_path):
        log.info(f"[{attack_name}/{variant_name}] Loading cached scores from {scores_path}")
        scores = np.load(scores_path)
    else:
        log.info(f"[{attack_name}/{variant_name}] Loading dataset ({dataset})...")
        t0 = time.time()
        X, y = _load_dataset(dataset, attack_name, regularize)
        n_features = X.shape[1]
        log.info(f"  Loaded {X.shape} in {time.time()-t0:.1f}s")

        log.info(f"[{attack_name}/{variant_name}] Running KitNET "
                 f"(ae={ae_type}, cluster={clustering}, output={output_ae}, ar={ar})...")
        t0 = time.time()
        kn = KitNET(n_features=n_features, ae_type=ae_type,
                     clustering=clustering, output_ae_type=output_ae, ar=ar)
        scores = kn.run(X)
        elapsed = time.time() - t0
        log.info(f"  KitNET complete in {elapsed:.1f}s, {len(scores)} scores")

        np.save(scores_path, scores)
        log.info(f"  Saved scores to {scores_path}")

    # Load labels for evaluation (execution phase only)
    _, y = _load_dataset(dataset, attack_name, False)
    total_train = FM_GRACE_PERIOD + AD_GRACE_PERIOD
    y_exec = y[total_train:total_train + len(scores)]

    if len(y_exec) != len(scores):
        min_len = min(len(y_exec), len(scores))
        log.warning(f"  Score/label length mismatch: {len(scores)} vs {len(y_exec)}, trimming to {min_len}")
        scores = scores[:min_len]
        y_exec = y_exec[:min_len]

    # Apply detector post-processing
    if detector == 'centroid':
        det = CentroidDetector()
        # Use first 10% of scores as training for the detector
        n_train = max(1, len(scores) // 10)
        det.train(scores[:n_train])
        scores = det.execute(scores)
    elif detector == 'distribution':
        det = DistributionDetector()
        n_train = max(1, len(scores) // 10)
        det.train(scores[:n_train])
        scores = det.execute(scores)

    log.info(f"[{attack_name}/{variant_name}] Computing metrics (detector={detector})...")
    metrics = threshold_sweep(scores, y_exec)

    # Compute WindowDiff at the best threshold
    best_th = metrics['best_threshold']
    preds = (np.log(scores + 1e-9) >= best_th).astype(np.int32)
    wd = windowdiff(y_exec, preds)
    metrics['windowdiff'] = float(wd)

    log.info(f"  Best F1={metrics['best_f1']:.4f}, "
             f"Recall={metrics['best_recall']:.4f}, "
             f"FPR={metrics['best_fpr']:.4f}, "
             f"WindowDiff={wd:.4f}, "
             f"Threshold={metrics['best_threshold']:.2f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run KitNET experiments')
    parser.add_argument('--attacks', nargs='+', default=ATTACKS,
                        help='Attack/day names to run')
    parser.add_argument('--variants', nargs='+', default=DEFAULT_VARIANTS,
                        help='Variant names to run')
    parser.add_argument('--clustering', default=DEFAULT_CLUSTERING,
                        choices=['corr', 'dbscan', 'kmeans'],
                        help='Clustering method for feature mapping')
    parser.add_argument('--output-ae', default=DEFAULT_OUTPUT_AE,
                        choices=['elm', 'stat'],
                        help='Output aggregation layer type')
    parser.add_argument('--detector', default='threshold',
                        choices=['threshold', 'centroid', 'distribution'],
                        help='Post-processing detector type')
    parser.add_argument('--dataset', default='knad',
                        choices=['knad', 'cic2017', 'cic2018', 'aci-iot'],
                        help='Dataset to use')
    parser.add_argument('--day', default=None,
                        help='Specific day for CIC-2017/2018 datasets')
    parser.add_argument('--backend', default='torch', choices=['torch', 'tf'],
                        help='Backend framework (torch or tf)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Ignore cached scores and recompute')
    args = parser.parse_args()

    # Import KitNET from the selected backend
    global KitNET
    if args.backend == 'tf':
        os.environ['KITSUNE_BACKEND'] = 'tf'
        from src.tf.kitnet import KitNET
    else:
        from src.torch.kitnet import KitNET

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Resolve attack list for non-KNAD datasets
    if args.dataset == 'cic2017' and args.attacks == ATTACKS:
        args.attacks = [args.day] if args.day else CIC2017_DAYS
    elif args.dataset == 'cic2018' and args.attacks == ATTACKS:
        args.attacks = [args.day] if args.day else CIC2018_DAYS
    elif args.dataset == 'aci-iot' and args.attacks == ATTACKS:
        args.attacks = ['aci-iot']

    all_results = {}
    total_t0 = time.time()

    for attack in args.attacks:
        log.info(f"\n{'='*60}\n{args.dataset.upper()}: {attack}\n{'='*60}")
        attack_results = {}

        for variant_name in args.variants:
            if variant_name not in VARIANTS:
                log.error(f"Unknown variant: {variant_name}")
                continue

            # Remove cached scores if --no-cache
            if args.no_cache:
                attack_dir = os.path.join(RESULTS_DIR, args.dataset, attack)
                cached = os.path.join(attack_dir,
                                      f'{variant_name}_{args.clustering}_{args.output_ae}_scores.npy')
                if os.path.exists(cached):
                    os.remove(cached)

            try:
                metrics = run_single(
                    attack, variant_name, VARIANTS[variant_name],
                    clustering=args.clustering, output_ae=args.output_ae,
                    detector=args.detector, dataset=args.dataset,
                )
                attack_results[variant_name] = metrics
            except Exception as e:
                log.error(f"[{attack}/{variant_name}] FAILED: {e}", exc_info=True)
                attack_results[variant_name] = {'best_f1': 0, 'best_threshold': 0,
                                                  'best_recall': 0, 'best_fpr': 0,
                                                  'windowdiff': 1.0}

        all_results[attack] = attack_results

        # Plot ROC for this attack
        plot_variants = {k: v for k, v in attack_results.items()
                         if 'fpr_values' in v}
        if plot_variants:
            roc_path = os.path.join(RESULTS_DIR, args.dataset, attack, 'roc.png')
            plot_roc(plot_variants, roc_path, attack)
            log.info(f"  ROC saved to {roc_path}")

    # Save global results
    results_path = os.path.join(RESULTS_DIR, f'results_{args.dataset}.json')
    save_results(all_results, results_path)

    total_elapsed = time.time() - total_t0
    log.info(f"\n{'='*60}\nALL DONE in {total_elapsed/60:.1f} minutes\n{'='*60}")
    log.info(f"Results saved to {results_path}")

    # Print summary table
    print(f"\n{'Attack':<30} {'Variant':<18} {'F1':>8} {'Recall':>8} {'FPR':>8} {'WinDiff':>8}")
    print('-' * 85)
    for attack, variants in all_results.items():
        for variant, metrics in variants.items():
            wd = metrics.get('windowdiff', -1)
            print(f"{attack:<30} {variant:<18} "
                  f"{metrics['best_f1']:>8.4f} "
                  f"{metrics['best_recall']:>8.4f} "
                  f"{metrics['best_fpr']:>8.4f} "
                  f"{wd:>8.4f}")


if __name__ == '__main__':
    main()
