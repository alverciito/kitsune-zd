#!/usr/bin/env python3
"""
Run all KITSUNE experiments: 9 attacks x 4 variants.

Variants:
  - elm:         Original Kitsune autoencoder (numpy MLP, online SGD)
  - elm_reg:     Same as elm, but with z-score regularized input
  - conv1d:      Conv1D windowed autoencoder (PyTorch)
  - transformer: Multi-Head Attention windowed autoencoder (PyTorch)

Usage:
    python run_experiments.py                          # Run all
    python run_experiments.py --attacks Mirai_Botnet    # Run one attack
    python run_experiments.py --variants elm conv1d     # Run specific variants
"""
import argparse
import json
import logging
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.config import ATTACKS, RESULTS_DIR, FM_GRACE_PERIOD, AD_GRACE_PERIOD, N_FEATURES
from src.database import load_attack
from src.kitnet import KitNET
from src.detector import threshold_sweep, plot_roc, save_results

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(RESULTS_DIR, 'experiments.log'), mode='a'),
    ]
)
log = logging.getLogger('run_experiments')

VARIANTS = {
    'elm':         {'ae_type': 'elm',         'regularize': False},
    'elm_reg':     {'ae_type': 'elm',         'regularize': True},
    'conv1d':      {'ae_type': 'conv1d',      'regularize': False},
    'transformer': {'ae_type': 'transformer', 'regularize': False},
    'conv2d':      {'ae_type': 'conv2d',      'regularize': False},
    'deep_mlp':    {'ae_type': 'deep_mlp',    'regularize': False},
}


def run_single(attack_name: str, variant_name: str, variant_cfg: dict) -> dict:
    """
    Run a single experiment (one attack, one variant).

    Returns:
        metrics dict from threshold_sweep.
    """
    ae_type = variant_cfg['ae_type']
    regularize = variant_cfg['regularize']

    # Output paths
    attack_dir = os.path.join(RESULTS_DIR, attack_name)
    os.makedirs(attack_dir, exist_ok=True)
    scores_path = os.path.join(attack_dir, f'{variant_name}_scores.npy')

    # Check for cached scores (resume support)
    if os.path.exists(scores_path):
        log.info(f"[{attack_name}/{variant_name}] Loading cached scores from {scores_path}")
        scores = np.load(scores_path)
    else:
        log.info(f"[{attack_name}/{variant_name}] Loading dataset...")
        t0 = time.time()
        X, y = load_attack(attack_name, regularize=regularize)
        log.info(f"  Loaded {X.shape} in {time.time()-t0:.1f}s")

        log.info(f"[{attack_name}/{variant_name}] Running KitNET (ae_type={ae_type})...")
        t0 = time.time()
        kn = KitNET(n_features=N_FEATURES, ae_type=ae_type)
        scores = kn.run(X)
        elapsed = time.time() - t0
        log.info(f"  KitNET complete in {elapsed:.1f}s, {len(scores)} scores")

        np.save(scores_path, scores)
        log.info(f"  Saved scores to {scores_path}")

    # Load labels for evaluation
    # We only evaluate on the execution phase
    _, y = load_attack(attack_name, regularize=False)  # Labels don't depend on regularization
    total_train = FM_GRACE_PERIOD + AD_GRACE_PERIOD
    y_exec = y[total_train:total_train + len(scores)]

    if len(y_exec) != len(scores):
        min_len = min(len(y_exec), len(scores))
        log.warning(f"  Score/label length mismatch: {len(scores)} vs {len(y_exec)}, trimming to {min_len}")
        scores = scores[:min_len]
        y_exec = y_exec[:min_len]

    log.info(f"[{attack_name}/{variant_name}] Computing metrics...")
    metrics = threshold_sweep(scores, y_exec)
    log.info(f"  Best F1={metrics['best_f1']:.4f}, "
             f"Recall={metrics['best_recall']:.4f}, "
             f"FPR={metrics['best_fpr']:.4f}, "
             f"Threshold={metrics['best_threshold']:.2f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run KITSUNE experiments')
    parser.add_argument('--attacks', nargs='+', default=ATTACKS,
                        help='Attack names to run')
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS.keys()),
                        help='Variant names to run')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}
    total_t0 = time.time()

    for attack in args.attacks:
        log.info(f"\n{'='*60}\nATTACK: {attack}\n{'='*60}")
        attack_results = {}

        for variant_name in args.variants:
            if variant_name not in VARIANTS:
                log.error(f"Unknown variant: {variant_name}")
                continue
            try:
                metrics = run_single(attack, variant_name, VARIANTS[variant_name])
                attack_results[variant_name] = metrics
            except Exception as e:
                log.error(f"[{attack}/{variant_name}] FAILED: {e}", exc_info=True)
                attack_results[variant_name] = {'best_f1': 0, 'best_threshold': 0,
                                                  'best_recall': 0, 'best_fpr': 0}

        all_results[attack] = attack_results

        # Plot ROC for this attack
        plot_variants = {k: v for k, v in attack_results.items()
                         if 'fpr_values' in v}
        if plot_variants:
            roc_path = os.path.join(RESULTS_DIR, attack, 'roc.png')
            plot_roc(plot_variants, roc_path, attack)
            log.info(f"  ROC saved to {roc_path}")

    # Save global results
    results_path = os.path.join(RESULTS_DIR, 'results.json')
    save_results(all_results, results_path)

    total_elapsed = time.time() - total_t0
    log.info(f"\n{'='*60}\nALL DONE in {total_elapsed/60:.1f} minutes\n{'='*60}")
    log.info(f"Results saved to {results_path}")

    # Print summary table
    print(f"\n{'Attack':<25} {'Variant':<15} {'F1':>8} {'Recall':>8} {'FPR':>8}")
    print('-' * 70)
    for attack, variants in all_results.items():
        for variant, metrics in variants.items():
            print(f"{attack:<25} {variant:<15} "
                  f"{metrics['best_f1']:>8.4f} "
                  f"{metrics['best_recall']:>8.4f} "
                  f"{metrics['best_fpr']:>8.4f}")


if __name__ == '__main__':
    main()
