"""Ablation metrics sanity checks on synthetic data."""

import numpy as np

from pokertell.eval.metrics import ablation_report, bootstrap_delta_auc, calibration_table


def _synthetic(n=400, seed=0):
    """Baseline sees a weak signal; the full model sees a slightly stronger one."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p_base = np.clip(0.5 + 0.1 * (y - 0.5) + rng.normal(0, 0.15, n), 0.01, 0.99)
    p_full = np.clip(0.5 + 0.25 * (y - 0.5) + rng.normal(0, 0.15, n), 0.01, 0.99)
    return y, p_base, p_full


def test_ablation_report_detects_improvement():
    y, p_base, p_full = _synthetic()
    report = ablation_report(y, p_base, p_full)
    assert report["auc_full"] > report["auc_base"]
    assert report["delta_auc"] > 0
    assert report["logloss_full"] < report["logloss_base"]


def test_bootstrap_ci_excludes_zero_for_clear_improvement():
    y, p_base, p_full = _synthetic(n=800)
    ci = bootstrap_delta_auc(y, p_base, p_full, n_boot=300, seed=1)
    assert ci["ci_low"] > 0


def test_bootstrap_grouped_resampling_runs():
    y, p_base, p_full = _synthetic(n=200)
    groups = np.repeat(np.arange(50), 4)
    ci = bootstrap_delta_auc(y, p_base, p_full, groups=groups, n_boot=100, seed=1)
    assert ci["n_boot_effective"] > 0


def test_calibration_table_covers_all_points():
    y, p_base, _ = _synthetic()
    rows = calibration_table(y, p_base, n_bins=10)
    assert sum(r["count"] for r in rows) == len(y)
