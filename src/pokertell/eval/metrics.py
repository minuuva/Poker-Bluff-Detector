"""Ablation metrics: the honest numbers this project exists to report.

Context for interpreting results: human lie detection averages 54% accuracy
(Bond and DePaulo 2006, 25k+ judgments); the best audio-visual deception
models reach about 67% in-domain and 54-62% cross-domain. A behavioral delta
of +0.02 to +0.05 AUC over the betting baseline, with a confidence interval
excluding zero, is a literature-consistent positive result. Anything much
larger means leakage until proven otherwise.
"""

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def single_report(y_true: np.ndarray, p: np.ndarray) -> dict:
    """Metrics for one model's predictions. AUC is NaN if y is single-class."""
    y_true = np.asarray(y_true)
    single = len(np.unique(y_true)) < 2
    return {
        "auc": float("nan") if single else float(roc_auc_score(y_true, p)),
        "logloss": float(log_loss(y_true, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p)),
        "n": int(len(y_true)),
        "base_rate": float(np.mean(y_true)),
    }


def ablation_report(
    y_true: np.ndarray,
    p_base: np.ndarray,
    p_full: np.ndarray,
) -> dict:
    """Compare baseline and behavior-augmented predictions on the same test set."""
    base = single_report(y_true, p_base)
    full = single_report(y_true, p_full)
    return {
        "auc_base": base["auc"],
        "auc_full": full["auc"],
        "delta_auc": full["auc"] - base["auc"],
        "logloss_base": base["logloss"],
        "logloss_full": full["logloss"],
        "brier_base": base["brier"],
        "brier_full": full["brier"],
        "n": base["n"],
        "base_rate": base["base_rate"],
    }


def bootstrap_delta_auc(
    y_true: np.ndarray,
    p_base: np.ndarray,
    p_full: np.ndarray,
    groups: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 7,
) -> dict:
    """Bootstrap CI for the AUC delta.

    If groups is given (hand ids), resampling happens at the group level so
    correlated decisions within a hand do not narrow the interval artificially.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    deltas = []

    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        index_by_group = {g: np.nonzero(groups == g)[0] for g in unique_groups}

    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, len(y_true), len(y_true))
        else:
            sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            idx = np.concatenate([index_by_group[g] for g in sampled])
        y = y_true[idx]
        if len(np.unique(y)) < 2:
            continue
        deltas.append(roc_auc_score(y, p_full[idx]) - roc_auc_score(y, p_base[idx]))

    deltas = np.array(deltas)
    return {
        "delta_auc_mean": float(np.mean(deltas)),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "n_boot_effective": int(len(deltas)),
    }


def calibration_table(
    y_true: np.ndarray,
    p_pred: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """Reliability table: predicted probability vs observed frequency per bin."""
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges, edges[1:]):
        mask = (p_pred >= lo) & (p_pred < hi if hi < 1.0 else p_pred <= hi)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin_low": float(lo),
                "bin_high": float(hi),
                "mean_predicted": float(np.mean(p_pred[mask])),
                "observed_rate": float(np.mean(y_true[mask])),
                "count": int(mask.sum()),
            }
        )
    return rows
