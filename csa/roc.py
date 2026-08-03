"""Detector evaluation: ROC/AUC, calibration, correlation (no sklearn dep)."""

from __future__ import annotations

import numpy as np


def roc_curve(labels: np.ndarray, scores: np.ndarray):
    """Returns (fpr, tpr, auc). labels: {0,1}; higher score = predicts 1."""
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    ok = np.isfinite(scores) & np.isfinite(labels)
    labels, scores = labels[ok], scores[ok]
    P = labels.sum()
    N = len(labels) - P
    if P == 0 or N == 0:
        return np.array([0, 1]), np.array([0, 1]), float("nan")
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(labels[order])
    fp = np.cumsum(1 - labels[order])
    # collapse ties on score
    distinct = np.r_[np.diff(scores[order]) != 0, True]
    tpr = np.r_[0.0, tp[distinct] / P]
    fpr = np.r_[0.0, fp[distinct] / N]
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def auc_score(labels, scores) -> float:
    return roc_curve(labels, scores)[2]


def calibration_bins(labels, scores, n_bins: int = 10):
    """Decile-bin the score; return (bin_centers, empirical_rate, counts)."""
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    ok = np.isfinite(scores)
    labels, scores = labels[ok], scores[ok]
    qs = np.quantile(scores, np.linspace(0, 1, n_bins + 1))
    qs[-1] += 1e-9
    centers, rates, counts = [], [], []
    for i in range(n_bins):
        m = (scores >= qs[i]) & (scores < qs[i + 1])
        if m.sum() > 0:
            centers.append(float(scores[m].mean()))
            rates.append(float(labels[m].mean()))
            counts.append(int(m.sum()))
    return np.array(centers), np.array(rates), np.array(counts)


def spearman(x, y) -> float:
    from scipy.stats import spearmanr
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    r, _ = spearmanr(x[ok], y[ok])
    return float(r)
