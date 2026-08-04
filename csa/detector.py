"""Combined label-free detector (Ablation 1) + calibration and transfer.

A single signal is a heuristic; the proposal's Ablation 1 asks whether the
combination buys anything over dropped-mass alone ("is the rest complexity for
its own sake?"). This module fits a small logistic model over the label-free
signals, with:

  - standardization fitted on train only (no leakage),
  - L2 regularization (signals are correlated by construction),
  - grouped cross-validation by request, because steps within a request are
    not independent — the naive i.i.d. split inflates AUC badly.

No sklearn dependency: the fit is a few lines of scipy.optimize on the
regularized logistic loss.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .roc import auc_score


class LogisticDetector:
    """L2-regularized logistic regression with standardization."""

    def __init__(self, l2: float = 1.0):
        self.l2 = l2
        self.w = None
        self.mu = None
        self.sd = None

    def _prep(self, X, fit=False):
        X = np.asarray(X, dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if fit:
            self.mu = X.mean(0)
            self.sd = X.std(0)
            self.sd[self.sd < 1e-9] = 1.0
        Z = (X - self.mu) / self.sd
        return np.c_[Z, np.ones(len(Z))]

    def fit(self, X, y):
        A = self._prep(X, fit=True)
        y = np.asarray(y, dtype=float)

        def loss(w):
            z = A @ w
            # stable log(1 + exp(z))
            ll = np.logaddexp(0.0, z) - y * z
            reg = self.l2 * np.sum(w[:-1] ** 2) / 2.0
            return ll.sum() + reg

        def grad(w):
            p = 1.0 / (1.0 + np.exp(-np.clip(A @ w, -50, 50)))
            g = A.T @ (p - y)
            g[:-1] += self.l2 * w[:-1]
            return g

        w0 = np.zeros(A.shape[1])
        res = minimize(loss, w0, jac=grad, method="L-BFGS-B")
        self.w = res.x
        return self

    def score(self, X):
        A = self._prep(X, fit=False)
        return 1.0 / (1.0 + np.exp(-np.clip(A @ self.w, -50, 50)))


def grouped_cv_auc(X, y, groups, n_folds: int = 5, l2: float = 1.0, seed: int = 0):
    """Cross-validated AUC with folds split by GROUP (request), not by row.

    Steps inside one request share a prompt, a budget and a KV state; splitting
    i.i.d. leaks the request across train/test and inflates AUC. Returns
    (mean_auc, std_auc, oof_scores).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, n_folds)

    aucs, oof = [], np.full(len(y), np.nan)
    for f in folds:
        te = np.isin(groups, f)
        tr = ~te
        if tr.sum() < 10 or te.sum() < 5:
            continue
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = LogisticDetector(l2).fit(X[tr], y[tr])
        s = m.score(X[te])
        oof[te] = s
        aucs.append(auc_score(y[te], s))
    if not aucs:
        return float("nan"), float("nan"), oof
    return float(np.mean(aucs)), float(np.std(aucs)), oof


def transfer_auc(Xtr, ytr, Xte, yte, l2: float = 1.0):
    """Train on one domain, evaluate on another. Returns AUC on the target."""
    ytr, yte = np.asarray(ytr, int), np.asarray(yte, int)
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float("nan")
    m = LogisticDetector(l2).fit(Xtr, ytr)
    return auc_score(yte, m.score(Xte))


def threshold_at_fpr(y, scores, target_fpr: float = 0.1):
    """Operating threshold achieving <= target FPR; returns (thr, tpr, fpr).

    This is the operationally relevant calibration: a serving system sets a
    probe-trigger threshold, and what matters is how much true divergence it
    catches at a tolerable false-probe rate.
    """
    y = np.asarray(y, int)
    s = np.asarray(scores, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    neg = np.sort(s[y == 0])[::-1]
    if len(neg) == 0 or (y == 1).sum() == 0:
        return float("nan"), float("nan"), float("nan")
    k = max(int(np.floor(target_fpr * len(neg))) - 1, 0)
    thr = neg[k]
    pred = s >= thr
    tpr = float(pred[y == 1].mean())
    fpr = float(pred[y == 0].mean())
    return float(thr), tpr, fpr
