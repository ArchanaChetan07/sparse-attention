import numpy as np

from csa.detector import (LogisticDetector, grouped_cv_auc, threshold_at_fpr,
                          transfer_auc)


def _synth(n=600, seed=0, informative=True):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    if informative:
        X = np.c_[y + 0.7 * rng.normal(size=n),
                  0.5 * y + rng.normal(size=n),
                  rng.normal(size=n)]
    else:
        X = rng.normal(size=(n, 3))
    groups = np.repeat(np.arange(n // 10), 10)[:n]
    return X, y, groups


def test_logistic_learns_separable_signal():
    X, y, _ = _synth()
    m = LogisticDetector().fit(X, y)
    s = m.score(X)
    assert s.min() >= 0 and s.max() <= 1
    assert np.corrcoef(s, y)[0, 1] > 0.4


def test_grouped_cv_auc_beats_chance_on_signal():
    X, y, g = _synth()
    auc, std, oof = grouped_cv_auc(X, y, g)
    assert auc > 0.75
    assert np.isfinite(oof).sum() > 0.8 * len(y)


def test_grouped_cv_auc_is_chance_on_noise():
    """Guards against leakage: pure noise must not produce a high CV AUC."""
    X, y, g = _synth(informative=False)
    auc, _, _ = grouped_cv_auc(X, y, g)
    assert 0.35 < auc < 0.65


def test_standardization_uses_train_stats_only():
    X, y, _ = _synth()
    m = LogisticDetector().fit(X, y)
    mu_before = m.mu.copy()
    m.score(X * 100 + 50)  # scoring must not refit the scaler
    assert np.allclose(mu_before, m.mu)


def test_transfer_auc_runs_and_degrades_on_shift():
    X, y, _ = _synth(seed=1)
    same = transfer_auc(X, y, X, y)
    Xs = X.copy()
    Xs[:, 0] *= -1  # flip the informative feature: transfer should suffer
    shifted = transfer_auc(X, y, Xs, y)
    assert same > 0.75
    assert shifted < same


def test_threshold_at_fpr_respects_budget():
    X, y, _ = _synth(seed=2)
    m = LogisticDetector().fit(X, y)
    thr, tpr, fpr = threshold_at_fpr(y, m.score(X), target_fpr=0.1)
    assert fpr <= 0.12
    assert tpr > fpr  # better than random at that operating point
