import numpy as np
import pytest

from csa.verify import (BettingCS, EmpiricalBernsteinCS, HoeffdingCS,
                        make_estimator)

ALPHA = 0.05
CS_CLASSES = [HoeffdingCS, EmpiricalBernsteinCS, BettingCS]


@pytest.mark.parametrize("cls", CS_CLASSES)
@pytest.mark.parametrize("mu", [0.05, 0.3])
def test_cs_coverage_anytime(cls, mu):
    """Time-uniform coverage: across reps, mu escapes the CS at ANY time in
    at most ~alpha of runs (allow monte-carlo slack)."""
    rng = np.random.default_rng(7)
    reps, T = 120, 800
    misses = 0
    for _ in range(reps):
        cs = cls(ALPHA)
        x = (rng.random(T) < mu).astype(float)
        violated = False
        for xi in x:
            lo, hi = cs.update(float(xi))
            if not (lo - 1e-9 <= mu <= hi + 1e-9):
                violated = True
        misses += violated
    assert misses / reps <= ALPHA + 0.05, f"{cls.__name__}: {misses}/{reps} runs missed"


@pytest.mark.parametrize("cls", CS_CLASSES)
def test_cs_width_shrinks(cls):
    rng = np.random.default_rng(1)
    cs = cls(ALPHA)
    w200 = w800 = None
    for t in range(800):
        cs.update(float(rng.random() < 0.1))
        if t == 199:
            w200 = cs.width
    w800 = cs.width
    assert w800 < w200 < 1.0


def test_betting_tighter_than_hoeffding_for_rare_events():
    rng = np.random.default_rng(3)
    x = (rng.random(1500) < 0.02).astype(float)
    h, b = HoeffdingCS(ALPHA), BettingCS(ALPHA)
    for xi in x:
        h.update(float(xi))
        b.update(float(xi))
    assert b.width < h.width


@pytest.mark.parametrize("adaptive", [False, True])
def test_sampled_verifier_bounds_contain_truth(adaptive):
    """End-to-end: probe ~10% of steps; the unscaled bound must contain the
    true diverged fraction. Signal correlates with divergence in adaptive mode."""
    rng = np.random.default_rng(11)
    reps, T, mu = 60, 1500, 0.08
    misses = 0
    for rep in range(reps):
        v = make_estimator("eb", ALPHA, p=0.1, seed=rep, adaptive=adaptive)
        xs = (rng.random(T) < mu).astype(int)
        signals = 0.2 + 0.8 * xs + 0.1 * rng.random(T)  # informative signal
        for x, s in zip(xs, signals):
            v.step(int(x), float(s))
        r = v.report()
        true_frac = xs.mean()
        if not (r.lo - 1e-9 <= true_frac <= r.hi + 1e-9):
            misses += 1
        assert 0 < r.n_probes < T
    assert misses / reps <= ALPHA + 0.07


def test_probe_rate_matches_target():
    v = make_estimator("hoeffding", ALPHA, p=0.15, seed=0)
    rng = np.random.default_rng(5)
    for _ in range(4000):
        v.step(int(rng.random() < 0.05), None)
    rate = v.n_probes / v.n_steps
    assert abs(rate - 0.15) < 0.03
