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


def test_betting_failure_degrades_to_vacuous_not_to_a_point():
    """A capital process that rejects every candidate has FAILED. It must not
    report a zero-width interval -- a confident-looking wrong certificate is
    the exact failure mode this project exists to prevent."""
    cs = BettingCS(ALPHA, grid=101)
    # a hard regime shift: long stretch at 0, then a long stretch at 1
    for _ in range(400):
        cs.update(0.0)
    for _ in range(400):
        cs.update(1.0)
    if cs.failed:
        assert (cs.lo, cs.hi) == (0.0, 1.0)
        assert cs.width == 1.0
    else:
        assert cs.width > 0.0, "a surviving CS must have positive width"


def test_report_surfaces_failure_flag():
    v = make_estimator("betting", ALPHA, p=1.0, seed=0)
    for i in range(300):
        v.step(0 if i < 200 else 1, None)
    r = v.report()
    assert isinstance(r.failed, bool)
    if r.failed:
        assert r.width == 1.0


def test_hoeffding_tracks_drifting_running_mean():
    """Regression for the running-intersection bug: with a late burst, a CS
    that intersects across time freezes near the pre-burst mean and misses
    the post-burst running mean. Raw union-bound Hoeffding must track it."""
    rng = np.random.default_rng(9)
    reps, T = 60, 900
    misses = 0
    for _ in range(reps):
        cs = HoeffdingCS(ALPHA)
        seen, ok = 0.0, True
        for t in range(1, T + 1):
            p = 0.01 if t <= 600 else 0.5  # phase-transition-style burst
            x = float(rng.random() < p)
            seen += x
            lo, hi = cs.update(x)
            if not (lo - 1e-9 <= seen / t <= hi + 1e-9):
                ok = False
        misses += not ok
    assert misses / reps <= ALPHA + 0.05


def test_zero_probe_floor_is_rejected():
    """HT weights are bounded by 1/p_min; a zero floor makes them unbounded."""
    from csa.verify import EmpiricalBernsteinCS, FixedRateSampler, SampledVerifier
    with pytest.raises(ValueError):
        SampledVerifier(EmpiricalBernsteinCS(ALPHA), FixedRateSampler(0.0))


def test_explicit_p_min_is_honoured_not_overridden():
    v = make_estimator("eb", ALPHA, p=0.2, seed=0, adaptive=True, p_min=0.2)
    assert v.sampler.p_min == 0.2
    assert v.scale == 0.2


def test_probe_rate_matches_target():
    v = make_estimator("hoeffding", ALPHA, p=0.15, seed=0)
    rng = np.random.default_rng(5)
    for _ in range(4000):
        v.step(int(rng.random() < 0.05), None)
    rate = v.n_probes / v.n_steps
    assert abs(rate - 0.15) < 0.03
