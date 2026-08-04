"""Mechanism 2 — sampled dense verification with anytime-valid bounds.

We bound the *diverged-step fraction* mu of a stream of Bernoulli indicators
x_t in {0,1} ("step t diverged from dense execution"), observing x_t only on
sampled steps (each observation costs one dense probe). All confidence
sequences here are time-uniform (anytime-valid): the interval is legitimate at
every step without pre-committing to a sample size, so the operator statement
"<= q% of steps diverged, at 95% confidence" is always well-posed.

Estimators (Study B compares them at equal probe cost):
  HoeffdingCS          - union-bound time-uniform Hoeffding. Simple, loosest.
  EmpiricalBernsteinCS - predictable plug-in EB (Waudby-Smith & Ramdas 2023).
                         Adapts to variance; tight when divergence is rare.
  BettingCS            - hedged capital process over a grid of candidate means.
                         Tightest in practice.

Sampling policies:
  FixedRateSampler     - probe with constant probability p.
  AdaptiveSampler      - probe probability scales with a label-free signal
                         (Mechanism 1), clipped to [p_min, p_max]. Predictable.

Inverse-propensity (Horvitz-Thompson) weighting keeps the estimate unbiased
under adaptive sampling: z_t = x_t * 1[sampled] / p_t is bounded in
[0, 1/p_min] and has conditional mean x_t; we scale to [0,1] and run the CS on
the scaled stream, then unscale the bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class ConfidenceSequence:
    """Base: anytime-valid CS for the running mean of a [0,1]-bounded stream.

    NOTE on intersection: intersecting intervals across time is only valid
    when the target is a FIXED mean. The serving stream is non-stationary by
    hypothesis (divergence arrives in bursts at a phase boundary), and the
    quantity certified is the RUNNING diverged fraction — a moving target.
    Hoeffding/EB intervals below are simultaneously valid for the running
    (conditional) mean at every t via martingale arguments, so we report the
    raw interval at each t. Set intersect=True only for a stationary target.
    """

    def __init__(self, alpha: float = 0.05, intersect: bool = False):
        self.alpha = alpha
        self.intersect = intersect
        self.n = 0
        self.lo = 0.0
        self.hi = 1.0

    def update(self, x: float):
        raise NotImplementedError

    def _set(self, lo, hi):
        lo, hi = max(0.0, lo), min(1.0, hi)
        if self.intersect:
            lo, hi = max(self.lo, lo), min(self.hi, hi)
            if lo > hi:
                lo = hi = 0.5 * (lo + hi)
        self.lo, self.hi = lo, hi
        return self.lo, self.hi

    @property
    def width(self):
        return self.hi - self.lo


class HoeffdingCS(ConfidenceSequence):
    """Union-bound Hoeffding: alpha_t = alpha / (t (t+1)), sums to alpha.

    Azuma-Hoeffding applies to the martingale sum of (x_i - E[x_i | past]),
    so each interval covers the running conditional mean even under drift;
    the union bound makes coverage simultaneous over all t.
    """

    def __init__(self, alpha: float = 0.05, intersect: bool = False):
        super().__init__(alpha, intersect)
        self.sum = 0.0

    def update(self, x: float):
        self.n += 1
        self.sum += x
        mean = self.sum / self.n
        eps = math.sqrt(math.log(2.0 * self.n * (self.n + 1) / self.alpha)
                        / (2.0 * self.n))
        return self._set(mean - eps, mean + eps)


class EmpiricalBernsteinCS(ConfidenceSequence):
    """Predictable plug-in empirical-Bernstein CS (WSR'23, eq. "EB-CS")."""

    def __init__(self, alpha: float = 0.05, c: float = 0.5,
                 intersect: bool = False):
        super().__init__(alpha, intersect)
        self.c = c
        self.sum_x = 0.0
        self.mu_prev = 0.5      # mu_hat_{t-1} with prior weight 1
        self.var_prev = 0.25    # sigma^2_hat_{t-1} with prior weight 1
        self.sum_sq_dev = 0.25
        self.S_lx = 0.0         # sum lambda_i * x_i
        self.S_l = 0.0          # sum lambda_i
        self.S_v = 0.0          # sum v_i * psi_E(lambda_i)

    @staticmethod
    def _psi_e(lam: float) -> float:
        # Fan's inequality form: valid with v_i = (x_i - mu_hat_{i-1})^2.
        # (WSR'23 state psi/4 with v = 4(.)^2; the factors cancel.)
        return -math.log1p(-lam) - lam

    def update(self, x: float):
        t = self.n + 1
        lam = math.sqrt(2.0 * math.log(2.0 / self.alpha)
                        / (self.var_prev * t * math.log(t + 1.0)))
        lam = min(lam, self.c)
        v = (x - self.mu_prev) ** 2
        self.S_lx += lam * x
        self.S_l += lam
        self.S_v += v * self._psi_e(lam)
        self.n = t
        self.sum_x += x
        self.mu_prev = (0.5 + self.sum_x) / (t + 1.0)
        self.sum_sq_dev += (x - self.mu_prev) ** 2
        self.var_prev = self.sum_sq_dev / (t + 1.0)
        if self.S_l <= 0:
            return self.lo, self.hi
        center = self.S_lx / self.S_l
        rad = (math.log(2.0 / self.alpha) + self.S_v) / self.S_l
        return self._set(center - rad, center + rad)


class BettingCS(ConfidenceSequence):
    """Hedged betting CS: for each candidate mean m, grow capital
    K+(m) = prod(1 + lam_t (x_t - m)) and K-(m) = prod(1 - lam_t (x_t - m));
    reject m once max(K+, K-) ever reaches 1/alpha (Ville's inequality makes
    rejection permanent). aGRAPA-style predictable bets.

    Scope: targets a FIXED mean (exchangeable-ish streams). Permanent
    rejection is intrinsic to the capital-process construction, so under a
    drifting target this CS can lock onto early behaviour — Study B's
    coverage audit quantifies exactly this. Use EB for bursty streams.
    """

    def __init__(self, alpha: float = 0.05, grid: int = 401, c: float = 0.5,
                 intersect: bool = True):
        super().__init__(alpha, intersect)
        self.m = np.linspace(0.0, 1.0, grid)
        self.logK_plus = np.zeros(grid)
        self.logK_minus = np.zeros(grid)
        self.rejected = np.zeros(grid, dtype=bool)
        self.failed = False   # True once every candidate mean is rejected
        self.c = c
        self.sum_x = 0.0
        self.mu_prev = 0.5
        self.var_prev = 0.25
        self.sum_sq_dev = 0.25
        self.thresh = math.log(1.0 / alpha)

    def update(self, x: float):
        t = self.n + 1
        m = self.m
        # predictable bet: approximate-GRAPA centered at running mean
        lam = (self.mu_prev - m) / (self.var_prev + (self.mu_prev - m) ** 2)
        lam_plus = np.clip(lam, 0.0, self.c / np.maximum(m, 1e-4))
        lam_minus = np.clip(-lam, 0.0, self.c / np.maximum(1.0 - m, 1e-4))
        self.logK_plus += np.log1p(lam_plus * (x - m))
        self.logK_minus += np.log1p(-lam_minus * (x - m))
        self.rejected |= (np.maximum(self.logK_plus, self.logK_minus)
                          >= self.thresh)
        self.n = t
        self.sum_x += x
        self.mu_prev = (0.5 + self.sum_x) / (t + 1.0)
        self.sum_sq_dev += (x - self.mu_prev) ** 2
        self.var_prev = self.sum_sq_dev / (t + 1.0)
        alive = ~self.rejected
        if alive.any():
            lo, hi = float(self.m[alive].min()), float(self.m[alive].max())
        else:
            # Every candidate mean has been rejected. That is not a
            # zero-width certificate -- it means the capital process has
            # failed (its fixed-mean assumption is violated, e.g. the target
            # drifted). Degrade to vacuous and say so, rather than emitting a
            # confident point estimate that is exactly what this project
            # exists to prevent.
            self.failed = True
            self.lo, self.hi = 0.0, 1.0
            return self.lo, self.hi
        return self._set(lo, hi)


# ---------------------------------------------------------------------------
# Sampling policies
# ---------------------------------------------------------------------------

class FixedRateSampler:
    def __init__(self, p: float, seed: int = 0):
        self.p = p
        self.p_min = p
        self.rng = np.random.default_rng(seed)

    def prob(self, signal: float | None = None) -> float:
        return self.p

    def draw(self, p: float) -> bool:
        return bool(self.rng.random() < p)


class AdaptiveSampler:
    """Probe probability proportional to a label-free signal.

    p_t = clip(p_base * s_t / (running mean of s + eps), p_min, p_max).
    Uses only information available before the sampling decision, so p_t is
    predictable and Horvitz-Thompson weighting is valid.
    """

    def __init__(self, p_base: float, p_min: float = 0.01, p_max: float = 1.0,
                 seed: int = 0):
        self.p_base = p_base
        self.p_min = p_min
        self.p_max = p_max
        self.rng = np.random.default_rng(seed)
        self._s_sum = 0.0
        self._s_n = 0

    def prob(self, signal: float | None) -> float:
        s = 0.0 if signal is None or not np.isfinite(signal) else max(float(signal), 0.0)
        ref = (self._s_sum / self._s_n) if self._s_n else max(s, 1e-6)
        self._s_sum += s
        self._s_n += 1
        p = self.p_base * (s / max(ref, 1e-9)) if ref > 0 else self.p_base
        return float(np.clip(p, self.p_min, self.p_max))

    def draw(self, p: float) -> bool:
        return bool(self.rng.random() < p)


@dataclass
class VerifierReport:
    n_steps: int
    n_probes: int
    lo: float
    hi: float
    failed: bool = False   # estimator self-reported an invalid certificate

    @property
    def width(self):
        return self.hi - self.lo


class SampledVerifier:
    """Glue: sampler decides when to probe; CS runs on HT-weighted stream.

    call step(x_true, signal): x_true is consumed only when the sampler fires
    (that is the dense probe). Returns (probed, lo, hi) with (lo, hi) bounding
    the true diverged-step fraction.
    """

    def __init__(self, cs: ConfidenceSequence, sampler):
        self.cs = cs
        self.sampler = sampler
        # Horvitz-Thompson weights are bounded by 1/p_min, so a zero floor
        # would make the weighted stream unbounded (and the bound vacuous or
        # a division by zero). A positive floor is a correctness requirement,
        # not a tuning choice.
        if not sampler.p_min > 0:
            raise ValueError("sampler.p_min must be > 0 for HT weighting")
        self.scale = sampler.p_min  # z in [0, 1/p_min] -> scaled to [0,1]
        self.n_steps = 0
        self.n_probes = 0

    def step(self, x_true: int, signal: float | None = None):
        p = self.sampler.prob(signal)
        probed = self.sampler.draw(p)
        z = (x_true / p) if probed else 0.0
        z_scaled = min(z * self.scale, 1.0)
        lo_s, hi_s = self.cs.update(z_scaled)
        self.n_steps += 1
        self.n_probes += int(probed)
        return probed, lo_s / self.scale, min(hi_s / self.scale, 1.0)

    def report(self) -> VerifierReport:
        return VerifierReport(self.n_steps, self.n_probes,
                              self.cs.lo / self.scale,
                              min(self.cs.hi / self.scale, 1.0),
                              bool(getattr(self.cs, "failed", False)))


def make_estimator(kind: str, alpha: float, p: float, seed: int = 0,
                   adaptive: bool = False,
                   p_min: float | None = None) -> SampledVerifier:
    cs = {"hoeffding": HoeffdingCS, "eb": EmpiricalBernsteinCS,
          "betting": BettingCS}[kind](alpha)
    if adaptive:
        floor = p_min if p_min is not None else max(p / 4, 0.005)
        sampler = AdaptiveSampler(p_base=p, p_min=floor,
                                  p_max=min(4 * p, 1.0), seed=seed)
    else:
        sampler = FixedRateSampler(p, seed=seed)
    return SampledVerifier(cs, sampler)
