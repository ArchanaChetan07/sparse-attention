"""Mechanism 3 — verification as elastic work (discrete-time simulator).

Model: a GPU with capacity C token-units per tick serves a dynamic batch of
decoding requests. Each active request wants one decode step (cost 1 unit) per
tick. A verification probe re-executes one step with dense attention: cost
`probe_cost` units (> 1, since dense attention over a long context is more
expensive than sparse). Probes are drawn per executed decode step with
probability `probe_rate`.

Policies:
  none    - no verification (status quo: unverified accuracy claim).
  inline  - a drawn probe executes in the same tick as its decode step. Under
            load, probes compete with decode -> latency (TPOT) degrades.
  elastic - drawn probes queue and consume only slack capacity, prioritized by
            the widest current confidence bound (max-min tightness). Under
            load, probes are displaced -> the *bound widens* while TPOT tracks
            the no-verification baseline. Contention degrades the guarantee,
            not the output (proposal §6.3).

Bound width is computed from each request's executed probe count via the
time-uniform Hoeffding radius — the guarantee the request ends with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def hoeffding_width(n_probes: int, alpha: float = 0.05) -> float:
    """Two-sided anytime-valid Hoeffding CS width after n probes."""
    if n_probes <= 0:
        return 1.0
    n = n_probes
    eps = math.sqrt(math.log(2.0 * n * (n + 1) / alpha) / (2.0 * n))
    return min(1.0, 2.0 * eps)


@dataclass
class Request:
    rid: int
    arrival: int
    decode_len: int
    steps_done: int = 0
    finish_tick: int = -1
    probes_wanted: int = 0
    probes_done: int = 0
    step_delays: list = field(default_factory=list)  # ticks per decode step
    _wait: int = 0

    def bound_width(self, alpha: float = 0.05) -> float:
        return hoeffding_width(self.probes_done, alpha)


@dataclass
class SimResult:
    policy: str
    arrival_rate: float
    utilization: float
    tpot_p50: float
    tpot_p99: float
    mean_width: float        # per-request bound width (needs ~1e3 probes to be tight)
    p90_width: float
    system_width: float      # width of the pooled system-level bound — the
                             # operator statement "x% of decode steps diverged"
    probe_completion: float  # fraction of drawn probes actually executed
    n_requests: int


def simulate(policy: str, arrival_rate: float, ticks: int = 4000,
             capacity: float = 8.0, probe_rate: float = 0.1,
             probe_cost: float = 3.0, mean_decode_len: int = 96,
             retention_ticks: int = 400, seed: int = 0,
             value_weighted: bool = False) -> SimResult:
    """Run one load point. arrival_rate = expected new requests per tick."""
    assert policy in ("none", "inline", "elastic")
    rng = np.random.default_rng(seed)
    arrivals = rng.poisson(arrival_rate, size=ticks)
    reqs: list[Request] = []
    active: list[Request] = []
    probe_queue: list[tuple[Request, int]] = []  # (request, created_tick)
    rid = 0
    used_total = 0.0

    for t in range(ticks):
        for _ in range(arrivals[t]):
            dl = max(8, int(rng.exponential(mean_decode_len)))
            reqs.append(Request(rid, t, dl))
            active.append(reqs[-1])
            rid += 1

        cap = capacity
        # serve decode steps, longest-waiting first
        order = sorted(active, key=lambda r: -r._wait)
        finished = []
        for r in order:
            draw_probe = policy != "none" and rng.random() < probe_rate
            cost = 1.0 + (probe_cost if (policy == "inline" and draw_probe) else 0.0)
            if cap < cost:
                r._wait += 1
                continue
            cap -= cost
            used_total += cost
            r.step_delays.append(r._wait + 1)
            r._wait = 0
            r.steps_done += 1
            if draw_probe:
                r.probes_wanted += 1
                if policy == "inline":
                    r.probes_done += 1
                else:  # elastic: defer into the queue
                    probe_queue.append((r, t))
            if r.steps_done >= r.decode_len:
                r.finish_tick = t
                finished.append(r)
        for r in finished:
            active.remove(r)

        if policy == "elastic" and cap > 0 and probe_queue:
            # drop probes whose KV state is no longer retained
            probe_queue = [(r, ct) for (r, ct) in probe_queue
                           if t - ct <= retention_ticks]
            # max-min tightness: widest current bound first; value-weighted
            # variant scales priority by remaining decode length
            def prio(item):
                r, _ = item
                w = r.bound_width()
                if value_weighted:
                    w *= (r.decode_len - r.steps_done + 1) / r.decode_len
                return -w
            probe_queue.sort(key=prio)
            while cap >= probe_cost and probe_queue:
                r, _ = probe_queue.pop(0)
                r.probes_done += 1
                cap -= probe_cost
                used_total += probe_cost

    done = [r for r in reqs if r.finish_tick >= 0]
    delays = np.array([d for r in done for d in r.step_delays], dtype=float)
    widths = (np.array([r.bound_width() for r in done])
              if done else np.array([1.0]))
    wanted = sum(r.probes_wanted for r in reqs)
    got = sum(r.probes_done for r in reqs)
    return SimResult(
        policy=policy,
        arrival_rate=arrival_rate,
        utilization=min(used_total / (capacity * ticks), 1.0),
        tpot_p50=float(np.percentile(delays, 50)) if len(delays) else float("nan"),
        tpot_p99=float(np.percentile(delays, 99)) if len(delays) else float("nan"),
        mean_width=float(widths.mean()),
        p90_width=float(np.percentile(widths, 90)),
        system_width=hoeffding_width(got),
        probe_completion=(got / wanted) if wanted else 1.0,
        n_requests=len(done),
    )


def sweep(policies=("none", "inline", "elastic"), rates=None, **kw):
    if rates is None:
        rates = np.linspace(0.02, 0.11, 8)
    return [simulate(pol, float(rate), **kw)
            for pol in policies for rate in rates]
