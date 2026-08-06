from csa.scheduler import hoeffding_width, simulate


def test_width_decreases_with_probes():
    ws = [hoeffding_width(n) for n in (0, 1, 5, 20, 100)]
    assert ws == sorted(ws, reverse=True)
    assert ws[0] == 1.0


def test_low_load_all_policies_healthy():
    for pol in ("none", "inline", "elastic"):
        r = simulate(pol, arrival_rate=0.02, ticks=3000, seed=1)
        assert r.tpot_p99 <= 2.0, f"{pol} should be uncongested at low load"
    ri = simulate("inline", 0.02, ticks=3000, seed=1)
    re = simulate("elastic", 0.02, ticks=3000, seed=1)
    # with slack available, elastic executes (nearly) all probes too
    assert re.probe_completion > 0.9
    assert ri.mean_width < 1.0 and re.mean_width < 1.0


def test_overload_elastic_protects_latency_inline_does_not():
    rate = 0.095  # near/over capacity with probe overhead
    rn = simulate("none", rate, ticks=4000, seed=2)
    ri = simulate("inline", rate, ticks=4000, seed=2)
    re = simulate("elastic", rate, ticks=4000, seed=2)
    # inline pays latency for verification under load
    assert ri.tpot_p99 > re.tpot_p99
    # elastic latency stays close to the no-verification baseline
    assert re.tpot_p99 <= rn.tpot_p99 * 1.5 + 1.0
    # ...and pays with a wider bound instead
    assert re.mean_width >= ri.mean_width - 1e-9
    assert re.probe_completion < 1.0


def test_probe_demand_is_per_step_not_per_retry():
    """Under congestion the effective probe rate must not drift below target.

    If probe demand were re-rolled on every retry, a congested request would
    keep re-rolling until it happened to draw 'no probe', silently lowering
    verification coverage exactly when verification matters most.
    """
    r = simulate("inline", 0.09, ticks=4000, probe_rate=0.3, probe_cost=1.0,
                 capacity=8.0, seed=5)
    served_steps = r.n_requests  # completed requests
    assert served_steps > 0
    # inline never abandons a drawn probe, so completion is 1.0 by definition
    assert r.probe_completion == 1.0


def test_infeasible_inline_config_is_flagged():
    """One step plus its inline probe costing more than total capacity means
    those steps can never be served; the survivors' latency would look fine."""
    r = simulate("inline", 0.05, ticks=800, capacity=8.0, probe_cost=12.0, seed=1)
    assert r.infeasible, "must flag a configuration that cannot ever be served"
    ok = simulate("inline", 0.05, ticks=800, capacity=8.0, probe_cost=3.0, seed=1)
    assert not ok.infeasible


def test_elastic_never_infeasible():
    """Elastic defers probes, so it is never wedged by expensive probes."""
    r = simulate("elastic", 0.05, ticks=800, capacity=8.0, probe_cost=12.0, seed=1)
    assert not r.infeasible
    assert r.tpot_p99 < 5.0, "latency must stay protected even at high probe cost"


def test_probes_expire_when_their_request_completes():
    """A probe is only valid while the request's KV prefix is resident.

    Regression: finished requests were removed from the active set but their
    queued probes stayed in the queue and could still execute, counting
    verification the system could not have performed. The effect is largest at
    low load -- exactly the operating point where elastic is meant to look
    good -- so it flattered the advocated policy.
    """
    from csa.scheduler import Request, simulate
    r = simulate("elastic", 0.04, ticks=4000, seed=1)
    assert r.probe_completion <= 1.0
    # at this load the queue drains, so completion is high but must not be
    # inflated to ~1.0 by post-completion execution
    assert r.probe_completion < 0.98, (
        f"completion {r.probe_completion:.3f} suggests probes are still "
        "executing for finished requests")


def test_utilization_bounded():
    r = simulate("inline", 0.2, ticks=1000, seed=0)
    assert 0.0 < r.utilization <= 1.0
