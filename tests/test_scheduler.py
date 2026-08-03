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


def test_utilization_bounded():
    r = simulate("inline", 0.2, ticks=1000, seed=0)
    assert 0.0 < r.utilization <= 1.0
