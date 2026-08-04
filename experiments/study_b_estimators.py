"""Study B — estimator design: bound width vs verification cost (H2).

Replays Study A step traces (exact per-step divergence labels + label-free
signals) through the Mechanism-2 estimators at a range of probe budgets, and
answers the pre-registered H2 question directly:

    H2 is FALSIFIED if a +/-5% bound at 95% confidence costs > 15% throughput.

Throughput cost is derived from the measured probe/step time ratio r:
    overhead    = p * r                 (p = probes per decode step)
    throughput  = 1 / (1 + p*r)   ->    loss = p*r / (1 + p*r)

Also runs a coverage audit under three regimes (rare-iid, moderate-iid, and
bursty — the phase-transition regime the proposal predicts), which is where
the fixed-mean assumption behind capital-process CSs is stressed.

Outputs under results/study_b/.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from csa.recording import save_results
from csa.verify import make_estimator

ESTIMATORS = [
    ("hoeffding", False, "fixed + Hoeffding"),
    ("eb", False, "fixed + EmpBernstein"),
    ("betting", False, "fixed + Betting"),
    ("eb", True, "adaptive + EmpBernstein"),
    ("betting", True, "adaptive + Betting"),
]
PROBE_RATES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
N_SEEDS = 20
TARGET_WIDTHS = {"+/-5% (w=0.10)": 0.10, "+/-2.5% (w=0.05)": 0.05,
                 "+/-10% (w=0.20)": 0.20}


def load_stream(steps_csv: Path, mode: str = "teacher"):
    df = pd.read_csv(steps_csv)
    df = df[(df["mode"] == mode) & df["top1_flip"].notna()]
    df = df.sort_values(["keep_frac", "task_id", "method", "step"])
    x = df["top1_flip"].astype(int).to_numpy()
    sig = df["est_dropped_mean_Lmean"].fillna(0.0).to_numpy()
    return x, sig, df


def _replay(kind, adaptive, p, seed, x, sig):
    v = make_estimator(kind, 0.05, p=p, seed=seed, adaptive=adaptive)
    for xi, si in zip(x, sig):
        v.step(int(xi), float(si))
    return v.report()


def width_vs_cost(df: pd.DataFrame, out_rows, per_budget=True):
    """Replay per BUDGET, because that is the realistic deployment stream.

    A stream concatenating several budgets is non-stationary by construction
    (the diverged fraction jumps at every budget boundary), which stresses the
    fixed-mean assumption behind capital-process estimators. Both are reported:
    per-budget = a single deployment; 'mixed' = heterogeneous traffic.
    """
    groups = [(f"keep={kf:g}", g) for kf, g in df.groupby("keep_frac")] \
        if per_budget else []
    groups.append(("mixed", df))

    for gname, g in groups:
        x = g["top1_flip"].astype(int).to_numpy()
        sig = g["est_dropped_mean_Lmean"].fillna(0.0).to_numpy()
        mu = float(x.mean())
        if len(x) < 50 or mu <= 0:
            continue
        for kind, adaptive, label in ESTIMATORS:
            for p in PROBE_RATES:
                res = [_replay(kind, adaptive, p, s, x, sig)
                       for s in range(N_SEEDS)]
                out_rows.append(dict(
                    stream=gname, estimator=label, target_rate=p,
                    probe_rate=float(np.mean([r.n_probes / max(r.n_steps, 1)
                                              for r in res])),
                    width_mean=float(np.mean([r.width for r in res])),
                    width_std=float(np.std([r.width for r in res])),
                    coverage=float(np.mean([r.lo - 1e-9 <= mu <= r.hi + 1e-9
                                            for r in res])),
                    true_divergence=mu, n_steps=int(len(x))))


MIN_COVERAGE = 0.90  # 1-alpha = 0.95, minus Monte-Carlo slack at N_SEEDS


def h2_test(wdf: pd.DataFrame, cost_ratio: float, stream: str = "mixed"):
    """Cheapest VALID estimator reaching each target width, and its cost.

    Validity is a precondition, not a tiebreak. Selecting on width alone picks
    whichever estimator undercovers most aggressively -- a narrow interval that
    is usually wrong is worse than no interval, and it would produce exactly
    the kind of unverified accuracy claim this project exists to eliminate.
    """
    out = {}
    sub = wdf[wdf["stream"] == stream]
    for tname, target in TARGET_WIDTHS.items():
        best, rejected = None, []
        for est, g in sub.groupby("estimator"):
            g = g.sort_values("probe_rate")
            hit = g[g["width_mean"] <= target]
            if not len(hit):
                continue
            valid = hit[hit["coverage"] >= MIN_COVERAGE]
            if not len(valid):
                rejected.append({"estimator": est,
                                 "best_coverage": float(hit["coverage"].max())})
                continue
            row = valid.iloc[0]
            if best is None or row["probe_rate"] < best["probe_rate"]:
                best = {"estimator": est, "probe_rate": float(row["probe_rate"]),
                        "width": float(row["width_mean"]),
                        "coverage": float(row["coverage"])}
        if best is None:
            out[tname] = {"reached": False, "rejected_for_undercoverage": rejected,
                          "note": "no valid estimator reached this width"}
        else:
            ov = best["probe_rate"] * cost_ratio
            best.update(reached=True, cost_ratio=cost_ratio, overhead_frac=ov,
                        throughput_loss=ov / (1.0 + ov),
                        rejected_for_undercoverage=rejected)
            out[tname] = best
    return out


def probes_needed(mu: float, targets=(0.20, 0.10, 0.05), alpha=0.05,
                  max_probes=200_000, seeds=8):
    """How many PROBES does each target width need? (scale-free)

    Bound width falls as ~1/sqrt(n_probes), so on a short trace no probe rate
    reaches a tight bound -- the binding constraint is the number of probes,
    not the fraction of steps probed. Reporting cost as a rate on a 300-step
    trace would measure trace length, not verification cost.

    Probes every step (p=1) on a synthetic Bernoulli(mu) stream, so probes ==
    steps and the answer is the probe count itself. The implied *rate* for a
    real deployment is then n_probes / stream_length, computed separately.
    """
    out = {}
    for kind, adaptive, label in ESTIMATORS:
        if adaptive:
            continue  # rate-shaping is irrelevant when every step is probed
        per_seed = {f"{t:g}": [] for t in targets}
        for seed in range(seeds):
            rng = np.random.default_rng(1000 + seed)
            v = make_estimator(kind, alpha, p=1.0, seed=seed)
            remaining = {f"{t:g}": None for t in targets}
            n = 0
            while n < max_probes and any(x is None for x in remaining.values()):
                v.step(int(rng.random() < mu), None)
                n += 1
                if n % 25:
                    continue
                r = v.report()
                for t in targets:
                    key = f"{t:g}"
                    if remaining[key] is None and not r.failed and r.width <= t:
                        remaining[key] = n
            for k, val in remaining.items():
                per_seed[k].append(val if val is not None else max_probes)
        # MEDIAN across seeds, not min: the minimum reports the luckiest run
        # and would understate the cost of the guarantee.
        out[label] = {k: (int(np.median(vals)) if vals else None)
                      for k, vals in per_seed.items()}
        out[label] = {k: (None if v is not None and v >= max_probes else v)
                      for k, v in out[label].items()}
    return out


def implied_rates(need: dict, cost_ratio: float,
                  stream_lengths=(1_000, 10_000, 100_000, 1_000_000)):
    """Translate 'probes needed' into a probe rate and throughput cost for
    streams of realistic length. A single short request cannot afford a tight
    bound; a long request or an aggregated tenant stream can."""
    rows = []
    for est, targets in need.items():
        for tgt, n in targets.items():
            if n is None:
                continue
            for L in stream_lengths:
                p = n / L
                if p > 1.0:
                    rows.append(dict(estimator=est, target_width=float(tgt),
                                     probes_needed=n, stream_len=L,
                                     probe_rate=float("nan"),
                                     throughput_loss=float("nan"),
                                     feasible=False))
                    continue
                ov = p * cost_ratio
                rows.append(dict(estimator=est, target_width=float(tgt),
                                 probes_needed=n, stream_len=L, probe_rate=p,
                                 throughput_loss=ov / (1 + ov), feasible=True))
    return rows


def coverage_audit(out_rows):
    """Anytime coverage of the RUNNING diverged fraction, three regimes."""
    rng = np.random.default_rng(0)
    T, reps = 1200, 120
    regimes = {
        "iid_rare(p=.03)": lambda: (rng.random(T) < 0.03).astype(int),
        "iid_moderate(p=.20)": lambda: (rng.random(T) < 0.2).astype(int),
        "bursty(.01/.35)": lambda: (rng.random(T) < np.where(
            (np.arange(T) // 100) % 3 == 2, 0.35, 0.01)).astype(int),
    }
    for regime, gen in regimes.items():
        for kind, adaptive, label in ESTIMATORS:
            misses, final_misses = 0, 0
            for rep in range(reps):
                xs = gen()
                v = make_estimator(kind, 0.05, p=0.15, seed=rep, adaptive=adaptive)
                sigs = 0.1 + 0.8 * xs + 0.15 * rng.random(T)
                seen, anytime_ok = 0.0, True
                for t, (xi, si) in enumerate(zip(xs, sigs), 1):
                    v.step(int(xi), float(si))
                    seen += xi
                    lo = v.cs.lo / v.scale
                    hi = min(v.cs.hi / v.scale, 1.0)
                    if not (lo - 1e-9 <= seen / t <= hi + 1e-9):
                        anytime_ok = False
                misses += not anytime_ok
                r = v.report()
                final_misses += not (r.lo - 1e-9 <= xs.mean() <= r.hi + 1e-9)
            out_rows.append(dict(regime=regime, estimator=label,
                                 anytime_miss_rate=misses / reps,
                                 final_miss_rate=final_misses / reps,
                                 target_alpha=0.05))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="results/study_a_1.5b/steps.csv")
    ap.add_argument("--summary", default=None,
                    help="Study A summary.json, for the measured probe/step ratio")
    ap.add_argument("--cost-ratio", type=float, default=None,
                    help="override probe_time / sparse_step_time")
    ap.add_argument("--out", default="results/study_b")
    args = ap.parse_args()
    out = Path(args.out)

    steps_path = Path(args.steps)
    if not steps_path.exists():
        raise SystemExit(f"{steps_path} not found — run Study A first")
    x, sig, df = load_stream(steps_path)
    print(f"replaying {len(x)} teacher-forced steps, "
          f"true diverged fraction = {x.mean():.4f}")

    # Cost ratio precedence: an explicit override, else the MEASURED ratio
    # from overhead_bench (production gather path -- the honest number), else
    # the paired-step ratio from Study A, which understates r because the
    # paired step computes dense and sparse together.
    cost_ratio, cost_src = args.cost_ratio, "override"
    if cost_ratio is None:
        ob = Path("results/overhead/overhead.json")
        if ob.exists():
            rows = json.load(open(ob))["rows"]
            cost_ratio = float(np.mean([r["cost_ratio_r"] for r in rows]))
            cost_src = "measured (overhead_bench, gather path)"
    if cost_ratio is None:
        sp = Path(args.summary) if args.summary else steps_path.parent / "summary.json"
        if sp.exists():
            t = json.load(open(sp)).get("timing_ms", {})
            cost_ratio = t.get("probe_over_paired_step") or t.get("probe_over_step")
            cost_src = "paired-step ratio (UNDERSTATES r)"
    if cost_ratio is None:
        cost_ratio, cost_src = 1.0, "default fallback"
    print(f"probe/step cost ratio r = {cost_ratio:.3f}  [{cost_src}]")

    rows_w, rows_c = [], []
    width_vs_cost(df, rows_w)
    coverage_audit(rows_c)
    save_results(rows_w, {"steps": str(steps_path)}, out, "width_vs_cost")
    save_results(rows_c, {}, out, "coverage")
    wdf = pd.DataFrame(rows_w)
    cdf = pd.DataFrame(rows_c)

    h2 = {s: h2_test(wdf, cost_ratio, s) for s in wdf["stream"].unique()}
    # the deployable claim is per-budget: a request runs at ONE budget
    per_budget = {k: v for k, v in h2.items() if k != "mixed"}
    verdicts = []
    for s, block in per_budget.items():
        five = block.get("+/-5% (w=0.10)", {})
        if five.get("reached"):
            verdicts.append(five["throughput_loss"] <= 0.15)
    h2_verdict = ("NOT FALSIFIED" if verdicts and all(verdicts) else
                  "FALSIFIED" if verdicts else "INCONCLUSIVE")

    # validity-first estimator ranking: mean width among estimators that
    # actually cover, across per-budget streams
    valid = wdf[(wdf["coverage"] >= MIN_COVERAGE) & (wdf["stream"] != "mixed")]
    best_est = (valid.groupby("estimator")["width_mean"].mean().idxmin()
                if len(valid) else None)
    beta = float("nan")
    if best_est:
        g = valid[valid["estimator"] == best_est]
        g = g.groupby("probe_rate")["width_mean"].mean().reset_index()
        ok = (g["width_mean"] > 0) & (g["probe_rate"] > 0)
        if ok.sum() >= 3:
            beta = float(-np.polyfit(np.log(g.loc[ok, "probe_rate"]),
                                     np.log(g.loc[ok, "width_mean"]), 1)[0])

    undercover = (wdf[wdf["coverage"] < MIN_COVERAGE]
                  .groupby("estimator")["coverage"].min().to_dict())

    # scale-free H2: probes needed, then the rate that implies at realistic
    # stream lengths (a 300-step trace cannot afford a tight bound at ANY rate)
    need = probes_needed(float(x.mean()))
    rate_rows = implied_rates(need, cost_ratio)
    save_results(rate_rows, {"mu": float(x.mean())}, out, "implied_rates")
    rdf = pd.DataFrame(rate_rows)

    # Only estimators that survived the coverage audit may support a verdict.
    # Betting is tightest and invalid under the bursty regime the
    # phase-transition premise predicts, so it must not set the headline.
    audit = cdf.groupby("estimator")["anytime_miss_rate"].max()
    valid_ests = [e for e in need if audit.get(e, 1.0) <= 0.05 + 0.02]
    feasible = rdf[(rdf["target_width"] == 0.10) & rdf["feasible"]
                   & (rdf["throughput_loss"] <= 0.15)
                   & rdf["estimator"].isin(valid_ests)]
    h2_scalefree_verdict = (
        f"NOT FALSIFIED at stream length >= {int(feasible['stream_len'].min())} "
        f"steps (valid estimators only: {', '.join(valid_ests)})"
        if len(feasible) else
        "FALSIFIED at all tested stream lengths for coverage-valid estimators")
    summary = {
        "trace_steps": int(len(x)),
        "true_divergence": float(x.mean()),
        "cost_ratio_probe_over_step": float(cost_ratio),
        "cost_ratio_source": cost_src,
        "h2_targets_by_stream": h2,
        "h2_verdict_at_5pct_on_short_traces": h2_verdict,
        "h2_probes_needed_for_width": need,
        "h2_verdict_scalefree": h2_scalefree_verdict,
        "coverage_valid_estimators": valid_ests,
        "best_valid_estimator": best_est,
        "width_scaling_exponent_beta": beta,
        "estimators_undercovering_on_replay": undercover,
        "coverage_worst_anytime_miss": float(cdf["anytime_miss_rate"].max()),
        "coverage_by_estimator_bursty": cdf[cdf["regime"].str.startswith("bursty")]
            .set_index("estimator")["anytime_miss_rate"].to_dict(),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n=== width vs cost (mixed stream) ===")
    mixed = wdf[wdf["stream"] == "mixed"]
    print(mixed.pivot_table(index="target_rate", columns="estimator",
                            values="width_mean").to_string())
    print("\n=== coverage on replay (must be >= 0.95; validity gates width) ===")
    print(wdf.pivot_table(index="estimator", columns="stream",
                          values="coverage").to_string())
    print("\n=== coverage audit (miss rate; target alpha = 0.05) ===")
    print(cdf.pivot_table(index="estimator", columns="regime",
                          values="anytime_miss_rate").to_string())
    print("\n" + json.dumps(summary, indent=2, default=float))

    _figures(wdf, cdf, x, out, cost_ratio)


def _figures(wdf, cdf, x, out: Path, cost_ratio: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for est, g in wdf[wdf["stream"] == "mixed"].groupby("estimator"):
        g = g.sort_values("probe_rate")
        axes[0].plot(g["probe_rate"] * 100, g["width_mean"], "o-", label=est)
        axes[0].fill_between(g["probe_rate"] * 100,
                             g["width_mean"] - g["width_std"],
                             g["width_mean"] + g["width_std"], alpha=0.12)
        loss = g["probe_rate"] * cost_ratio / (1 + g["probe_rate"] * cost_ratio)
        axes[1].plot(loss * 100, g["width_mean"], "o-", label=est)
    axes[0].axhline(0.10, ls="--", c="k", lw=1, label="+/-5% target width")
    axes[0].set(xlabel="dense probes per 100 decode steps",
                ylabel="95% bound width", xscale="log",
                title=f"Bound width vs verification cost (mu={x.mean():.3f})")
    axes[1].axhline(0.10, ls="--", c="k", lw=1)
    axes[1].axvline(15, ls=":", c="crimson", lw=1.5, label="H2 falsification (15%)")
    axes[1].set(xlabel=f"throughput loss % (r={cost_ratio:.2f})",
                ylabel="95% bound width",
                title="H2: is a +/-5% bound affordable?")
    for a in axes:
        a.legend(fontsize=7.5)
    plt.tight_layout(); plt.savefig(fig_dir / "width_vs_cost.png", dpi=140)
    plt.close()

    piv = cdf.pivot_table(index="estimator", columns="regime",
                          values="anytime_miss_rate")
    plt.figure(figsize=(7.5, 3.6))
    plt.imshow(piv.values, vmin=0, vmax=max(0.1, float(piv.values.max())),
               cmap="RdYlGn_r")
    plt.colorbar(label="anytime miss rate")
    plt.xticks(range(len(piv.columns)), piv.columns, rotation=20, ha="right")
    plt.yticks(range(len(piv.index)), piv.index, fontsize=8)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            plt.text(j, i, f"{piv.values[i, j]:.2f}", ha="center",
                     va="center", fontsize=8)
    plt.title("Coverage audit: miss rate vs target alpha = 0.05")
    plt.tight_layout(); plt.savefig(fig_dir / "coverage.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    main()
