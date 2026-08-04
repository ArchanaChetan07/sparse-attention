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
    df = df.sort_values(["task_id", "method", "keep_frac", "step"])
    x = df["top1_flip"].astype(int).to_numpy()
    sig = df["est_dropped_mean_Lmean"].fillna(0.0).to_numpy()
    return x, sig, df


def width_vs_cost(x, sig, out_rows):
    mu = x.mean()
    for kind, adaptive, label in ESTIMATORS:
        for p in PROBE_RATES:
            widths, probes, covered = [], [], []
            for seed in range(N_SEEDS):
                v = make_estimator(kind, 0.05, p=p, seed=seed, adaptive=adaptive)
                for xi, si in zip(x, sig):
                    v.step(int(xi), float(si))
                r = v.report()
                widths.append(r.width)
                probes.append(r.n_probes / max(r.n_steps, 1))
                covered.append(r.lo - 1e-9 <= mu <= r.hi + 1e-9)
            out_rows.append(dict(
                estimator=label, target_rate=p,
                probe_rate=float(np.mean(probes)),
                width_mean=float(np.mean(widths)),
                width_std=float(np.std(widths)),
                coverage=float(np.mean(covered)),
                true_divergence=float(mu), n_steps=int(len(x))))


def h2_test(wdf: pd.DataFrame, cost_ratio: float):
    """Minimum probe rate reaching each target width, and its throughput cost."""
    out = {}
    for tname, target in TARGET_WIDTHS.items():
        best = None
        for est, g in wdf.groupby("estimator"):
            g = g.sort_values("probe_rate")
            hit = g[g["width_mean"] <= target]
            if not len(hit):
                continue
            row = hit.iloc[0]
            if best is None or row["probe_rate"] < best["probe_rate"]:
                best = {"estimator": est, "probe_rate": float(row["probe_rate"]),
                        "width": float(row["width_mean"]),
                        "coverage": float(row["coverage"])}
        if best is None:
            out[tname] = {"reached": False,
                          "note": "not reached within swept probe rates"}
        else:
            ov = best["probe_rate"] * cost_ratio
            best.update(reached=True, cost_ratio=cost_ratio,
                        overhead_frac=ov, throughput_loss=ov / (1.0 + ov))
            out[tname] = best
    return out


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

    cost_ratio = args.cost_ratio
    if cost_ratio is None:
        sp = Path(args.summary) if args.summary else steps_path.parent / "summary.json"
        if sp.exists():
            with open(sp) as f:
                cost_ratio = json.load(f)["timing_ms"]["probe_over_step"]
        else:
            cost_ratio = 1.0
    print(f"probe/step cost ratio r = {cost_ratio:.3f}")

    rows_w, rows_c = [], []
    width_vs_cost(x, sig, rows_w)
    coverage_audit(rows_c)
    save_results(rows_w, {"steps": str(steps_path)}, out, "width_vs_cost")
    save_results(rows_c, {}, out, "coverage")
    wdf = pd.DataFrame(rows_w)
    cdf = pd.DataFrame(rows_c)

    h2 = h2_test(wdf, cost_ratio)
    five = h2.get("+/-5% (w=0.10)", {})
    h2_verdict = ("NOT FALSIFIED" if five.get("reached")
                  and five.get("throughput_loss", 1) <= 0.15 else
                  "FALSIFIED" if five.get("reached") else "INCONCLUSIVE")

    # sublinearity: does cost grow slower than the guarantee tightens?
    # width ~ C * p^(-beta); beta = 0.5 is the sqrt-n rate.
    best_est = wdf.groupby("estimator")["width_mean"].min().idxmin()
    g = wdf[wdf["estimator"] == best_est].sort_values("probe_rate")
    ok = (g["width_mean"] > 0) & (g["probe_rate"] > 0)
    beta = float(-np.polyfit(np.log(g.loc[ok, "probe_rate"]),
                             np.log(g.loc[ok, "width_mean"]), 1)[0])

    summary = {
        "trace_steps": int(len(x)),
        "true_divergence": float(x.mean()),
        "cost_ratio_probe_over_step": float(cost_ratio),
        "h2_targets": h2,
        "h2_verdict_at_5pct": h2_verdict,
        "best_estimator": best_est,
        "width_scaling_exponent_beta": beta,
        "coverage_worst_anytime_miss": float(cdf["anytime_miss_rate"].max()),
        "coverage_by_estimator_bursty": cdf[cdf["regime"].str.startswith("bursty")]
            .set_index("estimator")["anytime_miss_rate"].to_dict(),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n=== width vs cost ===")
    print(wdf.pivot_table(index="target_rate", columns="estimator",
                          values="width_mean").to_string())
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
    for est, g in wdf.groupby("estimator"):
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
