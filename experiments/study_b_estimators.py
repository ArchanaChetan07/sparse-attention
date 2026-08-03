"""Study B — estimator design: bound width vs verification cost (H2).

Replays Study A step traces (teacher-forced quest_topk rows: exact per-step
divergence labels + label-free signals) through the Mechanism-2 estimators at
a range of probe budgets, then measures the confidence-bound width actually
achieved per unit of verification.

Also runs a synthetic coverage audit: anytime-valid coverage of each CS under
i.i.d. and drifting Bernoulli streams (the phase-transition regime: long quiet
stretches, concentrated bursts).

Outputs under results/study_b/: width_vs_cost.csv, coverage.csv, figures.
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
PROBE_RATES = [0.02, 0.05, 0.1, 0.2, 0.4]
N_SEEDS = 20


def load_stream(steps_csv: Path):
    df = pd.read_csv(steps_csv)
    df = df[(df["mode"] == "teacher") & df["top1_flip"].notna()]
    df = df.sort_values(["task_id", "keep_frac", "step"])
    x = df["top1_flip"].astype(int).to_numpy()
    sig = df["est_dropped_mean_Lmean"].fillna(0.0).to_numpy()
    return x, sig, df


def width_vs_cost(x, sig, out_rows):
    for kind, adaptive, label in ESTIMATORS:
        for p in PROBE_RATES:
            widths, probes, covered = [], [], []
            mu = x.mean()
            for seed in range(N_SEEDS):
                v = make_estimator(kind, 0.05, p=p, seed=seed, adaptive=adaptive)
                for xi, si in zip(x, sig):
                    v.step(int(xi), float(si))
                r = v.report()
                widths.append(r.width)
                probes.append(r.n_probes / r.n_steps)
                covered.append(r.lo - 1e-9 <= mu <= r.hi + 1e-9)
            out_rows.append(dict(
                estimator=label, target_rate=p,
                probe_rate=float(np.mean(probes)),
                width_mean=float(np.mean(widths)),
                width_std=float(np.std(widths)),
                coverage=float(np.mean(covered)),
                true_divergence=float(mu), n_steps=int(len(x))))


def coverage_audit(out_rows):
    """Anytime coverage under iid and bursty (drifting) streams."""
    rng = np.random.default_rng(0)
    T, reps = 1200, 150
    regimes = {
        "iid_rare": lambda: (rng.random(T) < 0.03).astype(int),
        "iid_moderate": lambda: (rng.random(T) < 0.2).astype(int),
        "bursty": lambda: (rng.random(T) < np.where(
            (np.arange(T) // 100) % 3 == 2, 0.35, 0.01)).astype(int),
    }
    for regime, gen in regimes.items():
        for kind, adaptive, label in ESTIMATORS:
            misses = 0
            for rep in range(reps):
                xs = gen()
                v = make_estimator(kind, 0.05, p=0.15, seed=rep, adaptive=adaptive)
                # signal: noisy leading indicator of divergence probability
                sigs = 0.1 + 0.8 * xs + 0.15 * rng.random(T)
                anytime_ok = True
                seen = 0.0
                for t, (xi, si) in enumerate(zip(xs, sigs), 1):
                    v.step(int(xi), float(si))
                    seen += xi
                    mu_t = seen / t
                    if not (v.cs.lo / v.scale - 1e-9 <= mu_t
                            <= min(v.cs.hi / v.scale, 1.0) + 1e-9):
                        anytime_ok = False
                misses += not anytime_ok
            out_rows.append(dict(regime=regime, estimator=label,
                                 anytime_miss_rate=misses / reps,
                                 target_alpha=0.05))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="results/study_a/steps.csv")
    ap.add_argument("--out", default="results/study_b")
    args = ap.parse_args()
    out = Path(args.out)

    x, sig, df = load_stream(Path(args.steps))
    print(f"replaying {len(x)} teacher-forced steps, "
          f"true diverged fraction = {x.mean():.4f}")

    rows_w, rows_c = [], []
    width_vs_cost(x, sig, rows_w)
    coverage_audit(rows_c)
    save_results(rows_w, {"steps": args.steps}, out, "width_vs_cost")
    save_results(rows_c, {}, out, "coverage")

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    wdf = pd.DataFrame(rows_w)
    plt.figure(figsize=(7.5, 5))
    for est, g in wdf.groupby("estimator"):
        g = g.sort_values("probe_rate")
        plt.plot(g["probe_rate"] * 100, g["width_mean"], "o-", label=est)
        plt.fill_between(g["probe_rate"] * 100,
                         g["width_mean"] - g["width_std"],
                         g["width_mean"] + g["width_std"], alpha=0.12)
    plt.axhline(0.10, ls="--", c="k", lw=1, label="±5% target width")
    plt.xlabel("verification cost (dense probes per 100 decode steps)")
    plt.ylabel("95% confidence-bound width on diverged fraction")
    plt.title(f"Bound width vs verification cost "
              f"(replayed Study A trace, μ={x.mean():.3f})")
    plt.legend(fontsize=8)
    plt.xscale("log")
    plt.tight_layout(); plt.savefig(fig_dir / "width_vs_cost.png", dpi=140)
    plt.close()

    summary = {
        "trace_steps": int(len(x)),
        "true_divergence": float(x.mean()),
        "best_width_at_10pct": float(
            wdf[np.isclose(wdf["target_rate"], 0.1)]["width_mean"].min()),
        "coverage_min": float(pd.DataFrame(rows_c)["anytime_miss_rate"].max()),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(rows_c).to_string(index=False))


if __name__ == "__main__":
    main()
