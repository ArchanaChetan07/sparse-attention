"""Study C (simulation scale) — verification as elastic work (H3).

Sweeps offered load for three policies (none / inline / elastic) and plots the
central claim: under load, elastic verification keeps decode latency at the
no-verification baseline while the confidence bound widens gracefully;
inline verification instead trades latency. Also compares max-min-tightness
vs value-weighted probe priority for the elastic scheduler (RQ3).

Outputs under results/study_c/: sweep.csv, figures/elastic.png.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from csa.recording import save_results
from csa.scheduler import simulate, sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/study_c")
    ap.add_argument("--ticks", type=int, default=6000)
    args = ap.parse_args()
    out = Path(args.out)

    rates = np.linspace(0.02, 0.115, 9)
    results = sweep(rates=rates, ticks=args.ticks, capacity=8.0,
                    probe_rate=0.1, probe_cost=3.0, seed=3)
    rows = [asdict(r) for r in results]

    # RQ3: elastic priority variants at moderate load with a generous probe
    # budget, where per-request width differences are resolvable
    for vw in (False, True):
        for seed in range(5):
            r = simulate("elastic", 0.08, ticks=args.ticks, capacity=8.0,
                         probe_rate=0.3, probe_cost=3.0, seed=100 + seed,
                         value_weighted=vw)
            d = asdict(r)
            d["policy"] = f"elastic[{'value' if vw else 'maxmin'}]"
            rows.append(d)

    save_results(rows, {"ticks": args.ticks}, out, "sweep")
    df = pd.DataFrame(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    base = df[df["policy"].isin(["none", "inline", "elastic"])]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for pol, g in base.groupby("policy"):
        g = g.sort_values("arrival_rate")
        axes[0].plot(g["utilization"], g["tpot_p99"], "o-", label=pol)
        axes[1].plot(g["utilization"], g["system_width"], "o-", label=pol)
        axes[2].plot(g["utilization"], g["probe_completion"], "o-", label=pol)
    axes[0].set(xlabel="GPU utilization", ylabel="P99 TPOT (ticks/step)",
                title="Latency: elastic tracks the no-verify baseline")
    axes[0].set_yscale("log")
    axes[1].set(xlabel="GPU utilization",
                ylabel="system-level 95% bound width",
                title="Guarantee: elastic widens under load")
    axes[1].set_yscale("log")
    axes[2].set(xlabel="GPU utilization", ylabel="probes executed / drawn",
                title="Verification completion")
    for ax in axes:
        ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(fig_dir / "elastic.png", dpi=140)
    plt.close()

    hi = base[base["arrival_rate"] >= 0.1]
    mid = base[(base["utilization"] >= 0.55) & (base["utilization"] <= 0.9)]
    summary = {
        "high_load": hi.groupby("policy")[["tpot_p99", "system_width",
                                           "probe_completion"]].mean()
        .to_dict("index"),
        "mid_load": mid.groupby("policy")[["tpot_p99", "system_width",
                                           "probe_completion"]].mean()
        .to_dict("index"),
        "rq3_priority": df[df["policy"].str.startswith("elastic[")]
        .groupby("policy")[["mean_width", "p90_width", "system_width"]]
        .mean().to_dict("index"),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
