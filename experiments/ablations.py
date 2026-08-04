"""Ablation studies (proposal Table 3) — each answers a reviewer objection.

  1. Each detector signal in isolation and in combination.
     -> "Is dropped-mass alone sufficient? Is the rest complexity for its own sake?"
  2. Verification rate swept 0 -> 100%.
     -> Recovers the unverified-sparse and full-dense endpoints; situates the
        work on a continuum instead of a single favourable point.
  3. Fixed-rate vs signal-adaptive sampling at EQUAL probe cost.
     -> Isolates the value of Mechanism 1 to Mechanism 2.
  5. Detector transfer across models and task families.
     -> "Does this need per-deployment calibration?" — an operational cost
        that must be measured, not assumed away.

(Ablation 4, elastic scheduling off, is Study C's "inline" policy.
 Ablation 6, composition with per-layer budgets, is ablation6_layer_budget.py.)

Reads Study A outputs; writes results/ablations/.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from csa.detector import grouped_cv_auc, threshold_at_fpr, transfer_auc
from csa.recording import save_results
from csa.roc import auc_score
from csa.verify import make_estimator

SIGNAL_COLS = [
    "est_dropped_mean_Lmean",
    "est_dropped_max_Lmax",
    "consensus_est_Lmean",
    "fully_dropped_est_Lmean",
    "sparse_margin",
    "sparse_entropy_Lmean",
    "est_dropped_mean_Lstd",
]
# sign that makes "higher = more likely diverged"
SIGN = {c: (-1.0 if c == "sparse_margin" else 1.0) for c in SIGNAL_COLS}


def load(steps_csv: Path, mode: str = "teacher") -> pd.DataFrame:
    df = pd.read_csv(steps_csv)
    df = df[(df["mode"] == mode) & df["top1_flip"].notna()].copy()
    df["y"] = df["top1_flip"].astype(int)
    df["group"] = (df["task_id"].astype(str) + "|" + df["method"].astype(str)
                   + "|" + df["keep_frac"].astype(str))
    for c in SIGNAL_COLS:
        if c not in df:
            df[c] = np.nan
    return df


def signed_matrix(df: pd.DataFrame, cols=None) -> np.ndarray:
    cols = cols or SIGNAL_COLS
    return np.c_[tuple(SIGN[c] * df[c].to_numpy() for c in cols)]


# --------------------------------------------------------------------------
# Ablation 1 — signals in isolation and in combination
# --------------------------------------------------------------------------
def ablation1(df: pd.DataFrame, rows: list):
    y, groups = df["y"].to_numpy(), df["group"].to_numpy()
    for c in SIGNAL_COLS:
        rows.append(dict(ablation="1_signal_isolation", variant=c,
                         auc=auc_score(y, SIGN[c] * df[c].to_numpy()),
                         auc_std=np.nan, n=len(df)))
    # combination, honestly cross-validated by request group
    m, s, _ = grouped_cv_auc(signed_matrix(df), y, groups)
    rows.append(dict(ablation="1_signal_isolation", variant="ALL (grouped 5-fold CV)",
                     auc=m, auc_std=s, n=len(df)))
    # dropped mass alone, same CV protocol, for an apples-to-apples delta
    m1, s1, _ = grouped_cv_auc(signed_matrix(df, ["est_dropped_mean_Lmean"]),
                               y, groups)
    rows.append(dict(ablation="1_signal_isolation",
                     variant="dropped-mass only (grouped CV)",
                     auc=m1, auc_std=s1, n=len(df)))
    # leave-one-signal-out: which signals actually carry weight?
    for c in SIGNAL_COLS:
        rest = [x for x in SIGNAL_COLS if x != c]
        mo, so, _ = grouped_cv_auc(signed_matrix(df, rest), y, groups)
        rows.append(dict(ablation="1_leave_one_out", variant=f"without {c}",
                         auc=mo, auc_std=so, n=len(df)))


# --------------------------------------------------------------------------
# Ablation 2 — verification rate swept 0 -> 100%
# --------------------------------------------------------------------------
def ablation2(df: pd.DataFrame, rows: list):
    """At rate 0 the bound is vacuous (unverified sparse = status quo); at
    rate 1 every step is dense-checked (bound collapses to the exact value,
    and the sparsity win is gone). Everything between is the contribution."""
    y = df["y"].to_numpy()
    sig = SIGN["est_dropped_mean_Lmean"] * df["est_dropped_mean_Lmean"].to_numpy()
    truth = float(y.mean())
    for rate in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        if rate == 0.0:
            rows.append(dict(ablation="2_verification_rate", variant="rate=0",
                             rate=0.0, width=1.0, lo=0.0, hi=1.0,
                             probe_frac=0.0, truth=truth,
                             note="unverified sparse (status quo)"))
            continue
        ws, los, his, pf = [], [], [], []
        for seed in range(12):
            v = make_estimator("eb", 0.05, p=rate, seed=seed)
            for yi, si in zip(y, sig):
                v.step(int(yi), float(si))
            r = v.report()
            ws.append(r.width); los.append(r.lo); his.append(r.hi)
            pf.append(r.n_probes / max(r.n_steps, 1))
        rows.append(dict(ablation="2_verification_rate", variant=f"rate={rate:g}",
                         rate=rate, width=float(np.mean(ws)),
                         lo=float(np.mean(los)), hi=float(np.mean(his)),
                         probe_frac=float(np.mean(pf)), truth=truth,
                         note="full dense verification" if rate == 1.0 else ""))


# --------------------------------------------------------------------------
# Ablation 3 — fixed-rate vs adaptive sampling at equal cost
# --------------------------------------------------------------------------
def ablation3(df: pd.DataFrame, rows: list):
    """Fixed vs adaptive, compared on the ACTUAL probe rate (equal cost).

    Two adaptive variants, because the comparison is easy to rig by accident:
    Horvitz-Thompson weights are bounded by 1/p_min, so the bound width carries
    a 1/p_min range factor. An adaptive sampler with a lower floor is penalised
    by that factor regardless of whether its allocation is smarter.

      adaptive(floor=p)   - same floor as fixed, so the range factor is
                            identical and any difference is pure allocation.
      adaptive(floor=p/4) - aggressive reallocation; isolates how much the
                            range inflation costs in practice.
    """
    y = df["y"].to_numpy()
    sig = SIGN["est_dropped_mean_Lmean"] * df["est_dropped_mean_Lmean"].to_numpy()
    truth = float(y.mean())
    variants = [("fixed", False, None),
                ("adaptive(floor=p)", True, 1.0),
                ("adaptive(floor=p/4)", True, 0.25)]
    for rate in [0.02, 0.05, 0.1, 0.2]:
        for label, adaptive, floor_mult in variants:
            ws, cov, pf = [], [], []
            for seed in range(20):
                v = make_estimator("eb", 0.05, p=rate, seed=seed,
                                   adaptive=adaptive,
                                   p_min=(rate * floor_mult) if floor_mult else None)
                for yi, si in zip(y, sig):
                    v.step(int(yi), float(si))
                r = v.report()
                ws.append(r.width)
                cov.append(r.lo - 1e-9 <= truth <= r.hi + 1e-9)
                pf.append(r.n_probes / max(r.n_steps, 1))
            rows.append(dict(ablation="3_fixed_vs_adaptive", variant=label,
                             rate=rate, width=float(np.mean(ws)),
                             width_std=float(np.std(ws)),
                             coverage=float(np.mean(cov)),
                             probe_frac=float(np.mean(pf)), truth=truth))


# --------------------------------------------------------------------------
# Ablation 5 — detector transfer across models, families, methods, budgets
# --------------------------------------------------------------------------
def ablation5(frames: dict[str, pd.DataFrame], rows: list):
    # (a) across models
    names = list(frames)
    for src in names:
        for dst in names:
            a = transfer_auc(signed_matrix(frames[src]), frames[src]["y"],
                             signed_matrix(frames[dst]), frames[dst]["y"])
            rows.append(dict(ablation="5_transfer_model", train=src, test=dst,
                             auc=a, n_train=len(frames[src]), n_test=len(frames[dst])))
    # (b) across task families, and (c) across sparse methods, within each model
    for name, df in frames.items():
        for key, tag in (("family", "5_transfer_family"), ("method", "5_transfer_method")):
            if key not in df:
                continue
            for src, gs in df.groupby(key):
                for dst, gd in df.groupby(key):
                    a = transfer_auc(signed_matrix(gs), gs["y"],
                                     signed_matrix(gd), gd["y"])
                    rows.append(dict(ablation=tag, model=name, train=str(src),
                                     test=str(dst), auc=a,
                                     n_train=len(gs), n_test=len(gd)))
        # (d) across budgets: the operationally awkward case, since a serving
        # system may change its budget at runtime
        for src, gs in df.groupby("keep_frac"):
            for dst, gd in df.groupby("keep_frac"):
                a = transfer_auc(signed_matrix(gs), gs["y"],
                                 signed_matrix(gd), gd["y"])
                rows.append(dict(ablation="5_transfer_budget", model=name,
                                 train=f"{src:g}", test=f"{dst:g}", auc=a,
                                 n_train=len(gs), n_test=len(gd)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["0.5b=results/study_a_0.5b/steps.csv",
                             "1.5b=results/study_a_1.5b/steps.csv"],
                    help="name=path/to/steps.csv")
    ap.add_argument("--out", default="results/ablations")
    args = ap.parse_args()
    out = Path(args.out)

    frames = {}
    for spec in args.runs:
        name, _, path = spec.partition("=")
        p = Path(path)
        if p.exists():
            frames[name] = load(p)
            print(f"loaded {name}: {len(frames[name])} teacher-forced steps, "
                  f"flip rate {frames[name]['y'].mean():.3f}")
        else:
            print(f"skipping {name}: {p} not found")
    if not frames:
        raise SystemExit("no Study A runs found")

    primary_name = list(frames)[-1]  # largest model available
    primary = frames[primary_name]
    rows: list[dict] = []
    ablation1(primary, rows)
    ablation2(primary, rows)
    ablation3(primary, rows)
    ablation5(frames, rows)

    # operating point for the combined detector on the primary run
    m, s, oof = grouped_cv_auc(signed_matrix(primary), primary["y"].to_numpy(),
                               primary["group"].to_numpy())
    thr, tpr, fpr = threshold_at_fpr(primary["y"].to_numpy(), oof, 0.10)
    summary = {
        "primary_run": primary_name,
        "combined_detector_cv_auc": m,
        "combined_detector_cv_auc_std": s,
        "operating_point_at_10pct_fpr": {"threshold": thr, "tpr": tpr, "fpr": fpr},
        "runs": {k: {"steps": int(len(v)), "flip_rate": float(v["y"].mean())}
                 for k, v in frames.items()},
    }

    save_results(rows, {"runs": args.runs}, out, "ablations")
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    df = pd.DataFrame(rows)
    for ab in df["ablation"].unique():
        print(f"\n=== {ab} ===")
        sub = df[df["ablation"] == ab].dropna(axis=1, how="all")
        print(sub.to_string(index=False, max_rows=40))
    print("\n" + json.dumps(summary, indent=2, default=float))

    _figures(df, out)


def _figures(df: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Ablation 2: bound width vs verification rate
    a2 = df[df["ablation"] == "2_verification_rate"].sort_values("rate")
    if len(a2):
        plt.figure(figsize=(6.5, 4.4))
        plt.plot(a2["rate"] * 100, a2["width"], "o-")
        plt.axhline(a2["truth"].iloc[0], ls=":", c="gray",
                    label=f"true divergence = {a2['truth'].iloc[0]:.3f}")
        plt.xlabel("verification rate (% of decode steps dense-probed)")
        plt.ylabel("95% bound width")
        plt.title("Ablation 2: unverified (0%) to fully verified (100%)")
        plt.xscale("symlog", linthresh=1)
        plt.legend(fontsize=8)
        plt.tight_layout(); plt.savefig(fig_dir / "ablation2_rate.png", dpi=140)
        plt.close()

    # Ablation 3: fixed vs adaptive at equal cost
    a3 = df[df["ablation"] == "3_fixed_vs_adaptive"]
    if len(a3):
        plt.figure(figsize=(6.5, 4.4))
        for var, g in a3.groupby("variant"):
            g = g.sort_values("probe_frac")
            plt.errorbar(g["probe_frac"] * 100, g["width"], yerr=g["width_std"],
                         fmt="o-", capsize=3, label=var)
        plt.xlabel("actual probes per 100 decode steps")
        plt.ylabel("95% bound width")
        plt.title("Ablation 3: does Mechanism 1 help Mechanism 2?")
        plt.legend(fontsize=9)
        plt.tight_layout(); plt.savefig(fig_dir / "ablation3_adaptive.png", dpi=140)
        plt.close()

    # Ablation 5: transfer heatmaps
    for tag, fname in (("5_transfer_family", "ablation5_family"),
                       ("5_transfer_budget", "ablation5_budget")):
        a5 = df[df["ablation"] == tag]
        if not len(a5):
            continue
        model = a5["model"].iloc[-1]
        a5 = a5[a5["model"] == model]
        piv = a5.pivot_table(index="train", columns="test", values="auc")
        plt.figure(figsize=(1.1 * len(piv) + 3, 1.0 * len(piv) + 2.4))
        plt.imshow(piv.values, vmin=0.4, vmax=1.0, cmap="viridis")
        plt.colorbar(label="AUC on test domain")
        plt.xticks(range(len(piv.columns)), piv.columns, rotation=45, ha="right")
        plt.yticks(range(len(piv.index)), piv.index)
        for i in range(len(piv.index)):
            for j in range(len(piv.columns)):
                v = piv.values[i, j]
                if np.isfinite(v):
                    plt.text(j, i, f"{v:.2f}", ha="center", va="center",
                             color="w" if v < 0.8 else "k", fontsize=8)
        plt.xlabel("tested on"); plt.ylabel("trained on")
        plt.title(f"Ablation 5: detector transfer ({tag.split('_')[-1]}, {model})")
        plt.tight_layout(); plt.savefig(fig_dir / f"{fname}.png", dpi=140)
        plt.close()


if __name__ == "__main__":
    main()
