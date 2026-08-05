"""Emit markdown tables from result JSON/CSV so REPORT.md never hand-copies a
number. Run after all studies; paste or redirect into the report.

    python experiments/make_tables.py > results/TABLES.md
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

R = Path("results")


def load(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def h(title):
    print(f"\n### {title}\n")


def table(rows, cols):
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")


def fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return "n/a" if x != x else f"{x:.{nd}f}"
    return str(x)


def study_a():
    runs = {p.name.replace("study_a_", ""): load(p / "summary.json")
            for p in sorted(R.glob("study_a*"))
            if (p / "summary.json").exists()
            and not (p / "SUPERSEDED.md").exists()}  # provenance discipline
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        return
    h("H1 — label-free detection of per-step divergence (AUC)")
    names = list(runs)
    sig_names = sorted({n for s in runs.values() for n in s["auc_teacher_flip"]})
    rows = []
    for n in sig_names:
        rows.append([("**" + n + "**") if n.startswith("ORACLE") else n]
                    + [fmt(runs[r]["auc_teacher_flip"].get(n)) for r in names])
    table(rows, ["signal"] + names)
    print(f"\nFalsification threshold: best label-free AUC < 0.65 kills Mechanism 1.\n")

    h("H1 — within-budget AUC (deployable number; pooling inflates)")
    for r in names:
        wb = runs[r].get("auc_within_budget", {})
        if not wb:
            continue
        print(f"\n**{r}**\n")
        keys = sorted(wb)
        rows = [[k, wb[k].get("_n"), fmt(wb[k].get("_flip_rate")),
                 fmt(wb[k].get("est dropped mass (mean)")),
                 fmt(wb[k].get("eviction consensus")),
                 fmt(wb[k].get("top1-top2 margin")),
                 fmt(wb[k].get("ORACLE dropped mass"))] for k in keys]
        table(rows, ["budget", "n steps", "flip rate", "est dropped mass",
                     "consensus", "top1-top2 margin", "ORACLE dropped mass"])

    h("Is a signal detecting damage, or just uncertainty?")
    for r in names:
        sv = runs[r].get("signal_vs_damage", {})
        if not sv:
            continue
        print(f"\n**{r}**\n")
        rows = sorted(([k, fmt(v["auc_for_flip"]),
                        fmt(v["rho_with_oracle_dropped_mass"]),
                        fmt(v["rho_with_damage_within_budget"])]
                       for k, v in sv.items()),
                      key=lambda x: -(float(x[1]) if x[1] not in ("—", "n/a") else 0))
        table(rows, ["signal", "AUC (flip)", "rho with damage",
                     "rho with damage, within budget"])
    print("\nA signal with a high AUC but a near-zero damage correlation is an "
          "uncertainty detector, not a fidelity signal: it fires on steps that "
          "would flip under any perturbation and stays quiet on confidently "
          "wrong steps, which is the case that matters.\n")

    h("H4 — does divergence predict end-task wrongness?")
    rows = []
    for r in names:
        s = runs[r]
        a = s.get("h4_all_requests", {})
        b = s.get("h4_answerable", {})
        w = s.get("h4_answerable_within_budget", {})
        fe = s.get("h4_floor_effect", {})
        rows.append([r, fmt(s.get("dense_qa_accuracy")),
                     f"{fe.get('n_answerable')}/{fe.get('n_answerable', 0) + fe.get('n_dense_already_wrong', 0)}",
                     fmt(a.get("spearman_flipfrac_correct")),
                     fmt(b.get("spearman_flipfrac_correct")),
                     fmt(w.get("weighted_mean_spearman")),
                     fmt(b.get("auc_flipfrac_incorrect"))])
    table(rows, ["run", "dense acc", "answerable", "rho (all)",
                 "rho (answerable)", "rho (within budget)", "AUC answerable"])
    print("\nFalsification threshold: |rho| < 0.5 kills H4. "
          "'All' pools requests the dense model already fails, where sparse "
          "cannot degrade anything; 'within budget' rules out budget acting "
          "as a common cause.\n")

    h("Fidelity cliff — accuracy and divergence vs budget")
    for r in names:
        cl = pd.DataFrame(runs[r]["cliff"])
        if not len(cl):
            continue
        print(f"\n**{r}** (dense accuracy {fmt(runs[r]['dense_qa_accuracy'])})\n")
        piv_a = cl.pivot_table(index="keep_frac", columns="method", values="acc")
        piv_f = cl.pivot_table(index="keep_frac", columns="method", values="flip")
        cols = list(piv_a.columns)
        rows = [[f"{kf:g}"] + [fmt(piv_a.loc[kf, c]) for c in cols]
                + [fmt(piv_f.loc[kf, c]) for c in cols]
                for kf in sorted(piv_a.index, reverse=True)]
        table(rows, ["keep frac"] + [f"acc:{c}" for c in cols]
              + [f"flip:{c}" for c in cols])


def study_b():
    s = load(R / "study_b/summary.json")
    if not s:
        return
    h("H2 — bound width vs verification cost (per-budget streams)")
    rows = []
    for stream, targets in s.get("h2_targets_by_stream", {}).items():
        for target, v in targets.items():
            if not v.get("reached"):
                rows.append([stream, target, "not reached", "—", "—", "—"])
            else:
                rows.append([stream, target, v["estimator"],
                             f"{v['probe_rate']*100:.1f}%",
                             f"{v['throughput_loss']*100:.1f}%",
                             fmt(v["coverage"], 2)])
    table(rows, ["stream", "target width", "cheapest VALID estimator",
                 "probes / 100 steps", "throughput loss", "coverage"])
    print(f"\nMeasured probe/step cost ratio r = "
          f"{fmt(s['cost_ratio_probe_over_step'], 2)} "
          f"[{s.get('cost_ratio_source', 'unknown source')}]. "
          f"H2 falsification: a +/-5% bound costing >15% throughput.")
    print(f"\n- Short-trace verdict: **{s['h2_verdict_at_5pct_on_short_traces']}**"
          f"\n- Scale-free verdict: **{s['h2_verdict_scalefree']}**")

    h("H2 — probes needed per target width (scale-free)")
    need = s.get("h2_probes_needed_for_width", {})
    widths = sorted({w for t in need.values() for w in t}, reverse=True)
    table([[est] + [fmt(t.get(w)) for w in widths] for est, t in need.items()],
          ["estimator"] + [f"width {w}" for w in widths])
    print(f"\nCoverage-valid estimators: "
          f"{', '.join(s.get('coverage_valid_estimators', []))} — "
          f"best: {s.get('best_valid_estimator')}. "
          f"Width scaling exponent beta = "
          f"{fmt(s['width_scaling_exponent_beta'], 3)} "
          f"(width ~ p^-beta; 0.5 is the sqrt-n rate).")

    cov = pd.read_csv(R / "study_b/coverage.csv") if (R / "study_b/coverage.csv").exists() else None
    if cov is not None:
        h("Coverage audit — anytime miss rate (target alpha = 0.05)")
        piv = cov.pivot_table(index="estimator", columns="regime",
                              values="anytime_miss_rate")
        rows = [[i] + [fmt(piv.loc[i, c], 3) for c in piv.columns] for i in piv.index]
        table(rows, ["estimator"] + list(piv.columns))


def study_c():
    s = load(R / "study_c/summary.json")
    if not s:
        return
    h("H3 — verification as elastic work")
    for label, block in (("high load", s.get("high_load", {})),
                         ("moderate load", s.get("mid_load", {}))):
        if not block:
            continue
        print(f"\n**{label}**\n")
        rows = [[pol, fmt(v.get("tpot_p99"), 2), fmt(v.get("system_width")),
                 fmt(v.get("probe_completion"))] for pol, v in block.items()]
        table(rows, ["policy", "P99 TPOT (ticks/step)", "bound width",
                     "probes executed / drawn"])


def ablations():
    p = R / "ablations/ablations.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    s = load(R / "ablations/summary.json") or {}

    h("Ablation 1 — signals in isolation vs combination")
    a = df[df["ablation"] == "1_signal_isolation"].sort_values("auc", ascending=False)
    table([[r["variant"], fmt(r["auc"]), fmt(r.get("auc_std"))]
           for _, r in a.iterrows()], ["variant", "AUC", "+/- std (CV)"])

    h("Ablation 2 — verification rate swept 0 to 100%")
    a = df[df["ablation"] == "2_verification_rate"].sort_values("rate")
    table([[r["variant"], fmt(r["width"]), f"[{fmt(r['lo'])}, {fmt(r['hi'])}]",
            r.get("note", "")] for _, r in a.iterrows()],
          ["verification rate", "bound width", "interval", "note"])

    h("Ablation 3 — fixed vs adaptive sampling at equal cost")
    a = df[df["ablation"] == "3_fixed_vs_adaptive"]
    piv = a.pivot_table(index="rate", columns="variant", values="width")
    table([[f"{i:g}"] + [fmt(piv.loc[i, c]) for c in piv.columns] for i in piv.index],
          ["target rate"] + [f"width: {c}" for c in piv.columns])

    for tag, title in (("5_transfer_model", "Ablation 5 — transfer across models"),
                       ("5_transfer_family", "Ablation 5 — transfer across task families"),
                       ("5_transfer_budget", "Ablation 5 — transfer across budgets")):
        a = df[df["ablation"] == tag]
        if not len(a):
            continue
        h(title)
        if "model" in a.columns and a["model"].notna().any():
            a = a[a["model"] == a["model"].dropna().iloc[-1]]
        piv = a.pivot_table(index="train", columns="test", values="auc")
        table([[i] + [fmt(piv.loc[i, c]) for c in piv.columns] for i in piv.index],
              ["train \\ test"] + list(piv.columns))

    if s:
        print(f"\nCombined detector, grouped 5-fold CV AUC: "
              f"{fmt(s.get('combined_detector_cv_auc'))} "
              f"(+/- {fmt(s.get('combined_detector_cv_auc_std'))}). "
              f"Operating point at 10% FPR: TPR "
              f"{fmt((s.get('operating_point_at_10pct_fpr') or {}).get('tpr'))}.")


def ablation6():
    s = load(R / "ablation6/summary.json")
    if not s:
        return
    h("Ablation 6 — composition with per-layer budget schedules")
    f = pd.DataFrame(s["fidelity_by_schedule"])
    piv = f.pivot_table(index="keep_frac", columns="schedule", values="flip_rate")
    table([[f"{i:g}"] + [fmt(piv.loc[i, c]) for c in piv.columns] for i in piv.index],
          ["mean keep frac"] + [f"flip: {c}" for c in piv.columns])
    w = s["detector_within_schedule"]
    table([[k, fmt(v["cv_auc"]), fmt(v["dropped_mass_auc"]), v["n"]]
           for k, v in w.items()],
          ["schedule", "combined CV AUC", "dropped-mass AUC", "n steps"])
    t = s["detector_transfer_across_schedules"]
    table([[k, fmt(v)] for k, v in t.items()], ["train -> test", "AUC"])


def overhead():
    s = load(R / "overhead/overhead.json")
    if not s:
        return
    h("Measured verification overhead")
    table([[f"{r['keep_frac']:g}", fmt(r["dense_ms"], 1), fmt(r["sparse_only_ms"], 1),
            fmt(r["speedup_vs_dense"], 2), fmt(r["cost_ratio_r"], 2),
            f"{r['loss_at_5pct']*100:.1f}%"] for r in s["rows"]],
          ["keep frac", "dense ms", "sparse ms", "speedup", "r = probe/step",
           "throughput loss @5% probes"])
    print(f"\n_{s['caveat']}_")


if __name__ == "__main__":
    print("<!-- generated by experiments/make_tables.py — do not hand-edit -->")
    print("# Result tables")
    study_a(); study_b(); study_c(); ablations(); ablation6(); overhead()
