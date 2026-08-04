"""Ablation 6 — composition with PSA-style per-layer budgets.

Objection answered: "this is just adaptive sparsity again."
Response: the contribution is a guarantee, not a Pareto position — and it
should sit ON TOP of an existing budget allocator rather than compete with it.
This ablation runs the paired harness with three per-layer budget schedules at
a MATCHED average budget (so total KV work is equal) and asks:

  1. Do the schedules actually differ in fidelity? (If not, the substrate is
     inert here and the composition claim is untested rather than supported.)
  2. Does the label-free detector still work on top of a non-uniform
     allocator — i.e. is Mechanism 1 orthogonal to the budget policy?

A detector whose AUC survives schedule changes converts competing allocators
into substrates. A detector that needs re-calibration per schedule is an
operational cost that must be reported, not hidden.

Outputs under results/ablation6/.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from csa.detector import grouped_cv_auc, transfer_auc
from csa.paired import PairedModel
from csa.recording import save_results
from csa.roc import auc_score
from csa.sparse import SparseConfig, layer_keep_fracs
from csa.tasks import make_tasks

SIGNALS = ["est_dropped_mean_Lmean", "est_dropped_max_Lmax",
           "consensus_est_Lmean", "fully_dropped_est_Lmean",
           "sparse_margin", "sparse_entropy_Lmean", "est_dropped_mean_Lstd"]
SIGN = {c: (-1.0 if c == "sparse_margin" else 1.0) for c in SIGNALS}
SCHEDULES = ["uniform", "pyramid", "inv_pyramid"]


def signed(df, cols=SIGNALS):
    return np.c_[tuple(SIGN[c] * df[c].fillna(0.0).to_numpy() for c in cols)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.125, 0.0625])
    ap.add_argument("--per-family", type=int, default=2)
    ap.add_argument("--out", default="results/ablation6")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pm = PairedModel(args.model, device=device)
    n_layers = pm.model.config.num_hidden_layers
    count_tokens = lambda s: len(pm.tokenizer(s).input_ids)
    tasks = make_tasks(count_tokens, args.context, per_family=args.per_family,
                       families=["multi_entity", "multi_hop", "coreference",
                                 "reasoning"],
                       seed=23)
    print(f"{len(tasks)} tasks, {n_layers} layers, schedules={SCHEDULES}")

    # verify the schedules are genuinely budget-matched before spending GPU time
    for kf in args.budgets:
        for sch in SCHEDULES:
            fr = layer_keep_fracs(kf, n_layers, sch)
            assert abs(float(np.mean(fr)) - kf) < 1e-6, (sch, kf, np.mean(fr))
            print(f"  keep={kf:<8g} {sch:<12s} per-layer "
                  f"[{fr.min():.4f} .. {fr.max():.4f}] mean={fr.mean():.4f}")

    step_rows = []
    for ti, task in enumerate(tasks):
        ids = pm.encode_chat(task.prompt)
        dense_toks, dense_text = pm.generate_dense(ids, 24)
        if not dense_toks:
            continue
        for kf in args.budgets:
            for sch in SCHEDULES:
                cfg = SparseConfig(method="quest_topk", keep_frac=kf,
                                   layer_schedule=sch)
                r = pm.generate_paired(ids, len(dense_toks), cfg,
                                       teacher_tokens=dense_toks)
                meta = dict(task_id=task.task_id, family=task.family,
                            keep_frac=kf, schedule=sch,
                            dense_correct=task.check(dense_text) if task.gold else None)
                for row in r.rows:
                    step_rows.append({**row, **meta})
        print(f"  [{ti + 1}/{len(tasks)}] {task.task_id}")

    out = Path(args.out)
    save_results(step_rows, {"model": args.model, "schedules": SCHEDULES,
                             "n_layers": n_layers}, out, "steps")
    df = pd.DataFrame(step_rows)
    df["y"] = df["top1_flip"].astype(int)
    df["group"] = df["task_id"].astype(str) + "|" + df["keep_frac"].astype(str)

    # 1. do the schedules differ in fidelity at matched budget?
    fidelity = (df.groupby(["keep_frac", "schedule"])
                .agg(flip_rate=("y", "mean"),
                     oracle_dropped=("oracle_dropped_mean_Lmean", "mean"),
                     out_cos=("out_cos_mean_Lmean", "mean"),
                     n=("y", "size")).reset_index())

    # 2. does the detector survive the substrate change?
    within, transfer = {}, {}
    for sch, g in df.groupby("schedule"):
        if g["y"].nunique() < 2:
            continue
        m, s, _ = grouped_cv_auc(signed(g), g["y"].to_numpy(), g["group"].to_numpy())
        within[sch] = {"cv_auc": m, "cv_auc_std": s,
                       "dropped_mass_auc": auc_score(
                           g["y"].to_numpy(), g["est_dropped_mean_Lmean"].to_numpy()),
                       "n": int(len(g))}
    for src, gs in df.groupby("schedule"):
        for dst, gd in df.groupby("schedule"):
            transfer[f"{src}->{dst}"] = transfer_auc(
                signed(gs), gs["y"], signed(gd), gd["y"])

    summary = {"n_layers": int(n_layers), "model": args.model,
               "fidelity_by_schedule": fidelity.to_dict("records"),
               "detector_within_schedule": within,
               "detector_transfer_across_schedules": transfer}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\n" + fidelity.to_string(index=False))
    print("\n" + json.dumps({"within": within, "transfer": transfer},
                            indent=2, default=float))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for sch, g in fidelity.groupby("schedule"):
        g = g.sort_values("keep_frac")
        axes[0].plot(g["keep_frac"], g["flip_rate"], "o-", label=sch)
        axes[1].plot(g["keep_frac"], g["oracle_dropped"], "o-", label=sch)
    axes[0].set(xlabel="mean keep fraction", ylabel="greedy-flip rate",
                title="Fidelity at matched average budget", xscale="log")
    axes[1].set(xlabel="mean keep fraction", ylabel="oracle dropped mass",
                title="Attention mass discarded", xscale="log")
    for a in axes:
        a.invert_xaxis(); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(fig_dir / "layer_budget.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    main()
