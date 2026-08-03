"""Study A (smoke scale) — kill-or-confirm measurement for H1/H4.

Paired dense/sparse execution over synthetic long-context tasks:
  - free-running sparse (production-like; gives end-task accuracy for H4)
  - teacher-forced on the dense trajectory (clean per-step attribution)
Every decode step gets a dense probe on identical cache state, so per-step
divergence labels (greedy-token flip, logit KL) are exact ground truth.

Outputs under results/study_a/:
  steps.csv      one row per decode step (signals + labels + timings)
  requests.csv   one row per (task, config) run (accuracy, aggregates)
  summary.json   AUCs, cliff table, H4 correlations
  figures/*.png
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from csa.paired import PairedModel
from csa.recording import machine_fingerprint, save_results
from csa.roc import auc_score, roc_curve, spearman
from csa.sparse import SparseConfig
from csa.tasks import make_tasks

QA_FAMILIES = ("multi_entity", "multi_hop", "coreference")

SIGNALS = {  # column -> (pretty name, higher-predicts-divergence sign)
    "est_dropped_mean_Lmean": ("est dropped mass (mean)", +1),
    "est_dropped_max_Lmax": ("est dropped mass (max)", +1),
    "consensus_est_Lmean": ("eviction consensus", +1),
    "fully_dropped_est_Lmean": ("fully-dropped mass", +1),
    "sparse_entropy_Lmean": ("sparse attn entropy", +1),
    "sparse_margin": ("top1-top2 margin", -1),
    "est_dropped_mean_Lstd": ("cross-layer dropped-mass std", +1),
}
ORACLES = {
    "oracle_dropped_mean_Lmean": ("ORACLE dropped mass", +1),
    "out_cos_mean_Lmean": ("ORACLE output cos-dist", +1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="results/study_a")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    contexts = [1024] if args.quick else [1024, 2048]
    budgets = [0.25, 0.0625] if args.quick else [0.5, 0.25, 0.125, 0.0625, 0.03125]
    methods = ["quest_topk"] if args.quick else ["quest_topk", "mean_topk", "local_sink"]
    per_family = 2 if args.quick else 3
    max_new_qa, max_new_lf = 16, 48

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pm = PairedModel(args.model, device=device)
    count_tokens = lambda s: len(pm.tokenizer(s).input_ids)

    tasks = []
    for ctx in contexts:
        tasks += [(ctx, t) for t in make_tasks(count_tokens, ctx,
                                               per_family=per_family, seed=17)]
    print(f"{len(tasks)} tasks x {len(methods)} methods x {len(budgets)} budgets "
          f"(+ teacher-forced quest runs)")

    step_rows, req_rows = [], []
    t_start = time.time()

    for ti, (ctx, task) in enumerate(tasks):
        ids = pm.encode_chat(task.prompt)
        max_new = max_new_lf if task.family == "longform" else max_new_qa
        dense_toks, dense_text = pm.generate_dense(ids, max_new)
        dense_correct = task.check(dense_text) if task.gold else None
        req_rows.append(dict(task_id=task.task_id, family=task.family, ctx=ctx,
                             method="dense", keep_frac=1.0, mode="free",
                             correct=dense_correct, n_steps=len(dense_toks),
                             text=dense_text[:120]))

        for method in methods:
            for keep in budgets:
                cfg = SparseConfig(method=method, keep_frac=keep)
                # --- free-running (production-like) ---
                r = pm.generate_paired(ids, max_new, cfg)
                meta = dict(task_id=task.task_id, family=task.family, ctx=ctx,
                            prompt_tokens=int(ids.shape[1]), method=method,
                            keep_frac=keep, mode="free")
                for row in r.rows:
                    step_rows.append({**row, **meta})
                correct = task.check(r.text) if task.gold else None
                flips = [x.get("top1_flip", False) for x in r.rows]
                kls = [x.get("logit_kl", np.nan) for x in r.rows]
                req_rows.append({**meta,
                                 "correct": correct,
                                 "n_steps": len(r.rows),
                                 "flip_frac": float(np.mean(flips)) if flips else np.nan,
                                 "mean_kl": float(np.nanmean(kls)) if kls else np.nan,
                                 "mean_est_dropped": float(np.nanmean(
                                     [x.get("est_dropped_mean_Lmean", np.nan) for x in r.rows])),
                                 "mean_oracle_dropped": float(np.nanmean(
                                     [x.get("oracle_dropped_mean_Lmean", np.nan) for x in r.rows])),
                                 "text": r.text[:120]})
                # --- teacher-forced on dense trajectory (quest only) ---
                if method == "quest_topk" and dense_toks:
                    rt = pm.generate_paired(ids, len(dense_toks), cfg,
                                            teacher_tokens=dense_toks)
                    tmeta = {**meta, "mode": "teacher"}
                    for row in rt.rows:
                        step_rows.append({**row, **tmeta})
        el = time.time() - t_start
        print(f"  [{ti + 1}/{len(tasks)}] {task.task_id} done ({el:.0f}s elapsed)")

    out = Path(args.out)
    meta = dict(model=args.model, contexts=contexts, budgets=budgets,
                methods=methods, per_family=per_family)
    save_results(step_rows, meta, out, "steps")
    save_results(req_rows, meta, out, "requests")

    analyze(pd.DataFrame(step_rows), pd.DataFrame(req_rows), out)


def analyze(steps: pd.DataFrame, reqs: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary = {"fingerprint": machine_fingerprint()}

    # ---------------- H1: per-signal ROC/AUC (teacher-forced rows) ----------
    teach = steps[(steps["mode"] == "teacher") & steps["top1_flip"].notna()]
    labels = teach["top1_flip"].astype(int).to_numpy()
    aucs = {}
    for col, (name, sign) in {**SIGNALS, **ORACLES}.items():
        if col in teach:
            aucs[name] = auc_score(labels, sign * teach[col].to_numpy())
    summary["auc_teacher_flip"] = aucs
    summary["teacher_steps"] = int(len(teach))
    summary["teacher_flip_rate"] = float(labels.mean()) if len(labels) else None

    plt.figure(figsize=(7, 5.5))
    for col, (name, sign) in {**SIGNALS, **ORACLES}.items():
        if col not in teach:
            continue
        fpr, tpr, auc = roc_curve(labels, sign * teach[col].to_numpy())
        style = "--" if name.startswith("ORACLE") else "-"
        plt.plot(fpr, tpr, style, lw=1.6, label=f"{name} ({auc:.2f})")
    plt.plot([0, 1], [0, 1], ":k", lw=1)
    plt.xlabel("false positive rate"); plt.ylabel("true positive rate")
    plt.title("Per-step divergence detection (label = greedy-token flip)\n"
              "teacher-forced, quest_topk, all budgets pooled")
    plt.legend(fontsize=7.5, loc="lower right")
    plt.tight_layout(); plt.savefig(fig_dir / "roc.png", dpi=140); plt.close()

    # ---------------- cliff: accuracy & divergence vs budget ----------------
    qa = reqs[reqs["family"].isin(QA_FAMILIES) & (reqs["method"] != "dense")]
    dense_qa = reqs[(reqs["method"] == "dense") & reqs["family"].isin(QA_FAMILIES)]
    dense_acc = dense_qa["correct"].mean()
    summary["dense_qa_accuracy_by_family"] = {
        k: float(v) for k, v in dense_qa.groupby("family")["correct"].mean().items()}
    cliff = (qa.groupby(["method", "keep_frac"])
             .agg(acc=("correct", "mean"), flip=("flip_frac", "mean"),
                  odrop=("mean_oracle_dropped", "mean"), n=("correct", "size"))
             .reset_index())
    summary["dense_qa_accuracy"] = float(dense_acc)
    summary["cliff"] = cliff.to_dict("records")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for m, g in cliff.groupby("method"):
        g = g.sort_values("keep_frac")
        axes[0].plot(g["keep_frac"], g["acc"], "o-", label=m)
        axes[1].plot(g["keep_frac"], g["flip"], "o-", label=m)
    axes[0].axhline(dense_acc, ls="--", c="k", lw=1, label="dense")
    axes[0].set(xscale="log", xlabel="keep fraction (block budget)",
                ylabel="task accuracy", title="End-task accuracy vs budget")
    axes[1].set(xscale="log", xlabel="keep fraction (block budget)",
                ylabel="greedy-flip fraction per request",
                title="Per-step divergence vs budget")
    for ax in axes:
        ax.invert_xaxis(); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(fig_dir / "cliff.png", dpi=140); plt.close()

    # ---------------- H4: divergence vs end-task correctness ----------------
    qa_ok = qa[qa["correct"].notna()]
    h4 = {
        "spearman_flipfrac_correct": spearman(qa_ok["flip_frac"],
                                              qa_ok["correct"].astype(float)),
        "spearman_meankl_correct": spearman(qa_ok["mean_kl"],
                                            qa_ok["correct"].astype(float)),
        "spearman_estdrop_correct": spearman(qa_ok["mean_est_dropped"],
                                             qa_ok["correct"].astype(float)),
        "auc_flipfrac_incorrect": auc_score(
            1 - qa_ok["correct"].astype(int).to_numpy(),
            qa_ok["flip_frac"].to_numpy()),
        "n_requests": int(len(qa_ok)),
    }
    summary["h4"] = h4

    # ---------------- calibration of the best signal ------------------------
    from csa.roc import calibration_bins
    if "est_dropped_mean_Lmean" in teach:
        c, r, n = calibration_bins(labels, teach["est_dropped_mean_Lmean"].to_numpy())
        plt.figure(figsize=(5.5, 4.2))
        plt.plot(c, r, "o-")
        plt.xlabel("estimated dropped mass (label-free)")
        plt.ylabel("empirical flip rate")
        plt.title("Detector calibration (teacher-forced steps)")
        plt.tight_layout(); plt.savefig(fig_dir / "calibration.png", dpi=140); plt.close()

    # ---------------- example divergence trace ------------------------------
    ex = steps[(steps["mode"] == "teacher") & (steps["keep_frac"] <= 0.0626)]
    if len(ex):
        tid = ex["task_id"].iloc[0]
        exr = ex[ex["task_id"] == tid].sort_values("step")
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(exr["step"], exr["est_dropped_mean_Lmean"], "-o", ms=3,
                 label="est dropped mass")
        ax1.plot(exr["step"], exr["oracle_dropped_mean_Lmean"], "-s", ms=3,
                 label="oracle dropped mass")
        ax1.set_xlabel("decode step"); ax1.set_ylabel("dropped attention mass")
        flips = exr[exr["top1_flip"] == True]  # noqa: E712
        for s in flips["step"]:
            ax1.axvline(s, color="crimson", alpha=0.25, lw=3)
        ax1.plot([], [], color="crimson", alpha=0.4, lw=3, label="token flip")
        ax1.legend(fontsize=8)
        ax1.set_title(f"Divergence trace: {tid}, keep={exr['keep_frac'].iloc[0]:g}")
        plt.tight_layout(); plt.savefig(fig_dir / "trace.png", dpi=140); plt.close()

    # ---------------- verification overhead (measured) ----------------------
    probe_ms = steps["probe_s"].astype(float) * 1000
    step_ms = steps["step_s"].astype(float) * 1000
    summary["timing_ms"] = {"probe_mean": float(probe_ms.mean()),
                            "sparse_step_mean": float(step_ms.mean()),
                            "probe_over_step": float(probe_ms.mean() / step_ms.mean())}

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
