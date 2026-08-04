"""Study A analysis, decoupled from the sweep that produced the data.

Keeping analysis out of the sweep matters here: sweeps are expensive and are
run at different times, but every run must be analysed by the SAME code for
the comparison between them to mean anything. Re-runnable over any
results directory containing steps.csv + requests.csv.

Three methodological points are enforced here rather than left to the reader:

  1. AUC pooled across budgets is inflated. Budget itself predicts divergence,
     and a serving system knows its own budget, so the deployable number is
     the WITHIN-budget AUC.
  2. H4 must condition on headroom. On requests the dense model already gets
     wrong, sparse execution cannot make them more wrong, so those rows
     contribute label noise uncorrelated with divergence.
  3. H4 must also be evaluated WITHIN budget, or a negative correlation can be
     produced entirely by budget acting as a common cause of both divergence
     and error.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .recording import machine_fingerprint
from .roc import auc_score, calibration_bins, roc_curve, spearman

QA_FAMILIES = ("multi_entity", "multi_hop", "coreference", "reasoning")
LONG_DECODE = ("reasoning", "longform")  # families needing long decode traces

SIGNALS = {  # column -> (pretty name, sign making "higher = more diverged")
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
ALL_SIG = {**SIGNALS, **ORACLES}


def _aucs(df: pd.DataFrame, labels: np.ndarray) -> dict:
    return {name: auc_score(labels, sign * df[col].to_numpy())
            for col, (name, sign) in ALL_SIG.items()
            if col in df and df[col].notna().any()}


def _h4_block(d: pd.DataFrame) -> dict:
    if len(d) < 5 or d["correct"].nunique() < 2:
        return {"n_requests": int(len(d)),
                "note": "insufficient variation to estimate"}
    y = d["correct"].astype(float)
    yi = 1 - d["correct"].astype(int).to_numpy()
    out = {
        "spearman_flipfrac_correct": spearman(d["flip_frac"], y),
        "spearman_meankl_correct": spearman(d["mean_kl"], y),
        "spearman_estdrop_correct": spearman(d["mean_est_dropped"], y),
        "auc_flipfrac_incorrect": auc_score(yi, d["flip_frac"].to_numpy()),
        "auc_estdrop_incorrect": auc_score(yi, d["mean_est_dropped"].to_numpy()),
        "n_requests": int(len(d)),
        "accuracy": float(y.mean()),
    }
    return out


def _h4_within_budget(d: pd.DataFrame) -> dict:
    """Budget held fixed, so budget cannot act as a common cause."""
    rs, ns, per = [], [], {}
    for kf, g in d.groupby("keep_frac"):
        if g["correct"].nunique() < 2 or len(g) < 5:
            per[f"keep={kf:g}"] = {"n": int(len(g)), "note": "no variation"}
            continue
        r = spearman(g["flip_frac"], g["correct"].astype(float))
        per[f"keep={kf:g}"] = {
            "n": int(len(g)), "accuracy": float(g["correct"].mean()),
            "spearman": r,
            "auc": auc_score(1 - g["correct"].astype(int).to_numpy(),
                             g["flip_frac"].to_numpy())}
        if np.isfinite(r):
            rs.append(r); ns.append(len(g))
    return {"per_budget": per,
            "weighted_mean_spearman": (float(np.average(rs, weights=ns))
                                       if rs else float("nan")),
            "n_budgets_estimable": len(rs),
            "n_requests": int(sum(ns))}


def analyze(steps: pd.DataFrame, reqs: pd.DataFrame, out: Path,
            make_figures: bool = True) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"fingerprint": machine_fingerprint()}

    # ---------------- H1: per-signal detection ----------------------------
    teach = steps[(steps["mode"] == "teacher") & steps["top1_flip"].notna()]
    labels = teach["top1_flip"].astype(int).to_numpy()
    summary["teacher_steps"] = int(len(teach))
    summary["teacher_flip_rate"] = float(labels.mean()) if len(labels) else None
    summary["auc_teacher_flip"] = _aucs(teach, labels) if len(teach) else {}

    summary["auc_within_budget"] = {}
    for kf, g in teach.groupby("keep_frac"):
        lab = g["top1_flip"].astype(int).to_numpy()
        if 0 < lab.mean() < 1:
            block = _aucs(g, lab)
            block["_n"] = int(len(g))
            block["_flip_rate"] = float(lab.mean())
            summary["auc_within_budget"][f"keep={kf:g}"] = block

    # Is a signal detecting sparse-induced DAMAGE, or merely step-level
    # UNCERTAINTY? A step whose top-1/top-2 margin is small flips easily under
    # any perturbation, so a margin can score well on flip-prediction without
    # carrying information about what sparse attention actually discarded.
    # Correlating each signal against the oracle damage measures separates
    # them: a signal that predicts flips but not damage is an uncertainty
    # detector wearing a fidelity signal's clothes.
    summary["signal_vs_damage"] = {}
    for col, (name, sign) in SIGNALS.items():
        if col not in teach or teach[col].isna().all():
            continue
        s = sign * teach[col]
        entry = {
            "auc_for_flip": summary["auc_teacher_flip"].get(name),
            "rho_with_oracle_dropped_mass": spearman(
                s, teach["oracle_dropped_mean_Lmean"]),
            "rho_with_output_divergence": spearman(
                s, teach["out_cos_mean_Lmean"]),
        }
        # within-budget, so budget cannot drive both
        within = [spearman(sign * g[col], g["oracle_dropped_mean_Lmean"])
                  for _, g in teach.groupby("keep_frac") if len(g) > 10]
        within = [w for w in within if np.isfinite(w)]
        entry["rho_with_damage_within_budget"] = (
            float(np.mean(within)) if within else float("nan"))
        summary["signal_vs_damage"][name] = entry

    free = steps[(steps["mode"] == "free") & steps["top1_flip"].notna()]
    summary["auc_free_by_method"] = {}
    for meth, g in free.groupby("method"):
        lab = g["top1_flip"].astype(int).to_numpy()
        if 0 < lab.mean() < 1:
            block = _aucs(g, lab)
            block["_n"] = int(len(g))
            summary["auc_free_by_method"][meth] = block

    # ---------------- cliff -----------------------------------------------
    if "dense_correct" not in reqs.columns:
        dense = (reqs[reqs["method"] == "dense"][["task_id", "correct"]]
                 .rename(columns={"correct": "dense_correct"}))
        reqs = reqs.merge(dense, on="task_id", how="left")
    qa = reqs[reqs["family"].isin(QA_FAMILIES) & (reqs["method"] != "dense")]
    dense_qa = reqs[(reqs["method"] == "dense")
                    & reqs["family"].isin(QA_FAMILIES)]
    summary["dense_qa_accuracy"] = float(dense_qa["correct"].mean())
    summary["dense_qa_accuracy_by_family"] = {
        k: float(v) for k, v in dense_qa.groupby("family")["correct"].mean().items()}
    cliff = (qa.groupby(["method", "keep_frac"])
             .agg(acc=("correct", "mean"), flip=("flip_frac", "mean"),
                  odrop=("mean_oracle_dropped", "mean"), n=("correct", "size"))
             .reset_index())
    summary["cliff"] = cliff.to_dict("records")

    # Nominal budget != actual budget. Sink and local blocks are always kept,
    # so a nominal keep_frac below that floor is silently clamped upward, and
    # at short contexts two distinct nominal budgets can be the SAME actual
    # budget (which makes the methods collapse to identical selections). The
    # harness records the true kept-token fraction; report it so the cliff is
    # read against what was actually computed.
    if "keep_tokens_frac_Lmean" in steps:
        eff = (steps[steps["keep_tokens_frac_Lmean"].notna()]
               .groupby(["ctx", "keep_frac"])["keep_tokens_frac_Lmean"]
               .mean().reset_index()
               if "ctx" in steps else
               steps.groupby("keep_frac")["keep_tokens_frac_Lmean"]
               .mean().reset_index())
        summary["effective_keep_fraction"] = eff.to_dict("records")
        clamped = [r for r in eff.to_dict("records")
                   if r["keep_tokens_frac_Lmean"] > r["keep_frac"] * 1.25]
        summary["budgets_clamped_by_sink_local_floor"] = clamped

    # ---------------- H4 --------------------------------------------------
    qa_ok = qa[qa["correct"].notna()]
    summary["h4_all_requests"] = _h4_block(qa_ok)
    answerable = qa_ok[qa_ok["dense_correct"] == True]  # noqa: E712
    summary["h4_answerable"] = _h4_block(answerable)
    summary["h4_answerable_within_budget"] = _h4_within_budget(answerable)
    summary["h4_by_family"] = {fam: _h4_block(g)
                               for fam, g in qa_ok.groupby("family")}
    summary["h4_floor_effect"] = {
        "n_answerable": int((qa_ok["dense_correct"] == True).sum()),  # noqa: E712
        "n_dense_already_wrong": int((qa_ok["dense_correct"] == False).sum()),  # noqa: E712
        "sparse_accuracy_when_dense_wrong": float(
            qa_ok[qa_ok["dense_correct"] == False]["correct"].mean())  # noqa: E712
        if (qa_ok["dense_correct"] == False).any() else None,  # noqa: E712
    }

    # ---------------- overhead -------------------------------------------
    probe_ms = steps["probe_s"].astype(float) * 1000
    step_ms = steps["step_s"].astype(float) * 1000
    summary["timing_ms"] = {
        "probe_mean": float(probe_ms.mean()),
        "paired_step_mean": float(step_ms.mean()),
        "probe_over_paired_step": float(probe_ms.mean() / step_ms.mean()),
        "note": ("the paired step computes dense AND sparse for measurement, "
                 "so this ratio understates the cost of a probe relative to a "
                 "production sparse step; see experiments/overhead_bench.py"),
    }

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    if make_figures:
        figures(steps, cliff, teach, labels, out)
    return summary


def figures(steps, cliff, teach, labels, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(out) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if len(teach):
        plt.figure(figsize=(7, 5.5))
        for col, (name, sign) in ALL_SIG.items():
            if col not in teach or teach[col].isna().all():
                continue
            fpr, tpr, auc = roc_curve(labels, sign * teach[col].to_numpy())
            plt.plot(fpr, tpr, "--" if name.startswith("ORACLE") else "-",
                     lw=1.6, label=f"{name} ({auc:.2f})")
        plt.plot([0, 1], [0, 1], ":k", lw=1)
        plt.xlabel("false positive rate"); plt.ylabel("true positive rate")
        plt.title("Per-step divergence detection (label = greedy-token flip)\n"
                  "teacher-forced, all budgets pooled")
        plt.legend(fontsize=7.5, loc="lower right")
        plt.tight_layout(); plt.savefig(fig_dir / "roc.png", dpi=140); plt.close()

        c, r, n = calibration_bins(labels, teach["est_dropped_mean_Lmean"].to_numpy())
        plt.figure(figsize=(5.5, 4.2))
        plt.plot(c, r, "o-")
        plt.xlabel("estimated dropped mass (label-free)")
        plt.ylabel("empirical flip rate")
        plt.title("Detector calibration")
        plt.tight_layout()
        plt.savefig(fig_dir / "calibration.png", dpi=140); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for m, g in cliff.groupby("method"):
        g = g.sort_values("keep_frac")
        axes[0].plot(g["keep_frac"], g["acc"], "o-", label=m)
        axes[1].plot(g["keep_frac"], g["flip"], "o-", label=m)
    axes[0].set(xscale="log", xlabel="keep fraction (block budget)",
                ylabel="task accuracy", title="End-task accuracy vs budget")
    axes[1].set(xscale="log", xlabel="keep fraction (block budget)",
                ylabel="greedy-flip fraction per request",
                title="Per-step divergence vs budget")
    for ax in axes:
        ax.invert_xaxis(); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(fig_dir / "cliff.png", dpi=140); plt.close()

    ex = steps[(steps["mode"] == "teacher") & (steps["keep_frac"] <= 0.0626)]
    if len(ex):
        tid = ex["task_id"].iloc[0]
        exr = ex[ex["task_id"] == tid].sort_values("step")
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(exr["step"], exr["est_dropped_mean_Lmean"], "-o", ms=3,
                 label="est dropped mass (label-free)")
        ax1.plot(exr["step"], exr["oracle_dropped_mean_Lmean"], "-s", ms=3,
                 label="oracle dropped mass")
        for s in exr[exr["top1_flip"] == True]["step"]:  # noqa: E712
            ax1.axvline(s, color="crimson", alpha=0.25, lw=3)
        ax1.plot([], [], color="crimson", alpha=0.4, lw=3, label="token flip")
        ax1.set_xlabel("decode step"); ax1.set_ylabel("dropped attention mass")
        ax1.legend(fontsize=8)
        ax1.set_title(f"Divergence trace: {tid}, keep={exr['keep_frac'].iloc[0]:g}")
        plt.tight_layout(); plt.savefig(fig_dir / "trace.png", dpi=140); plt.close()


def analyze_dir(path: str | Path, make_figures: bool = True) -> dict:
    p = Path(path)
    steps = pd.read_csv(p / "steps.csv")
    reqs = pd.read_csv(p / "requests.csv")
    return analyze(steps, reqs, p, make_figures)
