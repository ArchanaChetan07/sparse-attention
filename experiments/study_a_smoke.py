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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from csa.analysis import LONG_DECODE, analyze
from csa.paired import PairedModel
from csa.recording import save_results
from csa.sparse import SparseConfig
from csa.tasks import make_tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="results/study_a")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--per-family", type=int, default=None)
    ap.add_argument("--max-new-long", type=int, default=64,
                    help="decode steps for long-trace families (reasoning/longform)")
    args = ap.parse_args()

    contexts = [1024] if args.quick else [1024, 2048]
    budgets = [0.25, 0.0625] if args.quick else [0.5, 0.25, 0.125, 0.0625, 0.03125]
    methods = ["quest_topk"] if args.quick else ["quest_topk", "mean_topk", "local_sink"]
    per_family = args.per_family or (2 if args.quick else 3)
    max_new_qa, max_new_lf = 16, args.max_new_long

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
        max_new = max_new_lf if task.family in LONG_DECODE else max_new_qa
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

                def _nanmean(key):
                    # local_sink computes no block scores, so its label-free
                    # columns are legitimately all-NaN; return NaN quietly
                    # rather than warning on an empty slice.
                    vals = [x[key] for x in r.rows
                            if x.get(key) is not None and np.isfinite(x.get(key, np.nan))]
                    return float(np.mean(vals)) if vals else np.nan

                req_rows.append({**meta,
                                 "correct": correct,
                                 "dense_correct": dense_correct,
                                 "n_steps": len(r.rows),
                                 "flip_frac": float(np.mean(flips)) if flips else np.nan,
                                 "mean_kl": _nanmean("logit_kl"),
                                 "mean_est_dropped": _nanmean("est_dropped_mean_Lmean"),
                                 "mean_oracle_dropped": _nanmean("oracle_dropped_mean_Lmean"),
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


if __name__ == "__main__":
    main()
