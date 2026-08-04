"""Does KV drift contaminate the teacher-forced detection result?

Teacher forcing fixes the token trajectory, but the KV cache is still written
by sparse-attention hidden states, so divergence at step t reflects both this
step's attention approximation AND accumulated state drift. `dense_state=True`
re-advances the cache densely after each measurement, isolating the former.

If the detector's AUC survives that isolation, H1 is about the mechanism the
proposal claims (omitted attention mass predicts divergence) rather than about
drift accumulation. Small, cheap, and directly answers a reviewer question.

Outputs results/kv_drift/.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from csa.paired import PairedModel
from csa.recording import save_results
from csa.roc import auc_score
from csa.sparse import SparseConfig
from csa.tasks import make_tasks

SIGNALS = {"est_dropped_mean_Lmean": +1, "consensus_est_Lmean": +1,
           "fully_dropped_est_Lmean": +1, "sparse_margin": -1,
           "oracle_dropped_mean_Lmean": +1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--budgets", type=float, nargs="+",
                    default=[0.25, 0.125, 0.0625])
    ap.add_argument("--per-family", type=int, default=2)
    ap.add_argument("--out", default="results/kv_drift")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pm = PairedModel(args.model, device=device)
    count = lambda s: len(pm.tokenizer(s).input_ids)
    tasks = make_tasks(count, args.context, per_family=args.per_family,
                       families=["multi_entity", "multi_hop", "coreference"],
                       seed=31)

    rows = []
    for ti, t in enumerate(tasks):
        ids = pm.encode_chat(t.prompt)
        dense_toks, _ = pm.generate_dense(ids, 16)
        if not dense_toks:
            continue
        for kf in args.budgets:
            cfg = SparseConfig(method="quest_topk", keep_frac=kf)
            for label, ds in (("drifting_kv", False), ("isolated_kv", True)):
                r = pm.generate_paired(ids, len(dense_toks), cfg,
                                       teacher_tokens=dense_toks, dense_state=ds)
                for row in r.rows:
                    rows.append({**row, "task_id": t.task_id, "family": t.family,
                                 "keep_frac": kf, "regime": label})
        print(f"  [{ti + 1}/{len(tasks)}] {t.task_id}")

    out = Path(args.out)
    save_results(rows, {"model": args.model}, out, "steps")
    df = pd.DataFrame(rows)

    summary = {"model": args.model, "n_steps": int(len(df)), "by_regime": {}}
    for regime, g in df.groupby("regime"):
        lab = g["top1_flip"].astype(int).to_numpy()
        block = {"n": int(len(g)), "flip_rate": float(lab.mean()),
                 "auc": {}}
        if 0 < lab.mean() < 1:
            for col, sign in SIGNALS.items():
                if col in g and g[col].notna().any():
                    block["auc"][col] = auc_score(lab, sign * g[col].to_numpy())
        summary["by_regime"][regime] = block

    a = summary["by_regime"].get("drifting_kv", {}).get("auc", {})
    b = summary["by_regime"].get("isolated_kv", {}).get("auc", {})
    summary["auc_delta_isolated_minus_drifting"] = {
        k: (b[k] - a[k]) for k in a if k in b}
    summary["interpretation"] = (
        "A near-zero delta means the teacher-forced detection result is not an "
        "artifact of accumulated KV drift; a large negative delta would mean "
        "the detector was partly reading drift rather than this step's omitted "
        "attention mass.")

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
