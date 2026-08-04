"""Measured verification overhead: what does a dense probe actually cost?

H2's falsification criterion is stated in throughput, so the probe/step cost
ratio r is the load-bearing number. Study A's r is measured against the
*paired* step (which computes dense AND sparse for measurement) and therefore
understates r. This benchmark measures the three costs separately:

    dense_step   - ordinary dense decode step (SDPA fast path)
    sparse_step  - production sparse decode step: block selection, then
                   attention over gathered selected blocks only
    probe        - one dense step re-executed on retained KV state

and reports r = probe / sparse_step, plus the sparsity speedup itself.

HONEST CAVEAT, stated in the output: on a small model and a small GPU, decode
is dominated by kernel-launch and MLP time rather than KV-attention bandwidth,
so the measured sparsity speedup here is a LOWER bound and r is an UPPER
bound. The regime where sparse attention pays (8B+ at 32K-128K on a
bandwidth-bound card) is the rented-GPU phase of the programme.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from csa import paired
from csa.paired import PairedModel
from csa.recording import machine_fingerprint
from csa.sparse import SparseConfig


@torch.no_grad()
def time_mode(pm: PairedModel, ids, mode: str, cfg: SparseConfig,
              n_steps: int = 24, warmup: int = 4):
    """Time n_steps of decode in a given attention mode; returns ms/step."""
    paired.STATE.cfg = cfg
    cache = pm._new_cache()
    paired.STATE.mode = "off"
    paired.STATE.recorder = None
    out = pm._forward(ids, cache)
    tok = int(out.logits[0, -1].argmax())

    times = []
    for i in range(n_steps + warmup):
        paired.STATE.mode = mode
        t0 = time.perf_counter()
        out = pm._forward(torch.tensor([[tok]], device=pm.device), cache)
        if pm.device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt * 1000)
        tok = int(out.logits[0, -1].argmax())
    paired.STATE.mode = "off"
    return float(np.median(times)), float(np.std(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--budgets", type=float, nargs="+",
                    default=[0.5, 0.25, 0.125, 0.0625])
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--out", default="results/overhead")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pm = PairedModel(args.model, device=device)

    filler = ("The archive at Millbrook recorded every transaction of the "
              "season, and the ledger was kept in the upper room. ")
    text = filler * (args.context // 20 + 8)
    ids = pm.encode_chat(text + "\nSummarize the archive.")
    ids = ids[:, : args.context]
    print(f"context tokens: {ids.shape[1]}")

    base_cfg = SparseConfig(keep_frac=1.0, min_kv_sparse=64)
    dense_ms, dense_sd = time_mode(pm, ids, "dense", base_cfg, args.steps)
    print(f"dense step      : {dense_ms:7.2f} ms  (+/-{dense_sd:.2f})")

    rows = []
    for kf in args.budgets:
        cfg = SparseConfig(method="quest_topk", keep_frac=kf, min_kv_sparse=64)
        sparse_ms, sparse_sd = time_mode(pm, ids, "sparse_only", cfg, args.steps)
        paired_ms, _ = time_mode(pm, ids, "sparse", cfg, max(args.steps // 3, 6))
        r = dense_ms / sparse_ms  # probe (a dense step) relative to a sparse step
        row = dict(keep_frac=kf, context=int(ids.shape[1]),
                   dense_ms=dense_ms, sparse_only_ms=sparse_ms,
                   sparse_only_std=sparse_sd, paired_ms=paired_ms,
                   speedup_vs_dense=dense_ms / sparse_ms,
                   cost_ratio_r=r,
                   # throughput loss if we probe p% of steps
                   loss_at_1pct=0.01 * r / (1 + 0.01 * r),
                   loss_at_5pct=0.05 * r / (1 + 0.05 * r),
                   loss_at_10pct=0.10 * r / (1 + 0.10 * r))
        rows.append(row)
        print(f"keep={kf:<7g} sparse {sparse_ms:7.2f} ms | speedup "
              f"{row['speedup_vs_dense']:.2f}x | r={r:.2f} | "
              f"loss@5% probes = {row['loss_at_5pct'] * 100:.1f}%")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "rows": rows,
        "fingerprint": machine_fingerprint(),
        "caveat": ("Small model on a small GPU: decode is launch/MLP-bound, "
                   "not KV-bandwidth-bound. Measured sparsity speedup is a "
                   "lower bound and r is an upper bound; the bandwidth-bound "
                   "regime requires the rented-GPU phase."),
    }
    with open(out / "overhead.json", "w") as f:
        json.dump(payload, f, indent=2, default=float)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "overhead.csv", index=False)
    print(f"\nwrote {out/'overhead.json'}")
    print(payload["caveat"])


if __name__ == "__main__":
    main()
