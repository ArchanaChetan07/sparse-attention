"""Integration smoke check: paired attention on a real model.

Verifies, on one prompt:
  1. keep_frac=1.0 -> sparse output ~= dense output (cos dist ~ 0, no flips)
  2. keep_frac=0.25 -> runs, records signals, flips possible
  3. dense probe leaves no trace (crop works): free-running with probe=True
     produces identical tokens to probe=False at the same config/seed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from csa.paired import PairedModel
from csa.sparse import SparseConfig

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    pm = PairedModel(MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    prompt = ("The town of Millbrook kept its records in the old library. " * 40
              + "\nIn one sentence, what did Millbrook keep in the library?")
    ids = pm.encode_chat(prompt)
    print(f"prompt tokens: {ids.shape[1]}")

    ref_toks, ref_text = pm.generate_dense(ids, 24)
    print(f"dense ref: {ref_text!r}")

    # 1) full budget == dense
    cfg = SparseConfig(method="quest_topk", keep_frac=1.0, min_kv_sparse=64)
    r = pm.generate_paired(ids, 24, cfg)
    flips = sum(row.get("top1_flip", 0) for row in r.rows)
    cos = max(row["out_cos_max_Lmax"] for row in r.rows)
    print(f"[keep=1.0 ] steps={len(r.rows)} flips={flips} max_cos_dist={cos:.2e}")
    assert flips == 0, "full budget must not flip tokens"
    assert cos < 1e-3, f"full budget divergence too high: {cos}"

    # 2) sparse budget runs and records signals
    cfg = SparseConfig(method="quest_topk", keep_frac=0.25, min_kv_sparse=64)
    r = pm.generate_paired(ids, 24, cfg)
    row = r.rows[0]
    need = ["est_dropped_mean_Lmean", "oracle_dropped_mean_Lmean",
            "consensus_est_Lmean", "out_cos_mean_Lmean", "logit_kl", "top1_flip"]
    missing = [k for k in need if k not in row]
    assert not missing, f"missing recorded fields: {missing}"
    print(f"[keep=0.25] steps={len(r.rows)} "
          f"flips={sum(x['top1_flip'] for x in r.rows)} "
          f"est_drop={row['est_dropped_mean_Lmean']:.3f} "
          f"oracle_drop={row['oracle_dropped_mean_Lmean']:.3f} "
          f"text={r.text[:60]!r}")

    # 3) probe must not perturb the trajectory
    r_probe = pm.generate_paired(ids, 16, cfg, probe=True)
    r_nop = pm.generate_paired(ids, 16, cfg, probe=False)
    assert r_probe.tokens == r_nop.tokens, (
        f"probe polluted the cache!\n with: {r_probe.tokens}\n w/o : {r_nop.tokens}")
    print("[probe    ] trajectory identical with/without dense probes  OK")

    # 4) production gather path must match the measurement masking path
    from csa import paired as P
    P.STATE.cfg = cfg
    toks_masked, toks_gather = [], []
    for mode, sink in (("sparse", toks_masked), ("sparse_only", toks_gather)):
        cache = pm._new_cache()
        P.STATE.mode = "off"; P.STATE.recorder = None
        out = pm._forward(ids, cache)
        tok = int(out.logits[0, -1].argmax())
        for _ in range(12):
            sink.append(tok)
            P.STATE.mode = mode
            out = pm._forward(torch.tensor([[tok]], device=pm.device), cache)
            tok = int(out.logits[0, -1].argmax())
        P.STATE.mode = "off"
    assert toks_masked == toks_gather, (
        f"gather path diverges from masked path!\n masked: {toks_masked}\n "
        f"gather: {toks_gather}")
    print("[gather   ] production sparse path matches masked path  OK")

    # 5) timing snapshot
    ps = [x["probe_s"] for x in r_probe.rows]
    ss = [x["step_s"] for x in r_probe.rows]
    print(f"[timing   ] probe {1e3*sum(ps)/len(ps):.1f} ms | "
          f"paired step {1e3*sum(ss)/len(ss):.1f} ms")
    print("SMOKE CHECK PASSED")


if __name__ == "__main__":
    main()
