"""Selection-based block-sparse KV attention primitives.

All methods here are *selection-based, not eviction-based*: the full KV cache is
retained and sparsity is applied by masking non-selected blocks at attention
time. This is methodologically load-bearing for the paired harness — eviction
would destroy the dense counterfactual (proposal §7, Study A).

Shapes follow HF conventions:
    query        (B, Hq,  Q, D)
    key / value  (B, Hkv, T, D)   (pre-GQA-expansion, RoPE already applied)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SparseConfig:
    method: str = "quest_topk"  # quest_topk | mean_topk | local_sink
    keep_frac: float = 0.25     # mean fraction of KV blocks attended per head
    block_size: int = 32
    min_kv_sparse: int = 256    # below this many cached tokens, run dense
    sink_blocks: int = 1        # always-kept prefix blocks
    local_blocks: int = 1       # always-kept suffix blocks
    # per-layer budget allocation (Ablation 6: composition with PSA-style
    # allocators). "uniform" reproduces a flat budget exactly.
    layer_schedule: str = "uniform"  # uniform | pyramid | inv_pyramid
    schedule_strength: float = 0.6

    def label(self) -> str:
        s = "" if self.layer_schedule == "uniform" else f"/{self.layer_schedule}"
        return f"{self.method}@{self.keep_frac:g}{s}"


def layer_keep_fracs(keep_frac: float, n_layers: int, schedule: str = "uniform",
                     strength: float = 0.6):
    """Per-layer keep fractions whose MEAN equals keep_frac.

    Budget-matched by construction, so schedules differ only in *where* the
    budget goes, never in how much is spent — otherwise a schedule comparison
    would just be a budget comparison.

      uniform     - flat.
      pyramid     - more budget in early layers, less in late.
      inv_pyramid - the reverse.
    """
    import numpy as np
    n = max(int(n_layers), 1)
    kf = float(keep_frac)
    if not (0.0 < kf <= 1.0):
        raise ValueError(f"keep_frac must be in (0, 1], got {kf}")
    if n <= 1 or schedule == "uniform":
        return np.full(n, kf)
    t = np.linspace(-1.0, 1.0, n)
    if schedule == "pyramid":
        w = 1.0 - strength * t
    elif schedule == "inv_pyramid":
        w = 1.0 + strength * t
    else:
        raise ValueError(f"unknown layer schedule: {schedule}")
    w = np.maximum(np.asarray(w, dtype=float), 1e-12)
    # Exact mean before projection, then water-fill onto [eps, 1].
    fr = kf * (w / w.mean())
    lo, hi = 1e-6, 1.0
    target_sum = kf * n
    for _ in range(n + 5):
        fr = np.clip(fr, lo, hi)
        deficit = target_sum - fr.sum()
        if abs(deficit) < 1e-12 * n:
            break
        free = (fr < hi - 1e-15) if deficit > 0 else (fr > lo + 1e-15)
        if not free.any():
            break
        fr[free] += deficit / free.sum()
    fr = np.clip(fr, lo, hi)
    if abs(float(fr.mean()) - kf) > 1e-9:
        raise ValueError(
            f"infeasible layer schedule: mean keep_frac={kf} with "
            f"schedule={schedule!r}, strength={strength}, n_layers={n} "
            f"(achieved mean={fr.mean():.6f})"
        )
    return fr


def n_blocks(kv_len: int, block_size: int) -> int:
    return (kv_len + block_size - 1) // block_size


def block_token_counts(kv_len: int, block_size: int, device, dtype=torch.float32):
    nb = n_blocks(kv_len, block_size)
    counts = torch.full((nb,), float(block_size), device=device, dtype=dtype)
    rem = kv_len - (nb - 1) * block_size
    counts[-1] = float(rem)
    return counts


def block_minmax(keys: torch.Tensor, block_size: int):
    """Per-block coordinate-wise min/max pooled keys.

    keys: (B, H, T, D) -> (kmin, kmax) each (B, H, nb, D).
    Padded tail positions are +/-inf so they never win the min/max.
    """
    B, H, T, D = keys.shape
    nb = n_blocks(T, block_size)
    pad = nb * block_size - T
    kf = keys.float()
    # large-finite padding (not inf): downstream computes q*kmin, and a
    # zero query coordinate times inf would poison the score with NaN
    if pad:
        kmin_src = F.pad(kf, (0, 0, 0, pad), value=1e30)
        kmax_src = F.pad(kf, (0, 0, 0, pad), value=-1e30)
    else:
        kmin_src = kmax_src = kf
    kmin = kmin_src.view(B, H, nb, block_size, D).amin(dim=3)
    kmax = kmax_src.view(B, H, nb, block_size, D).amax(dim=3)
    return kmin, kmax


def block_mean(keys: torch.Tensor, block_size: int):
    """Per-block mean-pooled keys (padding-aware). (B,H,T,D) -> (B,H,nb,D)."""
    B, H, T, D = keys.shape
    nb = n_blocks(T, block_size)
    pad = nb * block_size - T
    kf = keys.float()
    if pad:
        kf = F.pad(kf, (0, 0, 0, pad), value=0.0)
    ksum = kf.view(B, H, nb, block_size, D).sum(dim=3)
    counts = block_token_counts(T, block_size, keys.device).view(1, 1, nb, 1)
    return ksum / counts


def expand_kv_heads(x: torch.Tensor, groups: int) -> torch.Tensor:
    """(B, Hkv, ..., ...) -> (B, Hkv*groups, ...); matches HF repeat_kv ordering."""
    if groups == 1:
        return x
    return x.repeat_interleave(groups, dim=1)


def quest_block_scores(query: torch.Tensor, keys: torch.Tensor, block_size: int, groups: int):
    """Quest-style per-block upper bound on q.k within each block.

    score[b] = sum_d max(q_d * kmin_{b,d}, q_d * kmax_{b,d})  >= max_{i in b} q.k_i
    query: (B, Hq, 1, D); keys: (B, Hkv, T, D) -> scores (B, Hq, nb), un-scaled logits.
    """
    kmin, kmax = block_minmax(keys, block_size)           # (B, Hkv, nb, D)
    kmin = expand_kv_heads(kmin, groups)                  # (B, Hq, nb, D)
    kmax = expand_kv_heads(kmax, groups)
    q = query.float().squeeze(2).unsqueeze(2)             # (B, Hq, 1, D)
    scores = torch.maximum(q * kmin, q * kmax).sum(dim=-1)  # (B, Hq, nb)
    return scores


def mean_block_scores(query: torch.Tensor, keys: torch.Tensor, block_size: int, groups: int):
    """Mean-pooled block relevance: q . mean(k in block). (B, Hq, nb), un-scaled."""
    kmean = expand_kv_heads(block_mean(keys, block_size), groups)  # (B, Hq, nb, D)
    q = query.float().squeeze(2).unsqueeze(2)
    return (q * kmean).sum(dim=-1)


def select_topk_blocks(scores: torch.Tensor, keep_frac: float,
                       sink_blocks: int = 1, local_blocks: int = 1) -> torch.Tensor:
    """Per-head top-k block selection under a block budget.

    scores: (B, H, nb). Returns bool mask (B, H, nb), True = block attended.
    Sink (prefix) and local (suffix) blocks are always selected and count
    against the budget.
    """
    B, H, nb = scores.shape
    k = int(round(keep_frac * nb))
    k = max(k, min(nb, sink_blocks + local_blocks))
    k = min(k, nb)
    boosted = scores.clone()
    if sink_blocks > 0:
        boosted[..., : min(sink_blocks, nb)] = float("inf")
    if local_blocks > 0:
        boosted[..., -min(local_blocks, nb):] = float("inf")
    idx = boosted.topk(k, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


def local_sink_mask(B: int, H: int, nb: int, keep_frac: float, device,
                    sink_blocks: int = 1) -> torch.Tensor:
    """StreamingLLM-style static pattern: sink prefix + local suffix window.

    The block budget implied by keep_frac is spent on the most recent blocks
    (after the sink), independent of content. Returns (B, H, nb) bool.
    """
    k = max(int(round(keep_frac * nb)), min(nb, sink_blocks + 1))
    k = min(k, nb)
    mask = torch.zeros(B, H, nb, dtype=torch.bool, device=device)
    s = min(sink_blocks, nb)
    mask[..., :s] = True
    local = max(k - s, 1)
    mask[..., -local:] = True
    return mask


def token_mask_from_blocks(block_mask: torch.Tensor, block_size: int, kv_len: int) -> torch.Tensor:
    """(B, H, nb) bool -> (B, H, kv_len) bool."""
    tok = block_mask.repeat_interleave(block_size, dim=-1)
    return tok[..., :kv_len]


def block_mass_estimate(scores: torch.Tensor, scaling: float, kv_len: int,
                        block_size: int) -> torch.Tensor:
    """Label-free estimate of the attention-probability mass in each block.

    Treats the (upper-bound or mean) block score as a representative logit for
    every token in the block:  est_mass_b ∝ count_b * exp(scaling * score_b).
    Returns (B, H, nb), rows summing to 1. This is exactly the quantity a
    selection-based method already computes and discards (proposal §6.1).
    """
    counts = block_token_counts(kv_len, block_size, scores.device)
    z = scores.float() * scaling + torch.log(counts).view(1, 1, -1)
    return torch.softmax(z, dim=-1)


def block_mass_from_probs(probs: torch.Tensor, block_size: int) -> torch.Tensor:
    """Oracle block mass from true attention probabilities.

    probs: (B, H, T) -> (B, H, nb) summing token probabilities per block.
    """
    B, H, T = probs.shape
    nb = n_blocks(T, block_size)
    pad = nb * block_size - T
    p = probs.float()
    if pad:
        p = F.pad(p, (0, pad), value=0.0)
    return p.view(B, H, nb, block_size).sum(dim=-1)


def effective_keep_frac(cfg: SparseConfig, layer_idx: int | None,
                        n_layers: int | None) -> float:
    """This layer's budget under the configured schedule."""
    if (cfg.layer_schedule == "uniform" or layer_idx is None
            or n_layers is None or n_layers <= 1):
        return cfg.keep_frac
    fr = layer_keep_fracs(cfg.keep_frac, n_layers, cfg.layer_schedule,
                          cfg.schedule_strength)
    return float(fr[min(max(layer_idx, 0), n_layers - 1)])


def compute_selection(cfg: SparseConfig, query: torch.Tensor, key: torch.Tensor,
                      scaling: float, groups: int,
                      layer_idx: int | None = None,
                      n_layers: int | None = None):
    """Run the configured selection method.

    Returns (block_mask (B,Hq,nb) bool, scores (B,Hq,nb) or None, est_mass or None).
    """
    kv_len = key.shape[2]
    keep = effective_keep_frac(cfg, layer_idx, n_layers)
    if cfg.method == "quest_topk":
        scores = quest_block_scores(query, key, cfg.block_size, groups)
    elif cfg.method == "mean_topk":
        scores = mean_block_scores(query, key, cfg.block_size, groups)
    elif cfg.method == "local_sink":
        B, Hq = query.shape[0], query.shape[1]
        nb = n_blocks(kv_len, cfg.block_size)
        mask = local_sink_mask(B, Hq, nb, keep, query.device, cfg.sink_blocks)
        return mask, None, None
    else:
        raise ValueError(f"unknown sparse method: {cfg.method}")
    mask = select_topk_blocks(scores, keep, cfg.sink_blocks, cfg.local_blocks)
    est_mass = block_mass_estimate(scores, scaling, kv_len, cfg.block_size)
    return mask, scores, est_mass


def gather_sparse_attention(query, key, value, block_mask, block_size,
                            scaling, groups, attn_bias=None):
    """Production sparse path: attend only over SELECTED blocks.

    Masking the full logit matrix measures nothing about cost, since the whole
    matmul still runs. Here the selected blocks are gathered per head and
    attention runs over the gathered subset only, so the work really is
    proportional to the budget — which is what an overhead measurement needs.

    query (B,Hq,1,D); key/value (B,Hkv,T,D); block_mask (B,Hq,nb) bool.
    """
    B, Hq, _, D = query.shape
    kv_len = key.shape[2]
    nb = block_mask.shape[-1]
    k = expand_kv_heads(key, groups)
    v = expand_kv_heads(value, groups)

    # Every current method keeps the same COUNT per head, so the gather is
    # rectangular. If a future method selects unequal counts, topk on the
    # float mask pads the short rows with UNSELECTED blocks -- attending to
    # blocks the method deliberately dropped. Gather the selection flag too
    # and mask those padded slots out, so the gather path can never silently
    # attend outside the selected set.
    counts = block_mask.sum(-1)
    kcount = max(int(counts.max()), 1)
    idx = torch.topk(block_mask.float(), kcount, dim=-1).indices  # (B,Hq,kc)
    idx, _ = idx.sort(dim=-1)
    selected = torch.gather(block_mask, 2, idx)                   # (B,Hq,kc)

    ar = torch.arange(block_size, device=query.device)
    tok_idx = (idx.unsqueeze(-1) * block_size + ar).reshape(B, Hq, -1)
    valid = (tok_idx < kv_len) & selected.repeat_interleave(block_size, dim=-1)
    tok_idx = tok_idx.clamp(max=kv_len - 1)

    gi = tok_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    kg = torch.gather(k, 2, gi)
    vg = torch.gather(v, 2, gi)

    logits = torch.matmul(query.float(), kg.float().transpose(2, 3)) * scaling
    logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
    if attn_bias is not None:
        bias = torch.gather(attn_bias.expand(B, Hq, kv_len), 2, tok_idx)
        logits = logits + bias.unsqueeze(2)
    # All-masked rows (empty selection or bias forbidding every gathered
    # token) make softmax(-inf) -> NaN. Emit zeros instead of poisoning
    # the residual stream.
    has_logit = torch.isfinite(logits).any(dim=-1, keepdim=True)
    logits = logits.masked_fill(~has_logit, 0.0)
    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0).masked_fill(~has_logit, 0.0)
    return torch.matmul(probs, vg.float())
