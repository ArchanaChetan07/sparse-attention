"""Mechanism 1 — label-free fidelity signals, plus oracle counterparts.

Every function is a pure tensor -> python-scalar(s) computation so it can be
unit-tested without a model. Signals marked LABEL-FREE use only state a
selection-based sparse serving system already materializes (block scores,
selected sets, sparse attention weights). ORACLE quantities additionally use
the dense attention distribution, which only the paired harness has.
"""

from __future__ import annotations

import torch


def _mm(x: torch.Tensor):
    """(mean, max) over all leading dims of a tensor of per-head scalars."""
    return float(x.mean()), float(x.max())


def dropped_mass_oracle(dense_probs: torch.Tensor, token_mask: torch.Tensor):
    """ORACLE: dense attention mass on non-selected tokens.

    dense_probs: (B, H, T) rows sum to 1;  token_mask: (B, H, T) True = kept.
    """
    kept = (dense_probs * token_mask.float()).sum(dim=-1)
    dropped = (1.0 - kept).clamp(0.0, 1.0)
    return _mm(dropped)


def dropped_mass_estimate(est_mass: torch.Tensor, block_mask: torch.Tensor):
    """LABEL-FREE: estimated mass in dropped blocks, from discarded block scores.

    est_mass: (B, H, nb) rows sum to 1;  block_mask: (B, H, nb) True = kept.
    """
    dropped = (est_mass * (~block_mask).float()).sum(dim=-1).clamp(0.0, 1.0)
    return _mm(dropped)


def eviction_consensus(block_mask: torch.Tensor, weights: torch.Tensor):
    """Cross-head eviction consensus: is dropped content dropped *everywhere*?

    For each block, d_b = fraction of heads that dropped it. Returns
    (consensus, fully_dropped_frac):
      consensus          = sum_b w_b * d_b / sum_b w_b
      fully_dropped_frac = sum_b w_b * 1[d_b == 1] / sum_b w_b
    where w_b is a per-block importance weight (estimated or oracle block mass,
    averaged over heads). The label-free analogue of the "global eviction"
    mechanism identified as causal for the fidelity cliff (arXiv 2603.01426).
    """
    d = 1.0 - block_mask.float().mean(dim=1)          # (B, nb)
    w = weights.float().mean(dim=1)                   # (B, nb)
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    consensus = float((w * d).sum(dim=-1).mean())
    fully = float((w * (d >= 1.0 - 1e-6).float()).sum(dim=-1).mean())
    return consensus, fully


def normalized_entropy(probs: torch.Tensor, valid_counts=None):
    """Entropy of attention rows, normalized to [0,1] by log(support size).

    probs: (B, H, T). Zero entries contribute 0. If valid_counts is None the
    support is T.
    """
    p = probs.float().clamp_min(0)
    plogp = torch.where(p > 0, p * torch.log(p), torch.zeros_like(p))
    ent = -plogp.sum(dim=-1)
    if valid_counts is None:
        denom = torch.log(torch.tensor(float(probs.shape[-1]), device=probs.device))
    else:
        denom = torch.log(valid_counts.float().clamp_min(2.0))
    return _mm(ent / denom.clamp_min(1e-6))


def output_divergence(dense_out: torch.Tensor, sparse_out: torch.Tensor):
    """Per-head divergence between paired attention outputs on identical state.

    dense_out/sparse_out: (B, H, D). Returns (cos_mean, cos_max, relL2_mean, relL2_max).
    """
    a = dense_out.float()
    b = sparse_out.float()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    cos_dist = (1.0 - cos).clamp_min(0.0)
    rel = (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-6)
    cm, cx = _mm(cos_dist)
    rm, rx = _mm(rel)
    return cm, cx, rm, rx


def logit_metrics(logits: torch.Tensor):
    """Top-1/top-2 margin and normalized entropy of a final logit row (V,)."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    top2 = lp.topk(2)
    margin = float(top2.values[0] - top2.values[1])
    p = lp.exp()
    ent = float(-(p * lp).sum() / torch.log(torch.tensor(float(logits.shape[-1]))))
    return margin, ent


def logit_divergence(dense_logits: torch.Tensor, sparse_logits: torch.Tensor):
    """KL(dense || sparse) in nats, and whether the greedy token flips."""
    ld = torch.log_softmax(dense_logits.float(), dim=-1)
    ls = torch.log_softmax(sparse_logits.float(), dim=-1)
    kl = float((ld.exp() * (ld - ls)).sum())
    flip = bool(dense_logits.argmax() != sparse_logits.argmax())
    return kl, flip
