import math

import torch

from csa import sparse as sp
from csa.sparse import SparseConfig


def rand_qkv(B=1, Hq=4, Hkv=2, T=200, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, Hq, 1, D, generator=g)
    k = torch.randn(B, Hkv, T, D, generator=g)
    v = torch.randn(B, Hkv, T, D, generator=g)
    return q, k, v


def test_block_minmax_bounds_actual_scores():
    q, k, _ = rand_qkv()
    groups = 2
    bs = 32
    scores = sp.quest_block_scores(q, k, bs, groups)  # (B,Hq,nb)
    kfull = sp.expand_kv_heads(k, groups)
    logits = torch.einsum("bhqd,bhtd->bht", q.float(), kfull.float())  # (B,Hq,T)
    nb = sp.n_blocks(k.shape[2], bs)
    for b in range(nb):
        blk = logits[..., b * bs:(b + 1) * bs].amax(-1)
        assert (scores[..., b] >= blk - 1e-4).all(), "quest score must upper-bound block max"


def test_full_budget_selects_everything():
    q, k, _ = rand_qkv()
    mask, scores, est = sp.compute_selection(
        SparseConfig(method="quest_topk", keep_frac=1.0), q, k, 0.25, groups=2)
    assert mask.all()


def test_token_mask_shape_and_partial_block():
    T = 200  # not a multiple of 32
    q, k, _ = rand_qkv(T=T)
    mask, _, _ = sp.compute_selection(
        SparseConfig(keep_frac=0.5, block_size=32), q, k, 0.25, groups=2)
    tok = sp.token_mask_from_blocks(mask, 32, T)
    assert tok.shape[-1] == T
    # sink and local blocks always kept
    assert tok[..., :32].all()
    assert tok[..., -8:].all()


def test_est_mass_sums_to_one():
    q, k, _ = rand_qkv()
    scores = sp.quest_block_scores(q, k, 32, 2)
    est = sp.block_mass_estimate(scores, 0.25, k.shape[2], 32)
    assert torch.allclose(est.sum(-1), torch.ones_like(est.sum(-1)), atol=1e-5)


def test_local_sink_pattern():
    mask = sp.local_sink_mask(1, 4, 10, keep_frac=0.4, device="cpu", sink_blocks=1)
    assert mask[..., 0].all()
    assert mask[..., -3:].all()
    assert not mask[..., 1:7].any()


def test_dropped_mass_monotone_in_budget():
    from csa import signals as S
    q, k, _ = rand_qkv(T=320)
    logits = torch.einsum("bhqd,bhtd->bht", q.float(),
                          sp.expand_kv_heads(k, 2).float()) * 0.25
    probs = torch.softmax(logits, -1)
    drops = []
    for frac in (0.125, 0.25, 0.5, 1.0):
        mask, _, _ = sp.compute_selection(
            SparseConfig(keep_frac=frac, block_size=32), q, k, 0.25, 2)
        tok = sp.token_mask_from_blocks(mask, 32, 320)
        d, _ = S.dropped_mass_oracle(probs, tok)
        drops.append(d)
    assert drops == sorted(drops, reverse=True)
    assert drops[-1] < 1e-6  # full budget drops (numerically) nothing
