import numpy as np
import torch

from csa import sparse as sp
from csa.sparse import SparseConfig, layer_keep_fracs


def rand_qkv(B=1, Hq=4, Hkv=2, T=320, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, Hq, 1, D, generator=g),
            torch.randn(B, Hkv, T, D, generator=g),
            torch.randn(B, Hkv, T, D, generator=g))


# --------------------------------------------------------------- gather path
def test_gather_matches_masked_attention():
    """The production gather path and the measurement masking path must agree.

    If they diverge, every overhead number is measured on a different
    computation than every fidelity number.
    """
    for seed in range(4):
        q, k, v = rand_qkv(seed=seed)
        groups, bs, T = 2, 32, k.shape[2]
        mask, _, _ = sp.compute_selection(
            SparseConfig(keep_frac=0.25, block_size=bs), q, k, 0.25, groups)

        gathered = sp.gather_sparse_attention(q, k, v, mask, bs, 0.25, groups)

        kk = sp.expand_kv_heads(k, groups).float()
        vv = sp.expand_kv_heads(v, groups).float()
        logits = torch.matmul(q.float(), kk.transpose(2, 3)) * 0.25
        tok = sp.token_mask_from_blocks(mask, bs, T)
        masked = torch.matmul(
            torch.softmax(logits.masked_fill(~tok.unsqueeze(2), float("-inf")), -1),
            vv)
        assert torch.allclose(gathered, masked, atol=1e-4), \
            f"seed {seed}: max diff {(gathered - masked).abs().max():.2e}"


def test_gather_handles_partial_tail_block():
    q, k, v = rand_qkv(T=300)  # 300 = 9*32 + 12, ragged tail
    mask, _, _ = sp.compute_selection(
        SparseConfig(keep_frac=0.5, block_size=32), q, k, 0.25, 2)
    out = sp.gather_sparse_attention(q, k, v, mask, 32, 0.25, 2)
    assert torch.isfinite(out).all()


def test_gather_full_budget_equals_dense():
    q, k, v = rand_qkv()
    mask, _, _ = sp.compute_selection(
        SparseConfig(keep_frac=1.0, block_size=32), q, k, 0.25, 2)
    out = sp.gather_sparse_attention(q, k, v, mask, 32, 0.25, 2)
    kk = sp.expand_kv_heads(k, 2).float()
    vv = sp.expand_kv_heads(v, 2).float()
    dense = torch.matmul(
        torch.softmax(torch.matmul(q.float(), kk.transpose(2, 3)) * 0.25, -1), vv)
    assert torch.allclose(out, dense, atol=1e-4)


def test_gather_respects_attention_bias():
    q, k, v = rand_qkv()
    mask, _, _ = sp.compute_selection(
        SparseConfig(keep_frac=0.5, block_size=32), q, k, 0.25, 2)
    T = k.shape[2]
    bias = torch.zeros(1, 1, T)
    bias[..., T // 2:] = float("-inf")  # forbid the second half
    out = sp.gather_sparse_attention(q, k, v, mask, 32, 0.25, 2, attn_bias=bias)
    assert torch.isfinite(out).all(), "masked-out positions must not produce NaN"


# ------------------------------------------------------------ layer schedules
def test_schedules_are_budget_matched():
    for n in (4, 24, 28):
        for kf in (0.5, 0.125, 0.03125):
            for sch in ("uniform", "pyramid", "inv_pyramid"):
                fr = layer_keep_fracs(kf, n, sch)
                assert len(fr) == n
                assert abs(float(fr.mean()) - kf) < 1e-9, (sch, n, kf, fr.mean())
                assert (fr > 0).all() and (fr <= 1.0).all()


def test_pyramid_front_loads_and_inverse_mirrors():
    n = 24
    p = layer_keep_fracs(0.25, n, "pyramid")
    q = layer_keep_fracs(0.25, n, "inv_pyramid")
    assert p[0] > p[-1]
    assert q[0] < q[-1]
    assert np.allclose(p, q[::-1])


def test_uniform_schedule_is_exactly_flat():
    fr = layer_keep_fracs(0.25, 12, "uniform")
    assert np.allclose(fr, 0.25)


def test_effective_keep_frac_defaults_to_uniform_behavior():
    cfg = SparseConfig(keep_frac=0.25)
    assert sp.effective_keep_frac(cfg, 5, 24) == 0.25
    assert sp.effective_keep_frac(cfg, None, None) == 0.25


def test_schedule_changes_selection_but_not_average_budget():
    q, k, _ = rand_qkv(T=640)
    n_layers = 24
    kept = {}
    for sch in ("uniform", "pyramid"):
        cfg = SparseConfig(keep_frac=0.25, block_size=32, layer_schedule=sch)
        fracs = []
        for li in range(n_layers):
            m, _, _ = sp.compute_selection(cfg, q, k, 0.25, 2, li, n_layers)
            fracs.append(float(m.float().mean()))
        kept[sch] = fracs
    assert kept["uniform"][0] == kept["uniform"][-1]
    assert kept["pyramid"][0] > kept["pyramid"][-1]
    assert abs(np.mean(kept["pyramid"]) - np.mean(kept["uniform"])) < 0.02


def test_unknown_schedule_raises():
    import pytest
    with pytest.raises(ValueError):
        layer_keep_fracs(0.25, 8, "nonsense")
