"""Cache-crop correctness — the invariant the whole measurement rests on.

A dense probe runs a second forward on the same token and then crops the KV
cache back. If that crop is ever ineffective, the probe's KV stays behind,
every later step attends to polluted state, and the divergence numbers remain
entirely plausible while being wrong. That failure must be loud.
"""

import pytest
import torch

from csa.paired import _cache_len, crop_cache


def _cache(n_tokens: int):
    from transformers import DynamicCache
    c = DynamicCache()
    c.update(torch.zeros(1, 2, n_tokens, 4), torch.zeros(1, 2, n_tokens, 4), 0)
    return c


def test_crop_trims_to_requested_length():
    c = _cache(9)
    assert _cache_len(c) == 9
    crop_cache(c, 4)
    assert _cache_len(c) == 4


def test_crop_to_zero_and_to_full_length():
    c = _cache(6)
    crop_cache(c, 6)          # no-op crop is legitimate
    assert _cache_len(c) == 6
    crop_cache(c, 0)
    assert _cache_len(c) == 0


def test_ineffective_crop_raises_rather_than_silently_polluting():
    class Deceptive:
        """A cache whose crop() silently does nothing."""

        def __init__(self):
            layer = type("L", (), {"keys": torch.zeros(1, 2, 9, 4),
                                   "values": torch.zeros(1, 2, 9, 4)})()
            self.layers = [layer]

        def crop(self, n):
            pass

    with pytest.raises(RuntimeError, match="ineffective"):
        crop_cache(Deceptive(), 4)


def test_unknown_cache_layout_raises():
    with pytest.raises(RuntimeError, match="don't know how to crop"):
        crop_cache(object(), 4)
