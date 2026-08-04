"""Paired dense/sparse execution harness (Study A substrate).

A custom attention function is registered with HF Transformers'
AttentionInterface. During decode it computes BOTH the dense attention output
and the selection-based sparse output from the *same* Q/KV state, records
per-layer signals + divergence, and returns whichever output the current mode
dictates. The full KV cache is always retained (selection, not eviction), so
the dense counterfactual is exact.

Modes (module-level STATE.mode):
  "off"    - fast dense (sdpa), no measurement. Prefill and reference runs.
  "dense"  - fast dense (sdpa), no measurement. Dense probe / dense trajectory.
  "sparse" - paired measurement; sparse output is returned (propagates).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

from . import signals as S
from . import sparse as sp
from .sparse import SparseConfig


class _State:
    mode: str = "off"
    cfg: Optional[SparseConfig] = None
    recorder: Optional["StepRecorder"] = None
    n_layers: Optional[int] = None  # for per-layer budget schedules


STATE = _State()


class StepRecorder:
    """Collects per-layer metric dicts within a step, aggregates on finalize."""

    def __init__(self):
        self.layer_buf: list[dict] = []
        self.rows: list[dict] = []

    def add_layer(self, layer_idx, metrics: dict):
        m = dict(metrics)
        m["layer"] = layer_idx
        self.layer_buf.append(m)

    def aggregate_layers(self) -> dict:
        out = {}
        if not self.layer_buf:
            return out
        keys = [k for k in self.layer_buf[0] if k != "layer"]
        for k in keys:
            vals = [r[k] for r in self.layer_buf if r.get(k) is not None]
            if vals:
                t = torch.tensor(vals, dtype=torch.float64)
                out[f"{k}_Lmean"] = float(t.mean())
                out[f"{k}_Lmax"] = float(t.max())
                out[f"{k}_Lstd"] = float(t.std(unbiased=False))
        self.layer_buf.clear()
        return out

    def finalize_step(self, extra: dict):
        row = self.aggregate_layers()
        row.update(extra)
        self.rows.append(row)


def csa_attention(module, query, key, value, attention_mask, scaling=None,
                  dropout=0.0, **kwargs):
    """Custom attention: dense fast path, or paired dense+sparse measurement."""
    if scaling is None:
        scaling = getattr(module, "scaling", None) or query.shape[-1] ** -0.5

    B, Hq, Q, D = query.shape
    Hkv = key.shape[1]
    groups = Hq // Hkv
    kv_len = key.shape[2]
    cfg = STATE.cfg

    eligible = (Q == 1 and cfg is not None and kv_len >= cfg.min_kv_sparse)
    sparse_active = STATE.mode == "sparse" and eligible
    sparse_only = STATE.mode == "sparse_only" and eligible

    if sparse_only:
        # production path: select, then attend over gathered blocks only.
        # No dense computation, no metrics -- this is what gets timed.
        block_mask, _, _ = sp.compute_selection(
            cfg, query, key, scaling, groups,
            getattr(module, "layer_idx", None), STATE.n_layers)
        bias = None
        if attention_mask is not None:
            bias = attention_mask[:, :, -1, :kv_len].float()
        out = sp.gather_sparse_attention(query, key, value, block_mask,
                                         cfg.block_size, scaling, groups, bias)
        return out.to(query.dtype).transpose(1, 2).contiguous(), None

    if not sparse_active:
        k = sp.expand_kv_heads(key, groups)
        v = sp.expand_kv_heads(value, groups)
        if attention_mask is not None:
            mask = attention_mask[:, :, :, :kv_len]
            out = F.scaled_dot_product_attention(query, k, v, attn_mask=mask, scale=scaling)
        else:
            out = F.scaled_dot_product_attention(query, k, v, is_causal=Q > 1, scale=scaling)
        return out.transpose(1, 2).contiguous(), None

    # ---- paired dense + sparse decode step -------------------------------
    k = sp.expand_kv_heads(key, groups)
    v = sp.expand_kv_heads(value, groups)
    logits = torch.matmul(query.float(), k.float().transpose(2, 3)) * scaling  # (B,Hq,1,T)
    if attention_mask is not None:
        logits = logits + attention_mask[:, :, :, :kv_len].float()
    dense_probs = torch.softmax(logits, dim=-1)                # (B,Hq,1,T)
    dense_out = torch.matmul(dense_probs, v.float())           # (B,Hq,1,D)

    block_mask, scores, est_mass = sp.compute_selection(
        cfg, query, key, scaling, groups,
        getattr(module, "layer_idx", None), STATE.n_layers)
    token_mask = sp.token_mask_from_blocks(block_mask, cfg.block_size, kv_len)  # (B,Hq,T)

    sparse_logits = logits.masked_fill(~token_mask.unsqueeze(2), float("-inf"))
    sparse_probs = torch.softmax(sparse_logits, dim=-1)
    sparse_out = torch.matmul(sparse_probs, v.float())

    # ---- per-layer metrics ----------------------------------------------
    dp = dense_probs.squeeze(2)      # (B,Hq,T)
    spr = sparse_probs.squeeze(2)
    m: dict = {}
    m["oracle_dropped_mean"], m["oracle_dropped_max"] = S.dropped_mass_oracle(dp, token_mask)
    oracle_block_mass = sp.block_mass_from_probs(dp, cfg.block_size)
    m["consensus_oracle"], m["fully_dropped_oracle"] = S.eviction_consensus(
        block_mask, oracle_block_mass)
    if est_mass is not None:  # label-free signals exist only for score-based methods
        m["est_dropped_mean"], m["est_dropped_max"] = S.dropped_mass_estimate(
            est_mass, block_mask)
        m["consensus_est"], m["fully_dropped_est"] = S.eviction_consensus(
            block_mask, est_mass)
    m["dense_entropy"], _ = S.normalized_entropy(dp)
    kept_counts = token_mask.sum(dim=-1)
    m["sparse_entropy"], _ = S.normalized_entropy(spr, kept_counts)
    (m["out_cos_mean"], m["out_cos_max"],
     m["out_relL2_mean"], m["out_relL2_max"]) = S.output_divergence(
        dense_out.squeeze(2), sparse_out.squeeze(2))
    m["keep_tokens_frac"] = float(token_mask.float().mean())

    if STATE.recorder is not None:
        STATE.recorder.add_layer(getattr(module, "layer_idx", -1), m)

    out = sparse_out.to(query.dtype)
    return out.transpose(1, 2).contiguous(), None


_REGISTERED = False


def _ensure_registered():
    global _REGISTERED
    if not _REGISTERED:
        AttentionInterface.register("csa_paired", csa_attention)
        _REGISTERED = True


def crop_cache(cache, length: int):
    """Trim a DynamicCache back to `length` tokens (removes probe pollution)."""
    if hasattr(cache, "crop"):
        cache.crop(length)
        return
    if hasattr(cache, "key_cache"):  # older layout
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i][..., :length, :]
            cache.value_cache[i] = cache.value_cache[i][..., :length, :]
        return
    if hasattr(cache, "layers"):  # transformers v5 layout
        for layer in cache.layers:
            layer.keys = layer.keys[..., :length, :]
            layer.values = layer.values[..., :length, :]
        return
    raise RuntimeError("don't know how to crop this cache type")


@dataclass
class StepResult:
    rows: list = field(default_factory=list)
    tokens: list = field(default_factory=list)
    text: str = ""


class PairedModel:
    """Loads a HF causal LM with the paired attention implementation."""

    def __init__(self, model_name: str, device: str = "cuda",
                 dtype: torch.dtype = torch.float16):
        _ensure_registered()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, attn_implementation="csa_paired",
        ).to(device).eval()
        self.device = device
        self.name = model_name
        STATE.n_layers = getattr(self.model.config, "num_hidden_layers", None)

    def _new_cache(self):
        from transformers import DynamicCache
        try:
            return DynamicCache(config=self.model.config)
        except TypeError:
            return DynamicCache()

    def _forward(self, ids: torch.Tensor, cache):
        return self.model(input_ids=ids, past_key_values=cache, use_cache=True)

    @torch.no_grad()
    def generate_dense(self, prompt_ids: torch.Tensor, max_new_tokens: int):
        """Fast dense greedy reference. Returns (token_ids, text)."""
        STATE.mode = "off"
        STATE.recorder = None
        cache = self._new_cache()
        out = self._forward(prompt_ids, cache)
        tok = int(out.logits[0, -1].argmax())
        toks = []
        for _ in range(max_new_tokens):
            if tok == self.tokenizer.eos_token_id:
                break  # EOS is not part of the trajectory (keeps teacher runs clean)
            toks.append(tok)
            out = self._forward(torch.tensor([[tok]], device=self.device), cache)
            tok = int(out.logits[0, -1].argmax())
        return toks, self.tokenizer.decode(toks, skip_special_tokens=True)

    @torch.no_grad()
    def generate_paired(self, prompt_ids: torch.Tensor, max_new_tokens: int,
                        cfg: SparseConfig, teacher_tokens: Optional[list] = None,
                        probe: bool = True) -> StepResult:
        """Sparse execution with per-step paired measurement.

        teacher_tokens=None  -> free-running: sparse model's own greedy tokens
                                propagate (production-like; compounding included).
        teacher_tokens=[...] -> teacher-forced on that trajectory: per-step
                                divergence without token-choice compounding.
        probe=True           -> before each sparse step, run the same token with
                                dense attention on the identical cache state and
                                record logit-level divergence labels; the cache
                                is cropped back so the probe leaves no trace.
        """
        rec = StepRecorder()
        STATE.cfg = cfg
        cache = self._new_cache()

        STATE.mode = "off"
        STATE.recorder = None
        t0 = time.perf_counter()
        out = self._forward(prompt_ids, cache)
        if self.device == "cuda":
            torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t0

        prompt_len = prompt_ids.shape[1]
        next_tok = int(out.logits[0, -1].argmax())
        result = StepResult()
        for step in range(max_new_tokens):
            tok = teacher_tokens[step] if teacher_tokens is not None else next_tok
            if tok == self.tokenizer.eos_token_id and teacher_tokens is None:
                break
            ids = torch.tensor([[tok]], device=self.device)
            cache_len = prompt_len + step

            dense_logits = None
            probe_s = 0.0
            if probe:
                STATE.mode = "dense"
                STATE.recorder = None
                t0 = time.perf_counter()
                dout = self._forward(ids, cache)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                probe_s = time.perf_counter() - t0
                dense_logits = dout.logits[0, -1].float().clone()
                crop_cache(cache, cache_len)

            STATE.mode = "sparse"
            STATE.recorder = rec
            t0 = time.perf_counter()
            sout = self._forward(ids, cache)
            if self.device == "cuda":
                torch.cuda.synchronize()
            step_s = time.perf_counter() - t0
            sparse_logits = sout.logits[0, -1].float()

            extra = {
                "step": step,
                "input_token": tok,
                "kv_len": cache_len,
                "prefill_s": prefill_s if step == 0 else 0.0,
                "probe_s": probe_s,
                "step_s": step_s,
            }
            extra["sparse_margin"], extra["sparse_logit_entropy"] = S.logit_metrics(sparse_logits)
            if dense_logits is not None:
                extra["logit_kl"], extra["top1_flip"] = S.logit_divergence(
                    dense_logits, sparse_logits)
                extra["dense_argmax"] = int(dense_logits.argmax())
            next_tok = int(sparse_logits.argmax())
            extra["sparse_argmax"] = next_tok
            rec.finalize_step(extra)
            result.tokens.append(tok)  # the committed token at this step

        STATE.mode = "off"
        STATE.recorder = None
        result.text = self.tokenizer.decode(result.tokens, skip_special_tokens=True)
        result.rows = rec.rows
        return result

    def encode_chat(self, user_msg: str, system_msg: str = "You are a careful assistant. Answer with a single word or short phrase.") -> torch.Tensor:
        msgs = [{"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}]
        ids = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids):  # transformers v5 returns a BatchEncoding
            ids = ids["input_ids"]
        return ids.to(self.device)
