# RFC: Runtime fidelity verification for sparse attention

**Status:** draft for community discussion
**Scope:** an optional, off-by-default serving feature that reports a
statistical bound on how far sparse-attention output has diverged from dense
execution, and schedules the work that produces that bound into slack capacity.

## Summary

Serving systems that use sparse attention ship an accuracy claim they cannot
check at runtime. The claim is inherited by citation from algorithm papers
evaluated offline on benchmark averages. Because fidelity loss under KV
compression behaves like a phase transition rather than a smooth decay, a
single statically chosen operating point is wrong for an unknown subset of
live requests, and nothing in the serving loop notices.

This RFC proposes that a serving system be able to answer, for a live request:

> "At most 2% of decode steps diverged from what dense attention would have
> produced, at 95% confidence, at 6% throughput cost."

No deployed system can currently say anything of this kind.

## Motivation

Three properties of sparse attention make runtime verification tractable, and
they are the reason this is a serving-layer feature rather than an algorithm
change:

1. **The detection signal is already computed and thrown away.** Every
   selection-based method scores KV blocks and keeps the top-k. The scores of
   the *discarded* blocks are a label-free estimate of the attention mass being
   omitted. Retaining a scalar summary of them costs approximately nothing.
2. **Ground truth is one dense pass away.** Because selection retains the full
   cache, any step can be re-executed with dense attention over identical
   state. That is an exact observation of divergence at the cost of one dense
   forward.
3. **That verification work is deferrable and batchable.** A probe does not
   block the token it verifies. It can be queued and run when the GPU has
   slack, which makes it schedulable rather than a fixed tax.

Together these give a design where, under contention, *the confidence interval
widens rather than the output quality silently degrading*.

## Design

### Component 1 — divergence detector (attention path)

At decode, a selection-based backend already produces per-block scores and a
selected set. The detector reduces these to a small fixed set of per-step
scalars, aggregated online across layers and heads:

- estimated attention mass in dropped blocks (mean and max over heads),
- cross-head eviction consensus: mass-weighted fraction of heads that dropped
  each block, and the fraction of mass in blocks dropped by *every* head —
  the label-free analogue of the globally-evicted failure mode,
- top-1/top-2 logit margin and attention entropy.

Cost: a reduction over quantities already in registers. No extra attention
work, no extra memory traffic over the KV cache. Raw per-block scores are
never materialized to global memory or logged.

Interface sketch:

```python
class DivergenceDetector(Protocol):
    def observe(self, layer: int, block_scores, selected_mask) -> None: ...
    def step_summary(self) -> dict[str, float]: ...   # called once per step
```

### Component 2 — verification pass (schedulable unit)

A probe is a request-scoped unit: *re-run decode step t of request r with
dense attention over the retained KV prefix, compare the resulting greedy
token (and logit distribution) with what sparse execution produced.*

Two constraints make it schedulable rather than immediate:

- It is only valid while the KV prefix for step *t* is still resident. Probes
  therefore carry an expiry and are dropped if the request's blocks are
  evicted or the request completes.
- It must not perturb the served trajectory. In the reference implementation
  the probe runs on a cloned cache view and its writes are discarded.

Probes accumulate into a per-request (and per-tenant) **anytime-valid
confidence sequence** over the diverged-step fraction. Anytime validity is the
crucial property: the bound is legitimate at every step without committing in
advance to how many probes will be affordable, which is exactly the situation
a scheduler creates. Sampling may be steered by the detector, in which case
inverse-propensity weighting keeps the estimate unbiased.

### Component 3 — elastic scheduling policy

Probes enter a deferred queue and are admitted only from capacity left after
decode. Priority is by widest current bound (max-min tightness), optionally
weighted by request value. The intended degradation shape:

| Load | No verification | Inline verification | Elastic verification |
|---|---|---|---|
| low | fast, unverified | fast, verified | fast, verified |
| high | fast, unverified | **latency degrades** | fast, **bound widens** |

The scheduler contribution is the abstraction — verification as elastic work —
plus the allocation problem it induces: distribute a finite probe budget
across concurrent requests to minimize worst-case fidelity uncertainty subject
to latency SLOs.

## What this composes with

This is deliberately not a competing sparsity method. It sits on top of
whatever budget allocator a deployment already uses — per-layer budgets,
per-head allocation, load-driven buffer elasticity. Those systems decide *how
much* to compress; this one reports *how much that cost*. An allocator can
consume the bound as a control signal, which is a natural follow-on but not
required for the first version.

## Staging

1. **Detector only**, behind a flag, exporting per-step signals as metrics.
   Useful immediately for observability; no scheduling changes; no probes.
2. **Fixed-rate probing** with a confidence sequence, reported per request.
   Verification cost is a configured constant.
3. **Elastic scheduling** of probes into slack capacity, with the allocation
   policy above.

Stage 1 is independently valuable and independently reviewable, which is the
intended path to landing this incrementally.

## Open questions

- **Where does the detector hook live** so it works across attention backends
  without constraining kernel choice? A post-selection callback is the least
  invasive option, but it must not force a synchronization.
- **Is per-deployment calibration required?** If the detector's operating
  threshold does not transfer across models or task mixes, that is an
  operational cost that must be quantified and stated, not hidden. Measured as
  an ablation in the reference implementation.
- **What is the right SLO contract?** Options: report the bound and let the
  operator act; or let the operator specify a target bound width and have the
  scheduler treat verification as a first-class demand.
- **Multi-tenant accounting.** If bound tightness becomes a billable tier,
  probe allocation becomes an economic mechanism-design problem rather than
  purely a scheduling one.
- **Cost in the bandwidth-bound regime.** The probe/step cost ratio determines
  everything about affordability and must be measured on hardware where sparse
  attention actually pays.

## Prior art and relationship to it

Systems that adapt sparsity at runtime — per-layer/per-token budget allocators,
adaptive block sizes, per-head budget balancing, load-driven buffer elasticity
— all *adapt*, and none *verify*: they report throughput, TTFT, TPOT and cache
hit rate, not any measure of fidelity relative to dense execution. The
analysis literature that identifies KV-compression failure as a structural
phase transition provides diagnostics rather than a serving mechanism, and its
central metric requires knowing which tokens were answer-critical — that is,
it needs labels and is not computable at serving time. This RFC's detector is
the label-free surrogate for that quantity, and the sampled dense probe is
what turns a heuristic into a bound.

## Reference implementation

A working out-of-tree implementation of all three components, with a paired
dense/sparse measurement harness, anytime-valid estimators, an elastic
scheduling simulator, and the ablations that answer the obvious objections,
is available for review. Its stated limitation is scale: results are at small
model sizes and short contexts, and the regime where sparse attention pays
requires larger hardware than the reference measurements used.
