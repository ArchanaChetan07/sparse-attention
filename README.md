# Certified Sparse Attention

**Runtime-verified fidelity guarantees for long-context LLM serving** — working
implementation of the full proposal at smoke scale: measurement harness,
label-free divergence detection, statistically sound sampled verification, and
elastic verification scheduling.

Sparse attention ships an accuracy claim it never verifies at runtime. Because
fidelity loss under KV compression behaves like a phase transition rather than
a smooth slope, operating points selected offline from benchmark averages are
wrong for an unknown subset of live requests — silently. This repo implements
the machinery to change that: a serving-time answer to *"is this response
degraded?"* with a measured confidence bound.

```
"≤ 2% of decode steps diverged from dense execution, at 95% confidence,
 at 6% throughput cost."
```

## The three mechanisms

| Mechanism | Idea | Code |
|---|---|---|
| **1. Label-free detection** | Selection-based sparse attention already scores every KV block and discards the non-top-k scores. The discarded mass is a free estimate of what attention is throwing away. Cross-head eviction consensus captures the "globally erased" failure mode. | [`csa/sparse.py`](csa/sparse.py), [`csa/signals.py`](csa/signals.py) |
| **2. Sampled dense verification** | Occasionally re-execute one decode step with dense attention on identical KV state (a *probe*), compare, and feed the Bernoulli outcome into an anytime-valid confidence sequence — a bound on the diverged-step fraction that is legitimate at every step, with no pre-committed sample size. Adaptive sampling steered by Mechanism 1 uses Horvitz-Thompson weighting to stay unbiased. | [`csa/verify.py`](csa/verify.py) |
| **3. Verification as elastic work** | Probes are deferrable and batchable, so schedule them into slack GPU capacity. Under load the system widens its confidence interval instead of degrading latency or silently dropping fidelity: contention degrades the *guarantee*, not the output. | [`csa/scheduler.py`](csa/scheduler.py) |

## The measurement substrate

[`csa/paired.py`](csa/paired.py) registers a custom attention function with HF
Transformers (`AttentionInterface`), so any supported model runs with **paired
dense/sparse execution**: at every decode step the dense output and the
selection-based sparse output are computed from the *same* Q/KV state, along
with per-layer signals and divergence. Selection (not eviction) keeps the full
cache, so the dense counterfactual is exact — this is what makes the
measurement clean (proposal §7: build Study A on Transformers, not vLLM).

Dense probes are implemented as a second forward of the same token followed by
a cache crop, so they leave no trace on the trajectory (verified by test).

Sparse methods implemented (all selection-based, block granularity, per-head):
- `quest_topk` — Quest-style per-block min/max key pooling, upper-bound scores
- `mean_topk` — mean-pooled block scores (MInference-flavored proxy)
- `local_sink` — StreamingLLM-style static sink+local pattern (no scores → no
  label-free signal; included as the pattern-method contrast)

## Studies

| Study | Question | Driver |
|---|---|---|
| **A** | H1: is divergence detectable label-free? H4: does divergence predict end-task wrongness? | [`experiments/study_a_smoke.py`](experiments/study_a_smoke.py) |
| **B** | H2: what bound width per unit of verification GPU time? Which estimator wins? | [`experiments/study_b_estimators.py`](experiments/study_b_estimators.py) |
| **C** | H3: can verification be displaced under load without SLO violation? | [`experiments/study_c_scheduler.py`](experiments/study_c_scheduler.py) |

Tasks are synthetic long-context probes with verifiable single-word answers —
multi-entity tracking, multi-hop chains, coreference (the hard cases from the
phase-transition literature), deliberately not NIAH-style retrieval alone.

## Run it

```bash
pip install -e .
python -m pytest tests -q                    # 23 tests, CPU-only, ~10 s
python experiments/smoke_check.py            # model integration check (GPU)
python experiments/study_a_smoke.py          # paired sweep -> results/study_a
python experiments/study_b_estimators.py     # replays Study A traces
python experiments/study_c_scheduler.py      # scheduler simulation
```

Default model is `Qwen/Qwen2.5-0.5B-Instruct` (fits an 8 GB card with room to
spare); pass `--model` to scale up. `--quick` runs a reduced Study A sweep.

Results land in `results/<study>/` as CSV + `summary.json` + figures, each
with a machine fingerprint (GPU, driver, power limit, library versions) per
the reproducibility discipline in the proposal (Appendix B: aggregate online,
never dump raw attention tensors).

## Findings at smoke scale

See [results/REPORT.md](results/REPORT.md) for the current results against the
pre-registered hypotheses H1–H4.

## Status vs the full proposal

This is the Week-3/4 deliverable of the execution plan (paired harness, all
five candidate signals, multiple methods/budgets/context lengths on a cheap
GPU) plus working implementations of Studies B and C at simulation scale.
Out of scope here: 8B–32B models at 16K–128K contexts, vLLM integration
(Study C proper), and the full baseline grid — those need the rented-GPU
phases of the proposal.
