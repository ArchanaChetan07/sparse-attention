# Certified Sparse Attention

**Runtime-verified fidelity guarantees for long-context LLM serving** — a
working implementation of all three mechanisms of the proposal, with a paired
dense/sparse measurement harness, anytime-valid statistical verification, an
elastic verification scheduler, and the ablations that answer the obvious
reviewer objections.

Sparse attention ships an accuracy claim it never verifies at runtime. Because
fidelity loss under KV compression behaves like a phase transition rather than
a smooth slope, operating points chosen offline from benchmark averages are
wrong for an unknown subset of live requests — silently. This repo implements
the machinery to change that: a serving-time answer to *"is this response
degraded?"*, with a measured confidence bound.

```
"<= 2% of decode steps diverged from dense execution, at 95% confidence,
 at 6% throughput cost."
```

## The three mechanisms

| Mechanism | Idea | Code |
|---|---|---|
| **1. Label-free detection** | Selection-based sparse attention already scores every KV block and discards the non-top-k scores. That discarded mass is a free estimate of what attention is throwing away; cross-head eviction consensus captures the "globally erased" failure mode the phase-transition literature identifies as causal. | [`csa/sparse.py`](csa/sparse.py), [`csa/signals.py`](csa/signals.py), [`csa/detector.py`](csa/detector.py) |
| **2. Sampled dense verification** | Occasionally re-execute one decode step with dense attention on identical KV state (a *probe*), and feed the outcome into an anytime-valid confidence sequence — a bound legitimate at every step with no pre-committed sample size. Adaptive sampling steered by Mechanism 1 stays unbiased via Horvitz-Thompson weighting. | [`csa/verify.py`](csa/verify.py) |
| **3. Verification as elastic work** | Probes are deferrable and batchable, so schedule them into slack GPU capacity. Under load the system widens its confidence interval instead of degrading latency or silently losing fidelity: contention degrades the *guarantee*, not the output. | [`csa/scheduler.py`](csa/scheduler.py) |

## The measurement substrate

[`csa/paired.py`](csa/paired.py) registers a custom attention function with HF
Transformers (`AttentionInterface`), so any supported model runs with **paired
dense/sparse execution**: at every decode step the dense output and the
selection-based sparse output are computed from the *same* Q/KV state, along
with per-layer signals and divergence. Selection (not eviction) keeps the full
cache, so the dense counterfactual is exact — this is what makes the
measurement clean, and it is why Study A is built on Transformers rather than
vLLM.

Two attention paths exist deliberately, and a test asserts they agree:

- **masking path** (`sparse` mode) — computes dense and sparse together for
  measurement. Correct, but the full matmul still runs, so it says nothing
  about cost.
- **gather path** (`sparse_only` mode) — gathers only selected blocks, so work
  is proportional to the budget. This is what gets timed.

Dense probes run as a second forward of the same token followed by a cache
crop, so they leave no trace on the trajectory — asserted by the smoke check,
because if it were false every divergence number here would be invalid.

Sparse methods (all selection-based, block granularity, per head):
`quest_topk` (min/max key pooling, upper-bound block scores), `mean_topk`
(mean-pooled block scores), `local_sink` (static sink+local pattern; produces
no block scores, hence no label-free signal — included as the contrast case).
Per-layer budget schedules (`uniform`, `pyramid`, `inv_pyramid`) are
budget-matched by construction so a schedule comparison is never accidentally
a budget comparison.

## Studies and ablations

| Study | Question | Driver |
|---|---|---|
| **A** | H1: is divergence detectable label-free? H4: does divergence predict end-task wrongness? | [`study_a_smoke.py`](experiments/study_a_smoke.py) |
| **B** | H2: what bound width per unit of verification cost, and which estimator wins? | [`study_b_estimators.py`](experiments/study_b_estimators.py) |
| **C** | H3: can verification be displaced under load without violating SLOs? | [`study_c_scheduler.py`](experiments/study_c_scheduler.py) |
| **Ablations 1,2,3,5** | signals alone vs combined; verification rate 0→100%; fixed vs adaptive at equal cost; detector transfer | [`ablations.py`](experiments/ablations.py) |
| **Ablation 6** | composition with per-layer budget allocators | [`ablation6_layer_budget.py`](experiments/ablation6_layer_budget.py) |
| **Overhead** | the probe/step cost ratio H2 is actually stated in | [`overhead_bench.py`](experiments/overhead_bench.py) |

Tasks are synthetic long-context probes with verifiable answers — multi-entity
tracking, multi-hop chains, coreference, and multi-step reasoning — chosen
because they exercise the failure modes the literature identifies, and
deliberately not NIAH-style retrieval alone.

## Run it

```bash
python -m pip install -e .
```

```bash
python -m pytest tests -q
```

```bash
python experiments/smoke_check.py
```

```bash
powershell experiments/run_all_study_a.ps1
```

```bash
python experiments/analyze_study_a.py results/study_a_0.5b results/study_a_1.5b
```

```bash
python experiments/study_b_estimators.py
```

```bash
python experiments/study_c_scheduler.py
```

```bash
python experiments/ablations.py
```

```bash
python experiments/overhead_bench.py
```

```bash
python experiments/make_tables.py > results/TABLES.md
```

Use `python -m pip`, not bare `pip`: on machines with more than one Python
(an Anaconda `python` plus a store-installed `pip`, say) bare `pip` can install
into a different interpreter than the one that runs the code.

Default model is `Qwen/Qwen2.5-0.5B-Instruct`; pass `--model` to scale up.
`--quick` runs a reduced Study A sweep. Results land in `results/<study>/` as
CSV + `summary.json` + figures, each with a machine fingerprint (GPU, driver,
power limit, PCIe link, library versions). Raw per-step attention tensors are
never written — only online-aggregated scalars.

Analysis is deliberately decoupled from the sweeps
([`csa/analysis.py`](csa/analysis.py)): sweeps are expensive and run at
different times, and cross-run comparisons only mean something if every run is
analysed by identical code. Re-run it any time without re-running a sweep.

## Results and how to read them

- [`results/REPORT.md`](results/REPORT.md) — verdicts against the
  pre-registered hypotheses H1–H4.
- [`results/TABLES.md`](results/TABLES.md) — generated tables (never
  hand-transcribed).
- [`results/METHODOLOGY.md`](results/METHODOLOGY.md) — how the measurements
  are constructed and the six ways they could mislead. **Read this before
  quoting any number**; it was written before the results were interpreted.
- [`docs/RFC-runtime-fidelity-verification.md`](docs/RFC-runtime-fidelity-verification.md)
  — the serving-system integration design this implies.

Three methodological corrections are applied rather than left to the reader,
because each changes the conclusion:

1. **AUC pooled across budgets is inflated** — budget predicts divergence and
   a server knows its own budget, so the deployable number is within-budget.
2. **H4 needs headroom** — on requests the dense model already fails, sparse
   execution cannot degrade anything, so those rows are label noise.
3. **Cross-validation must be grouped by request** — steps within a request
   share a prompt and a KV trajectory; an i.i.d. split leaks and inflates.

## Status vs the full proposal

Implemented and measured here: the complete mechanism set, the measurement
harness, Studies A/B/C at smoke scale, and the ablation suite.

Still pending, and explicitly out of reach of this hardware: the rented-GPU
phase — 8B–72B models at 16K–128K contexts, the full baseline grid, and the
in-tree vLLM implementation of Study C. Every result here is on 0.5B–1.5B
models at 1–2K contexts on an 8 GB T1000, which is *not* the regime where
sparse attention pays; see the threats-to-validity section of
[`METHODOLOGY.md`](results/METHODOLOGY.md).
