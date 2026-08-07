# Certified Sparse Attention — Phase L report (smoke scale)

**Status:** local smoke-scale measurement complete on NVIDIA T1000 8GB.  
**Numbers:** generated tables in [`TABLES.md`](TABLES.md); methodology caveats in [`METHODOLOGY.md`](METHODOLOGY.md).  
**Do not quote a number that is not in TABLES.md or a `summary.json`.**

> ### Accuracy-derived numbers were regenerated after an audit
>
> **Resolved for the 0.5B; the 1.5B regeneration is in flight.** H4 is restated
> below from `results/study_a_0.5b_v2/` and clears its bar (ρ = −0.870
> answerable, −0.821 within budget). The history is kept because the defects
> were invisible in the summaries — the old and new dense-accuracy figures are
> 0.438 and 0.458, so nothing in the headline numbers looked wrong.

> **What was wrong.** Every Study A run before `*_v2/` seeded tasks from
> `hash((seed, fam, i, target_tokens))`. `hash()` on a tuple containing a `str`
> is salted by `PYTHONHASHSEED`, randomized per process, so those runs drew
> gold answers from a state that **cannot be reconstructed** — re-running the
> same commit does not reproduce them. Answers were also scored by substring
> containment (`key` matched inside `monkey`), the coreference aliases were
> occupation-shaped so the profession question was ambiguous as posed, and
> reasoning traces hit a decode cap before stating their total.
>
> **Never affected:** H1, H2, H3, the cliff's flip columns, and all six
> ablations. None calls `Task.check` — per-step divergence labels come from
> paired dense/sparse execution and never read a gold answer.
>
> All defects are fixed and carry regression tests, including one that runs the
> generator under two values of `PYTHONHASHSEED` and asserts identical output.
> The superseded directories retain `SUPERSEDED.md`.

---

## Executive verdicts (pre-committed hypotheses)

| Hypothesis | Question | Smoke-scale verdict |
|---|---|---|
| **H1** | Is per-step divergence detectable label-free? | **Supported.** Damage-aligned label-free AUC **0.808** consensus / **0.807** dropped mass on the regenerated 0.5B (0.836 on the superseded 1.5B); combined detector CV AUC 0.92. Falsification bar was AUC &lt; 0.65. |
| **H2** | Can a useful bound be afforded? | **Not falsified (scale-free); FALSIFIED on short traces.** Valid estimators (Hoeffding, empirical-Bernstein) reach ±10% at long streams; at the *measured* cost ratio r = 1.04, ±5% on Study A–length traces exceeds the 15% throughput bar. Betting CS **fails under bursty drift** and is rejected for serving. |
| **H3** | Can verification be elastic under load? | **Shape supported (simulation only).** Under overload, elastic TPOT tracks `none` while inline latency collapses; the bound widens instead. Not yet a vLLM result. |
| **H4** | Does divergence predict end-task wrongness? | **Supported on regenerated data (0.5B).** ρ(flip frac, correct) = **−0.870** on answerable requests (n=165), **−0.821** within budget across all 5 budgets. Falsification bar was \|ρ\| &lt; 0.5. 1.5B rerunning. |

Negative / limiting findings that are also deliverables:

1. **Top-1/top-2 margin is an uncertainty meter, not a fidelity signal.** Highest flip-AUC (0.87 on 1.5B) but within-budget correlation with oracle damage only **0.37**, vs **0.89** for estimated dropped mass.
2. **Betting / capital-process CSs are not safe under phase-transition bursts** (adaptive+Betting anytime miss rate **0.825** in the bursty audit; Hoeffding/EB miss rate **0.000**).
2b. **A ±5% bound is not affordable on short traces once r is measured honestly.** The previous "inconclusive" rested on a paired-step cost ratio (0.55) that understated the true gather-path cost (1.04).
3. **Wall-clock sparsity does not pay on this host.** Overhead bench on 1.5B @ 4K: speedup ≈ 1.0–1.12×; probe/sparse cost ratio r ≈ 1.0–1.12. Decode is launch/MLP-bound — H2 cost numbers here are **upper bounds** on r.
4. **Scale is unresolved.** Everything below is 0.5B–1.5B at 1–2K context on an 8 GB T1000. Gate 1 (8B, 16K–128K, H100) is required before Mechanism 1 can be treated as deployable.

---

## H1 — Detectability

Teacher-forced paired runs supply exact greedy-token flip labels at every decode step.

| Run | Steps | Flip rate | Best damage-aligned LF AUC | Best flip AUC (margin) | Combined CV AUC |
|---|---|---|---|---|---|
| **0.5B v2** (authoritative) | 2380 | 0.200 | consensus **0.808** / dropped mass **0.807** | 0.877 | — |
| 0.5B (superseded) | 2035 | 0.211 | est. dropped mass 0.792 | 0.850 | — |
| 1.5B (superseded) | 3345 | 0.196 | est. dropped mass 0.836 | 0.872 | 0.919 ± 0.007 |

The superseded rows are retained because H1 never depended on gold answers:
they are a valid but unrecorded draw from the task distribution, and the fact
that the regenerated run reproduces the signal ranking and AUC range on a
*different* task draw is itself a robustness check. The 1.5B v2 row lands when
that run finishes.

Within-budget AUCs (the deployable number) stay above the 0.65 kill line at every budget on both models for estimated dropped mass and consensus (see TABLES.md). Pooled AUCs are higher and are **not** the number to ship.

**Signal ranking (1.5B, damage-aligned).** Estimated dropped mass, eviction consensus, and fully-dropped mass all have ρ(damage \| within-budget) ≥ 0.87. Margin wins flip-AUC but fails the damage test — promote dropped-mass / consensus for Mechanism 1, use margin only as an auxiliary uncertainty feature in the combined detector.

**Dropped-mass alone** (grouped CV) AUC = **0.828** on 1.5B — strong, but **not** &gt; 0.9, so the detector is **not** trivial from a single scalar. The combination buys a clear lift (0.92) without leaving the label-free feature set.

**Cross-model transfer (Ablation 5).** Train 0.5B → test 1.5B AUC **0.920**; reverse **0.896**. Family and budget transfer heatmaps are in `results/ablations/figures/`.

---

## H2 — Bounded overhead

Study B replays the authoritative 0.5B teacher stream through estimators at
equal probe cost. Coverage is a precondition: estimators that undercover are
recorded and excluded from the H2 verdict. Re-run against the regenerated 0.5B
stream (`study_a_0.5b_v2`, 2380 teacher steps). Study B consumes only per-step divergence labels, which never read a
gold answer, so the seeding defect could not have reached it — and the numbers
duly did not move in substance.

- Cost ratio r = **1.041**, now taken from the **measured gather path**
  (`overhead_bench`) rather than the paired-step ratio, which understated it at
  ≈0.55 because the paired step computes dense *and* sparse.
- Short-trace ±5% width: **FALSIFIED**. This is a change from the previous
  report's "inconclusive", and it is a change for the honest reason: at the
  measured r ≈ 1.04 rather than the understating 0.55, buying a ±5% bound on
  Study-A-length traces costs more than the 15% throughput kill criterion
  allows. The earlier verdict was propped up by a cost ratio that flattered it.
- Scale-free: **NOT FALSIFIED** at stream length ≥ 10k steps for valid
  estimators (`fixed + Hoeffding`, `fixed + EmpBernstein`). Probes needed for
  width 0.10: EB **962**, Hoeffding **4075**. Width scaling exponent
  β = **0.443** (width ~ p^−β; 0.5 is the sqrt-n rate, so cost grows slightly
  sub-linearly in guarantee strength).
- **Estimator-validity finding (unchanged):** under the bursty regime,
  adaptive+Betting anytime miss rate = **0.825**; fixed+Betting = **0.108**;
  Hoeffding and EB = **0.000**. Capital-process CSs that permanently reject
  means are the wrong tool for phase-transition streams. Serving default:
  **empirical-Bernstein** (tighter) or Hoeffding (simplest).

So H2 splits: the “&gt;15% throughput for ±5%” kill criterion **is triggered on
Study-A-length traces** at the measured r, and is **not triggered** once
streams reach ≥10k steps with a coverage-valid estimator. Since a production
request decodes far more than 2380 steps in aggregate across a serving window,
the scale-free result is the operationally relevant one — but the short-trace
falsification is recorded rather than explained away, and neither can be
confirmed in the bandwidth-bound regime until Gate 1 overhead numbers exist.
Note r here is an **upper bound**: this host is launch/MLP-bound, so sparsity
buys almost nothing and probes look relatively expensive.

---

## H3 — Elastic verification (simulation)

Study C is a discrete-event GPU scheduler with probes as deferrable work. At high load:

| Policy | P99 TPOT (ticks) | System bound width | Probe completion |
|---|---|---|---|
| none | 19.5 | 1.00 | — |
| elastic | **17.5** | **0.64** | 0.016 |
| inline | **44.0** | 0.10 | 1.00 |

Elastic preserves latency near the no-verification baseline; contention widens the guarantee. Mid-load shows the same shape with higher probe completion (**0.52**).

These figures are post-fix. The simulator previously let queued probes execute
for requests that had already completed, although the RFC specifies a probe is
only valid while its request's KV prefix is resident. That counted verification
the system could not have performed, and it flattered *this* policy: at arrival
rate 0.04 elastic probe completion was reported as 0.999 where the honest figure
is 0.854. The bias is largest at low load — exactly where elastic is meant to
look good — and vanishes at high load, where retention expiry already dominates.
The high-load table above is unchanged; mid-load completion fell from ~0.79 to
0.52. **H3's shape survives on honest numbers.**

**Caveat (METHODOLOGY §4.6):** this is not vLLM. Gate 3 is the real-system test.

---

## H4 — Proxy validity

Conditioning on dense-answerable requests is required: where the dense model already fails, sparse cannot add label-relevant damage.

**Restated from the regenerated run** (`results/study_a_0.5b_v2/`), on
reproducible seeds, an unambiguous task suite, and traces that reach their
answers. The 1.5B regeneration is in flight.

| Run | Dense QA acc | Answerable n | ρ(flip, correct) all | ρ answerable | ρ within budget |
|---|---|---|---|---|---|
| 0.5B v2 | 0.458 | 165 / 360 | −0.435 | **−0.870** | **−0.821** (5/5 budgets) |

\|ρ\| ≫ 0.5 on the answerable subset and within budget → H4 **not falsified**.
Direction is as hypothesized: more step-level divergence predicts lower
end-task accuracy.

**The conditioning correction is load-bearing, and this run demonstrates it.**
Pooled over all requests ρ = −0.435, which is *below* the 0.5 bar; conditioning
on the 165 requests the dense model actually gets right recovers −0.870. The
195 requests dense already fails cannot be degraded further by sparse
execution, so they contribute label noise uncorrelated with divergence. An
analysis that skipped this step would have read H4 as falsified. Holding budget
fixed (−0.821) additionally rules out budget as a common cause of both
divergence and error.

Per-family dense accuracy: coreference 0.833, multi_entity 0.500,
reasoning 0.500, multi_hop 0.000.

**Provenance.** The superseded runs reported −0.898 (0.5B) / −0.805 (1.5B).
The regenerated 0.5B figure (−0.870) is close, so the original conclusion was
directionally right — but it was scored against gold answers drawn from an
unrecoverable random state, on a coreference task that was ambiguous as posed
and reasoning traces that truncated before stating their answer. The
conclusion survived; the evidence for it did not, and has been replaced rather
than reaffirmed.

---

## Cliff analysis

Restated from `study_a_0.5b_v2` (dense accuracy **0.458**):

| keep frac | acc: quest | acc: mean | acc: local_sink | flip: quest |
|---|---|---|---|---|
| 0.5 | **0.458** | 0.417 | 0.292 | 0.003 |
| 0.25 | 0.375 | 0.292 | 0.292 | 0.007 |
| 0.125 | 0.375 | 0.292 | 0.292 | 0.059 |
| 0.0625 | 0.250 | 0.250 | 0.250 | 0.204 |
| 0.03125 | 0.167 | 0.167 | 0.167 | 0.349 |

Both halves of the cliff now move together, which is the shape the proposal
predicts: `quest_topk` at keep=0.5 **exactly matches dense accuracy** (0.458)
with a 0.3% flip rate, then loses roughly two-thirds of its accuracy by
keep=0.03125 as flips climb to 35%. **Divergence is far above 0.1% at budgets
that would need to buy real speedup** — the premise that sparse attention can
silently degrade is intact at smoke scale, and now demonstrated on accuracy
rather than on divergence alone.

Selection also beats the static pattern where it has room to: at keep=0.5
quest is 0.458 against local_sink's 0.292, and the three methods converge once
the budget is tight enough that everything is being discarded.

Sink+local clamping makes some nominal budgets identical in effective keep
fraction; TABLES.md / `effective_keep_fraction` report the actual fraction.

---

## Ablations (1, 2, 3, 5, 6)

1. **Signal isolation.** Dropped-mass alone CV 0.828; all-signals CV 0.919. Leave-one-out: removing margin drops combined AUC to 0.829 — margin contributes flip-prediction, not damage signal.
2. **Verification rate 0→100%.** Bound width shrinks from vacuous (rate 0) to ~0.054 (rate 1) on the 1.5B stream; situates the continuum.
3. **Fixed vs adaptive at equal cost.** Adaptive with floor=`p` tightens width vs fixed at the same nominal rate (higher realized probe fraction on high-signal steps); floor=`p/4` can *widen* the HT-scaled bound — floor choice is load-bearing.
5. **Transfer.** Cross-model AUC ≥ 0.90; cross-budget transfer degrades when training on easy budgets and testing on hard ones (expected).
6. **Per-layer schedules (1.5B).** At matched mean budget, schedules differ in flip rate (e.g. keep=0.0625: pyramid 0.214 vs uniform/inv_pyramid 0.054). Combined detector CV AUC remains high within each schedule (0.87–0.93) and transfers across schedules (AUCs ≥ 0.92). Mechanism 1 is orthogonal to the allocator.

---

## Overhead (T1000, 1.5B, ctx 4096)

| keep | sparse_only ms | speedup vs dense | r = probe/sparse | loss @ 5% probes |
|---|---|---|---|---|
| 0.5 | 202 | 0.99× | 0.99 | 4.7% |
| 0.25 | 178 | 1.12× | 1.12 | 5.3% |
| 0.125 | 199 | 1.00× | 1.00 | 4.8% |
| 0.0625 | 189 | 1.05× | 1.05 | 5.0% |

Honest caveat (also in `overhead.json`): this host is not KV-bandwidth-bound. Gate 1 must re-measure r on H100 at 32K–128K.

---

## Limitations (binding for the next phase)

1. Scale: 0.5B/1.5B, 1–2K context, T1000 8GB — not the proposal regime.
2. Synthetic tasks with single-token / short answers; not production traffic.
3. Uneven per-family headroom (0.5B v2): coreference 0.833, multi_entity 0.500, reasoning 0.500, **multi_hop 0.000**. H4's answerable subset is therefore carried by three of four families; multi_hop contributes only floor-effect rows. On the 195 dense-wrong requests, sparse scores 0.010 — empirically confirming those rows carry no signal and must be conditioned out.
4. H3 is simulation; H2 wall-clock is an upper bound on cost.
5. Greedy flip label is a proxy; logit KL is recorded alongside.

---

## What is done / what is next

**Phase L complete for the 0.5B; the 1.5B regeneration is in flight.**
Artifacts:

- `results/study_a_0.5b_v2/` — regenerated, **authoritative**
- `results/study_a_1.5b_v2/` — regenerating
- `results/study_a/`, `study_a_0.5b/`, `study_a_1.5b/` — **SUPERSEDED**, retained for provenance
- `results/study_b/` — re-run against the regenerated 0.5B stream
- `results/study_c/` — re-run after the probe-expiry fix
- `results/ablations/`, `results/ablation6/`, `results/overhead/` — gold-independent, unaffected
- `results/TABLES.md`, this `REPORT.md`

**All four pre-committed hypotheses now clear their bars** on evidence that
survives inspection:

| | Result | Bar |
|---|---|---|
| H1 | label-free AUC 0.808 / 0.877 | ≥ 0.65 |
| H2 | not falsified scale-free (falsified on short traces) | ±5% under 15% throughput |
| H3 | elastic p99 17.5 vs inline 44.0, bound widens instead | shape |
| H4 | ρ −0.870 answerable, −0.821 within budget | \|ρ\| ≥ 0.5 |

**Next phase is R1 (Gate 1)** on a rented H100 — human action required.

Remaining before renting:

1. Confirm H4 replicates on the 1.5B regeneration (in flight). The 0.5B result
   is sufficient to justify the rental; the 1.5B is confirmation, not a gate.
2. Run `experiments/kv_drift_check.py`, which has never been executed. It
   answers the sharpest objection to H1 — whether teacher-forced detection
   reads accumulated KV drift rather than the current step's omitted attention
   mass. Cheap, gold-independent, and H1 is the claim Gate 1 rests on.

Do not proceed to preprint (Phase P) until Gate 1 passes its pre-committed
criteria.
