# Certified Sparse Attention — Phase L report (smoke scale)

**Status:** local smoke-scale measurement complete on NVIDIA T1000 8GB.  
**Authority:** `results/study_a_0.5b/`, `results/study_a_1.5b/` (supersedes `results/study_a/`).  
**Numbers:** generated tables in [`TABLES.md`](TABLES.md); methodology caveats in [`METHODOLOGY.md`](METHODOLOGY.md).  
**Do not quote a number that is not in TABLES.md or a `summary.json`.**

---

## Executive verdicts (pre-committed hypotheses)

| Hypothesis | Question | Smoke-scale verdict |
|---|---|---|
| **H1** | Is per-step divergence detectable label-free? | **Supported.** Best damage-aligned label-free AUC ≈ 0.84 (1.5B est. dropped mass); combined detector CV AUC 0.92. Falsification bar was AUC &lt; 0.65. |
| **H2** | Can a useful bound be afforded? | **Not falsified (scale-free), inconclusive on short traces.** Valid estimators (Hoeffding, empirical-Bernstein) reach ±10% width at long streams; ±5% not reached on Study A–length traces. Betting CS **fails under bursty drift** and is rejected for serving. |
| **H3** | Can verification be elastic under load? | **Shape supported (simulation only).** Under overload, elastic TPOT tracks `none` while inline latency collapses; the bound widens instead. Not yet a vLLM result. |
| **H4** | Does divergence predict end-task wrongness? | **Supported on answerable requests.** Spearman(flip frac, correct) = **−0.90** (0.5B) / **−0.80** (1.5B); within-budget (1.5B) **−0.75**. Falsification bar was \|ρ\| &lt; 0.5. |

Negative / limiting findings that are also deliverables:

1. **Top-1/top-2 margin is an uncertainty meter, not a fidelity signal.** Highest flip-AUC (0.87 on 1.5B) but within-budget correlation with oracle damage only **0.37**, vs **0.89** for estimated dropped mass.
2. **Betting / capital-process CSs are not safe under phase-transition bursts** (adaptive+Betting anytime miss rate **0.97** in the bursty audit; Hoeffding/EB miss rate **0.0**).
3. **Wall-clock sparsity does not pay on this host.** Overhead bench on 1.5B @ 4K: speedup ≈ 1.0–1.12×; probe/sparse cost ratio r ≈ 1.0–1.12. Decode is launch/MLP-bound — H2 cost numbers here are **upper bounds** on r.
4. **Scale is unresolved.** Everything below is 0.5B–1.5B at 1–2K context on an 8 GB T1000. Gate 1 (8B, 16K–128K, H100) is required before Mechanism 1 can be treated as deployable.

---

## H1 — Detectability

Teacher-forced paired runs supply exact greedy-token flip labels at every decode step.

| Run | Steps | Flip rate | Best damage-aligned LF AUC | Best flip AUC (margin) | Combined CV AUC |
|---|---|---|---|---|---|
| 0.5B | 2035 | 0.211 | est. dropped mass **0.792** | 0.850 | — |
| 1.5B | 3345 | 0.196 | est. dropped mass **0.836** | 0.872 | **0.919 ± 0.007** |

Within-budget AUCs (the deployable number) stay above the 0.65 kill line at every budget on both models for estimated dropped mass and consensus (see TABLES.md). Pooled AUCs are higher and are **not** the number to ship.

**Signal ranking (1.5B, damage-aligned).** Estimated dropped mass, eviction consensus, and fully-dropped mass all have ρ(damage \| within-budget) ≥ 0.87. Margin wins flip-AUC but fails the damage test — promote dropped-mass / consensus for Mechanism 1, use margin only as an auxiliary uncertainty feature in the combined detector.

**Dropped-mass alone** (grouped CV) AUC = **0.828** on 1.5B — strong, but **not** &gt; 0.9, so the detector is **not** trivial from a single scalar. The combination buys a clear lift (0.92) without leaving the label-free feature set.

**Cross-model transfer (Ablation 5).** Train 0.5B → test 1.5B AUC **0.920**; reverse **0.896**. Family and budget transfer heatmaps are in `results/ablations/figures/`.

---

## H2 — Bounded overhead

Study B replays the authoritative 0.5B teacher stream through estimators at equal probe cost. Coverage is a precondition: estimators that undercover are recorded and excluded from the H2 verdict.

- Paired-step cost ratio r ≈ **0.55** (understates true r). Overhead bench (production sparse path) gives r ≈ **1.0–1.12** on this host.
- Short-trace ±5% width: **INCONCLUSIVE** (not reached by any coverage-valid estimator).
- Scale-free: **NOT FALSIFIED** at stream length ≥ 10k steps for valid estimators (`fixed + Hoeffding`, `fixed + EmpBernstein`). EB needs ~1050 probes for width 0.10; Hoeffding ~4075.
- **Estimator-validity finding:** under the bursty regime, adaptive+Betting anytime miss rate = **0.967**; fixed+Betting = **0.108**; Hoeffding and EB = **0.0**. Capital-process CSs that permanently reject means are the wrong tool for phase-transition streams. Serving default: **empirical-Bernstein** (tighter) or Hoeffding (simplest).

H2’s “&gt;15% throughput for ±5%” kill criterion is therefore **not triggered** at smoke scale once invalid estimators are excluded; it also **cannot be confirmed** in the bandwidth-bound regime until Gate 1 overhead numbers exist.

---

## H3 — Elastic verification (simulation)

Study C is a discrete-event GPU scheduler with probes as deferrable work. At high load:

| Policy | P99 TPOT (ticks) | System bound width | Probe completion |
|---|---|---|---|
| none | 19.5 | 1.00 | — |
| elastic | **17.5** | **0.64** | 0.016 |
| inline | **44.0** | 0.10 | 1.00 |

Elastic preserves latency near the no-verification baseline; contention widens the guarantee. Mid-load shows the same shape with higher probe completion (~0.79). **Caveat (METHODOLOGY §4.6):** this is not vLLM. Gate 3 is the real-system test.

---

## H4 — Proxy validity

Conditioning on dense-answerable requests is required: where the dense model already fails, sparse cannot add label-relevant damage.

| Run | Dense QA acc | Answerable n | ρ(flip, correct) answerable | ρ within budget |
|---|---|---|---|---|
| 0.5B | 0.438 | 105 | **−0.898** | (see summary; small per-budget n) |
| 1.5B | 0.500 | 180 | **−0.805** | **−0.751** |

\|ρ\| ≫ 0.5 on both models → H4 **not** falsified. Direction is as hypothesized: more step-level divergence predicts lower end-task accuracy. Floor-effect counts are in each run’s `summary.json` / TABLES.md.

---

## Cliff analysis

Flip fraction and oracle dropped mass fall monotonically as `keep_frac` increases; end-task accuracy recovers. At the tightest budgets (0.03125) teacher flip rates are ~45–47%; at 0.5 they are ~3–5%. **Divergence is far above 0.1% at budgets that would need to buy real speedup** — the premise that sparse can silently degrade is intact at smoke scale. Sink+local clamping makes some nominal budgets identical in effective keep fraction; TABLES.md / `effective_keep_fraction` report the actual fraction.

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
3. Small answerable subsets for some families (e.g. multi_hop dense accuracy 0 on 1.5B).
4. H3 is simulation; H2 wall-clock is an upper bound on cost.
5. Greedy flip label is a proxy; logit KL is recorded alongside.

---

## What is done / what is next

**Phase L complete** with artifacts:

- `results/study_a_0.5b/`, `results/study_a_1.5b/`
- `results/study_b/`, `results/study_c/`
- `results/ablations/`, `results/ablation6/`, `results/overhead/`
- `results/TABLES.md`, this `REPORT.md`

**Next phase is R1 (Gate 1)** on a rented H100 — human action required (see session stop note). Do not proceed to preprint (Phase P) until Gate 1 passes its pre-committed criteria.
