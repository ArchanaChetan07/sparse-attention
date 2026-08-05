# Methodology and threats to validity

This file records how the measurements are constructed and, more importantly,
the ways they could mislead. It is written before the results are interpreted
so that the caveats are not chosen after seeing which ones are convenient.

## 1. What is measured

**The paired counterfactual.** A custom attention function computes, at every
decode step, both the dense attention output and the selection-based sparse
output from the *same* Q/KV state. Sparsity is applied by masking non-selected
blocks; the full KV cache is retained. Selection rather than eviction is
load-bearing: eviction destroys the state that the dense counterfactual would
need, so a paired comparison would no longer be on identical inputs.

**The divergence label.** A step is labelled *diverged* when the greedy argmax
token under dense attention differs from the greedy argmax under sparse
attention, given the same cache. This is exact, cheap, and requires no task
labels — but it is a proxy for "wrong", which is why H4 exists as a separate
hypothesis rather than an assumption.

**The dense probe.** A probe re-executes one token with dense attention on the
retained cache, then crops the cache back so the probe leaves no trace. The
smoke check asserts that a run with probes produces a token-identical
trajectory to a run without them; if that assertion ever fails, every
divergence number in this repo is invalid.

## 2. Execution modes

| Mode | Token trajectory | Purpose |
|---|---|---|
| free-running | sparse model's own greedy tokens | production-like; includes compounding |
| teacher-forced | forced onto the dense trajectory | isolates per-step attention error from token-choice compounding |

Both are reported. Teacher-forcing removes compounding of *token choice* but
not of *KV state*: the cache at step *t* was still written by sparse-attention
hidden states at steps < *t*. So the teacher-forced number isolates one of the
two compounding channels, not both. This is stated because it is easy to
overclaim "no compounding" here.

## 3. Corrections applied, and why

**Pooling across budgets inflates AUC.** Budget predicts divergence strongly,
so a detector evaluated on steps pooled across budgets gets credit for
information a serving system already has (it knows its own budget). The
deployable number is the within-budget AUC, and it is lower. Both are reported.

**H4 needs headroom.** On a request the dense model already answers wrongly,
sparse execution cannot degrade the answer further, so such requests
contribute label noise uncorrelated with divergence. Conditioning on
`dense_correct` is not cherry-picking — it is the population the hypothesis is
about. The unconditional number is reported alongside so the effect of the
conditioning is visible.

**H4 needs budget held fixed.** A negative correlation between divergence and
correctness can be produced entirely by budget acting as a common cause of
both. The within-budget correlation is therefore reported; if it collapses
while the pooled one is large, the pooled one is an artifact.
`tests/test_analysis.py` constructs exactly that artifact and asserts the
within-budget estimate detects it.

**Cross-validation must be grouped.** Steps within one request share a prompt,
a budget, and a KV trajectory. Splitting them i.i.d. leaks a request across
train and test folds and inflates the combined detector's AUC. Folds are split
by request group, and a noise-only control asserts the CV returns ~0.5.

**Nominal budget is not actual budget.** Sink and local blocks are always
retained, so a nominal keep fraction below that floor is clamped upward. At
short contexts two distinct nominal budgets can therefore be the *same* actual
budget, at which point the sparse methods collapse to identical selections and
appear — misleadingly — to agree. The harness records the true kept-token
fraction per step, and the analysis reports the nominal-to-effective mapping
plus an explicit list of clamped configurations. Read the cliff against the
effective fraction, not the nominal one.

**A flip predictor is not automatically a fidelity signal.** A step whose
top-1/top-2 margin is small flips under *any* perturbation, so a
confidence-based signal can score well at predicting flips while carrying no
information about what sparse attention actually discarded. Selecting the
detector by AUC alone would therefore ship an uncertainty meter — one that is
blind precisely to the dangerous case, a confidently wrong step caused by
globally evicted content. Every signal is therefore reported with *both* its
AUC for flips and its correlation with the oracle damage measures
(`signal_vs_damage`), computed within budget so budget cannot drive both.
Signals that predict flips but not damage are named as such rather than
promoted for their AUC.

**Validity gates width, not the other way round.** When comparing estimators,
the cheapest one reaching a target bound width is meaningless unless that
interval actually covers. Selecting on width alone picks whichever estimator
undercovers most aggressively, which would reproduce precisely the failure
this project exists to eliminate — a confident-looking claim that is wrong.
Estimators must pass a coverage audit before they may support a verdict, and
those rejected are recorded rather than dropped silently. Relatedly, a capital
-process CS that has rejected every candidate mean has *failed* rather than
converged; it reports a vacuous interval and a failure flag instead of a
zero-width one.

**Bound width is set by probe count, not probe rate.** Width falls as
1/sqrt(n_probes). On a short trace no sampling rate reaches a tight bound, so
quoting a rate there would measure trace length rather than verification cost.
Cost is therefore reported scale-free — probes needed for a target width — and
only then translated into a rate at realistic stream lengths. A corollary
worth stating plainly: a *short single request* cannot be certified tightly at
any affordable cost; tight guarantees are a property of long requests or of
aggregated per-tenant streams.

## 4. Threats to validity, in descending order of seriousness

1. **Scale.** Results are on 0.5B and 1.5B models at 1–2K contexts on an 8 GB
   T1000. The proposal's regime is 8B–72B at 16K–128K. The phase-transition
   literature this work builds on reports cliffs at those scales; whether the
   detector's AUC holds there is unverified and is the first thing the rented
   GPU phase must check.
2. **Task set.** Synthetic, procedurally generated tasks with single-token
   gold answers. They were designed to require the failure modes the
   literature identifies (multi-entity tracking, multi-hop chains,
   coreference, multi-step arithmetic), but they are not the natural
   distribution of long-context traffic.
3. **Small answerable subsets.** Where the base model cannot do a task family
   at all, that family contributes no usable H4 signal. Point estimates on the
   answerable subset can rest on few distinct tasks; the number of requests
   behind every conditional estimate is reported alongside it.
4. **Wall-clock overhead is not measured in the regime that matters.** Sparse
   attention pays off when decode is KV-bandwidth-bound. On a small model and
   a small card, decode is dominated by kernel launches and MLP time, so the
   measured sparsity speedup is a lower bound and the probe/step cost ratio is
   an upper bound. H2 is therefore reported as a function of that ratio rather
   than at a single point.
5. **The divergence label is binary and greedy.** A flipped token can be
   semantically harmless (a synonym) and an unflipped step can still be on a
   path to a wrong answer. Logit KL is recorded alongside as a graded measure.
6. **Scheduler results are simulation, not a serving system.** Study C is a
   discrete-event model with an assumed probe cost. It establishes that the
   elastic policy has the claimed *shape* of degradation; it does not
   establish that vLLM can be made to do this at the claimed cost.

## 5. Reproducibility

Every result directory carries a `*.meta.json` with the machine fingerprint
(GPU, driver, power limit, PCIe link, library versions) and the run
configuration. Analysis is decoupled from the sweep (`csa/analysis.py`) so
that runs performed at different times are analysed by identical code; re-run
it with `experiments/analyze_study_a.py`. Raw per-step attention tensors are
never written — only online-aggregated per-step scalars — per the recording
discipline the proposal specifies.

**Run provenance.** Measurement runs are never silently replaced. When a
harness change invalidates a run, the old directory stays in place with a
`SUPERSEDED.md` naming the replacement and the code delta. Current authority:
`results/study_a/` (first sweep, intermediate code) is superseded by
`results/study_a_0.5b/` (final code); the direction-level agreement between
the two (signal ranking, cliff shape, AUC 0.80 vs 0.85) is a robustness check
on the methodology fixes themselves.
