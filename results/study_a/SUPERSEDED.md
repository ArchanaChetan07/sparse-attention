# Superseded run

This directory holds the **first** full Study A sweep. It was produced by an
intermediate revision of the harness (commit `5101383` era): before the
gather-based production sparse path, per-layer budget schedules, the
exactly-isolated attribution mode, and the effective-budget reporting.

It was originally superseded by `results/study_a_0.5b/` (commit `7adae8d`).
Headline numbers agreed in direction across both runs (best label-free AUC
0.80 here vs 0.85 there; same signal ranking; same cliff shape), which remains
a useful robustness note — the methodology fixes sharpened, not created, the
effect. Those are flip-based quantities and are gold-independent, so the
comparison still holds.

**`results/study_a_0.5b/` has since been superseded in turn**, by the
`PYTHONHASHSEED` seeding defect described in its own `SUPERSEDED.md`. This
directory shares that defect: it predates the fix, so its accuracy-derived
quantities are invalid for the same reason and its gold answers are equally
unrecoverable. The **authoritative run is `results/study_a_0.5b_v2/`**.

Kept for provenance per METHODOLOGY.md discipline: measurement runs are never
silently replaced.
