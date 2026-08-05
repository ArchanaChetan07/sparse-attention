# Superseded run

This directory holds the **first** full Study A sweep. It was produced by an
intermediate revision of the harness (commit `5101383` era): before the
gather-based production sparse path, per-layer budget schedules, the
exactly-isolated attribution mode, and the effective-budget reporting.

The **authoritative smoke-scale run is `results/study_a_0.5b/`**, regenerated
with the final code (commit `7adae8d`). Headline numbers agree in direction
across both runs (best label-free AUC 0.80 here vs 0.85 there; same signal
ranking; same cliff shape), which is itself a useful robustness note — the
methodology fixes sharpened, not created, the effect.

Kept for provenance per METHODOLOGY.md discipline: measurement runs are never
silently replaced.
