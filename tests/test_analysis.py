"""Guards for the methodological corrections in csa.analysis.

These are the claims the report leans on, so they are tested on synthetic
frames where the right answer is known by construction.
"""

import numpy as np
import pandas as pd

from csa.analysis import _h4_block, _h4_within_budget


def _reqs(n_per_budget=20, budgets=(0.5, 0.25, 0.125), seed=0,
          divergence_predicts=True, n_unanswerable=0):
    rng = np.random.default_rng(seed)
    rows = []
    for kf in budgets:
        for i in range(n_per_budget):
            flip = rng.random() * (1.0 - kf)
            correct = (rng.random() > flip * 1.5) if divergence_predicts \
                else (rng.random() > 0.4)
            rows.append(dict(task_id=f"t{kf}-{i}", family="multi_entity",
                             keep_frac=kf, method="quest_topk",
                             flip_frac=flip, mean_kl=flip * 2,
                             mean_est_dropped=flip * 0.8,
                             correct=bool(correct), dense_correct=True))
    for j in range(n_unanswerable):
        kf = budgets[j % len(budgets)]
        rows.append(dict(task_id=f"u{j}", family="multi_hop", keep_frac=kf,
                         method="quest_topk",
                         flip_frac=rng.random() * (1.0 - kf),
                         mean_kl=rng.random(), mean_est_dropped=rng.random(),
                         correct=False, dense_correct=False))
    return pd.DataFrame(rows)


def test_h4_detects_real_relationship():
    d = _reqs(divergence_predicts=True)
    r = _h4_block(d)
    assert r["spearman_flipfrac_correct"] < -0.3
    assert r["auc_flipfrac_incorrect"] > 0.65


def test_h4_null_when_no_relationship():
    d = _reqs(divergence_predicts=False, seed=4)
    r = _h4_block(d)
    assert abs(r["spearman_flipfrac_correct"]) < 0.3


def test_floor_effect_dilutes_and_conditioning_recovers():
    """The core justification for reporting the answerable subset: adding
    requests the dense model already fails must weaken the pooled estimate,
    while the conditional estimate stays intact."""
    d = _reqs(divergence_predicts=True, n_unanswerable=120, seed=2)
    pooled = abs(_h4_block(d)["spearman_flipfrac_correct"])
    answerable = abs(_h4_block(d[d["dense_correct"] == True])  # noqa: E712
                     ["spearman_flipfrac_correct"])
    assert answerable > pooled, (
        f"conditioning on headroom should not weaken the estimate "
        f"(answerable {answerable:.3f} vs pooled {pooled:.3f})")


def test_within_budget_kills_budget_confound():
    """If divergence only tracks correctness because BUDGET drives both, the
    within-budget correlation must vanish even though the pooled one is large."""
    rng = np.random.default_rng(7)
    rows = []
    for kf, acc in [(0.5, 0.9), (0.25, 0.6), (0.125, 0.2)]:
        for i in range(30):
            rows.append(dict(task_id=f"t{kf}-{i}", family="multi_entity",
                             keep_frac=kf, method="quest_topk",
                             # divergence depends ONLY on budget, not on the
                             # per-request outcome
                             flip_frac=(1.0 - kf) + 0.001 * rng.random(),
                             mean_kl=1.0 - kf, mean_est_dropped=1.0 - kf,
                             correct=bool(rng.random() < acc),
                             dense_correct=True))
    d = pd.DataFrame(rows)
    pooled = _h4_block(d)["spearman_flipfrac_correct"]
    within = _h4_within_budget(d)["weighted_mean_spearman"]
    assert pooled < -0.3, "pooled correlation should look strong"
    assert abs(within) < 0.25, (
        f"within-budget must expose the confound (got {within})")


def test_within_budget_preserves_genuine_signal():
    d = _reqs(divergence_predicts=True, n_per_budget=40, seed=5)
    within = _h4_within_budget(d)
    assert within["n_budgets_estimable"] >= 2
    assert within["weighted_mean_spearman"] < -0.2


def test_h4_block_handles_degenerate_input():
    d = _reqs(n_per_budget=1, budgets=(0.5,))
    assert "note" in _h4_block(d)


def test_h4_reports_unmeasurable_rather_than_falsified_without_headroom():
    """No answerable requests must yield "cannot estimate", not a correlation.

    If the task suite is too hard for the model, every request has
    dense_correct=False, sparse execution cannot degrade anything, and H4 is
    UNTESTABLE. That is a different verdict from H4 being FALSIFIED, and the
    two must not be confusable: one says the experiment failed to test the
    hypothesis, the other says the hypothesis lost. A gate decision rests on
    telling them apart.
    """
    d = _reqs(n_per_budget=20)
    d["dense_correct"] = False
    d["correct"] = False
    answerable = d[d["dense_correct"] == True]  # noqa: E712
    assert len(answerable) == 0

    block = _h4_block(answerable)
    assert "note" in block, "must decline to estimate, not emit a number"
    assert block["n_requests"] == 0
    assert "spearman_flipfrac_correct" not in block

    within = _h4_within_budget(answerable)
    assert within["n_budgets_estimable"] == 0
    assert np.isnan(within["weighted_mean_spearman"])


def test_h4_declines_when_every_answerable_request_is_correct():
    """The mirror case: no variation in the label means no estimable rho."""
    d = _reqs(n_per_budget=20)
    d["correct"] = True
    block = _h4_block(d)
    assert "note" in block
    assert "spearman_flipfrac_correct" not in block
