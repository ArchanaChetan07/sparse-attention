# SUPERSEDED — do not cite accuracy-derived numbers from this run

This run was produced at commit `5117fe4`, before `1094a03` fixed task seeding
and answer checking. Two defects in that code make part of this result set
invalid, and one of them makes it **unreproducible even by checking out the
old commit**.

## Defect 1 — task seeds were salted (fatal, unrecoverable)

```python
rng = random.Random(hash((seed, fam, i, target_tokens)) & 0xFFFFFFFF)
```

`hash()` on a tuple containing a `str` is salted by `PYTHONHASHSEED`, which
CPython randomizes per process. Three consecutive interpreters give three
different seeds:

```
3579268514    3634147791    356980889
```

So the gold answers behind this run were drawn from an unrecorded random
state and **cannot be reconstructed**. Re-running `5117fe4` does not
reproduce them either. Fixed in `1094a03` with a `blake2b` digest; guarded by
`tests/test_tasks.py::test_task_seeds_stable_across_pythonhashseed`.

## Defect 2 — answer check matched substrings

```python
def check(self, generated): return self.gold.lower() in generated.lower()
```

`key` matched inside `monkey`; numeric golds matched intermediate operands.
Fixed in `1094a03` to use word boundaries and last-integer extraction.

## What is invalid

Everything downstream of `correct` / `dense_correct`:

- `dense_qa_accuracy` (0.5 here — inflated by substring false positives)
- `h4_all_requests`, `h4_answerable`, `h4_answerable_within_budget`
  (reported ρ = −0.805 answerable; **H4 is a pre-committed gate criterion**)
- `h4_floor_effect`
- the `acc:*` columns of the fidelity-cliff table

## What still stands

Per-step divergence labels come from paired dense/sparse execution and never
touch the gold answer, so these are structurally unaffected:

- `auc_teacher_flip`, `auc_within_budget`, `signal_vs_damage` (best label-free
  AUC 0.872, above the 0.65 falsification threshold → H1 supported)
- the `flip:*` columns of the cliff table
- all timing and overhead measurements

The caveat is reproducibility, not validity: these were measured on a valid
but unrecorded draw from the task distribution, so the conclusion holds while
exact re-execution does not.

Superseded by the regenerated run on the fixed codebase.
