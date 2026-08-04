import random

from csa.tasks import FAMILIES, make_tasks

# crude tokenizer stand-in: whitespace words are within ~1.4x of BPE tokens
count = lambda s: int(len(s.split()) * 1.35)


def test_all_families_generate_and_selfcheck():
    for fam, fn in FAMILIES.items():
        t = fn(random.Random(0), count, 400, task_id=f"{fam}-t")
        assert t.family == fam
        assert len(t.prompt) > 200
        if t.gold:
            # the gold answer must be derivable: it appears in the context for
            # extraction tasks, or is computed (reasoning) — check() must work
            assert t.check(f"the answer is {t.gold}.")
            assert not t.check("zzzz-not-the-answer")


def test_prompt_reaches_target_length():
    for target in (256, 1024):
        for t in make_tasks(count, target, per_family=1, seed=3):
            assert count(t.prompt) >= target, f"{t.task_id} too short"


def test_reasoning_gold_is_the_true_sum():
    t = FAMILIES["reasoning"](random.Random(11), count, 300)
    assert int(t.gold) == sum(t.meta["counts"])
    assert t.meta["total"] == int(t.gold)


def test_tasks_are_deterministic_per_seed():
    a = make_tasks(count, 300, per_family=2, seed=5)
    b = make_tasks(count, 300, per_family=2, seed=5)
    assert [t.prompt for t in a] == [t.prompt for t in b]
    assert [t.gold for t in a] == [t.gold for t in b]


def test_distinct_seeds_give_distinct_tasks():
    a = make_tasks(count, 300, per_family=1, seed=1)
    b = make_tasks(count, 300, per_family=1, seed=2)
    assert [t.prompt for t in a] != [t.prompt for t in b]
