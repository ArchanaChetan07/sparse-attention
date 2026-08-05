import os
import random
import subprocess
import sys

from csa.tasks import FAMILIES, Task, make_tasks

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


def test_check_rejects_substring_false_positives():
    assert not Task("x", "p", "key", "i").check("the monkey took it")
    assert Task("x", "p", "key", "i").check("the key was found")
    assert not Task("x", "p", "12", "i").check("The total is 112.")
    assert Task("x", "p", "12", "i").check("steps... total 12")
    # numeric gold uses the LAST integer (final-answer position)
    assert not Task("x", "p", "15", "i").check("brought 15 copies; total 30")
    assert Task("x", "p", "30", "i").check("brought 15 copies; total 30")


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


def test_task_seeds_stable_across_pythonhashseed():
    """Regression: hash()-based seeds changed under PYTHONHASHSEED."""
    code = (
        "from csa.tasks import make_tasks\n"
        "c=lambda s: int(len(s.split())*1.35)\n"
        "t=make_tasks(c,300,per_family=1,families=['multi_entity'],seed=5)\n"
        "print(t[0].gold+'|'+t[0].prompt[:40])\n"
    )
    outs = []
    for hs in ("0", "1"):
        env = {**os.environ, "PYTHONHASHSEED": hs}
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=env, check=True)
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1], outs


def test_distinct_seeds_give_distinct_tasks():
    a = make_tasks(count, 300, per_family=1, seed=1)
    b = make_tasks(count, 300, per_family=1, seed=2)
    assert [t.prompt for t in a] != [t.prompt for t in b]
