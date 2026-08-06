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


def test_aliases_are_not_profession_shaped():
    """A coreference alias must not be answerable as a profession.

    Regression: aliases were agent nouns ("the archivist", "the navigator"),
    so "what is the profession of the person known as the archivist?" had a
    defensible wrong answer and the model gave it. That measures task
    ambiguity, not retrieval.
    """
    from csa.tasks import ALIASES, PROFESSIONS
    for a in ALIASES:
        bare = a.removeprefix("the ").strip()
        assert bare not in PROFESSIONS, f"alias {a!r} is itself a profession"
        # agent-noun endings are what made the old set answerable-as-a-job
        assert not bare.endswith(("ist", "er", "or", "ian", "-bearer")), (
            f"alias {a!r} is occupation-shaped; pick a non-agentive nickname")


def test_reasoning_answer_fits_in_the_long_decode_budget():
    """The trace must REACH the total, or the family scores an intermediate
    count. 4 of 6 dense traces previously truncated at a 64-token cap."""
    from csa.tasks import LONG_DECODE_MIN_TOKENS
    assert LONG_DECODE_MIN_TOKENS >= 96
    t = FAMILIES["reasoning"](random.Random(4), count, 300)
    # a terse but complete answer must both fit the budget and score
    complete = ("Omar: 5. Hiro: 7. Kavya: 3. Total: %s" % t.gold)
    assert count(complete) < LONG_DECODE_MIN_TOKENS
    assert t.check(complete)
    # a trace truncated before the total must NOT score as correct
    truncated = "Omar brought 5 copies. Hiro brought 7 copies. Kavya brought 3"
    assert not t.check(truncated), "truncated trace must not count as correct"


def test_distinct_seeds_give_distinct_tasks():
    a = make_tasks(count, 300, per_family=1, seed=1)
    b = make_tasks(count, 300, per_family=1, seed=2)
    assert [t.prompt for t in a] != [t.prompt for t in b]
