"""Synthetic long-context tasks with verifiable answers.

Families follow the hard cases of arXiv 2603.01426 / The Sparse Frontier:
multi-entity tracking, multi-hop chains, coreference — deliberately NOT
needle-in-a-haystack alone (proposal §7). Facts are spread across procedurally
generated filler prose so answer-critical tokens sit at controlled depths, and
each task has a single-word gold answer for exact checking.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

NAMES = ["Alice", "Bob", "Carla", "Deepak", "Elena", "Farid", "Grace", "Hiro",
         "Ines", "Jonas", "Kavya", "Liam", "Mona", "Nadia", "Omar", "Priya"]
ITEMS = ["key", "coin", "map", "lantern", "compass", "ledger", "medal",
         "ribbon", "flask", "whistle", "locket", "quill"]
PLACES = ["attic", "cellar", "garden", "harbor", "library", "market",
          "mill", "orchard", "station", "tower"]
PROFESSIONS = ["architect", "baker", "chemist", "diver", "engineer",
               "florist", "geologist", "historian", "jeweler", "locksmith"]
ALIASES = ["the falcon", "the compass", "the lantern-bearer", "the archivist",
           "the navigator", "the cartographer"]

_FILLER_SUBJECTS = ["The village council", "A visiting merchant", "The old ferry",
                    "The northern road", "The clock tower", "A summer storm",
                    "The weekly market", "The river barge", "The stone bridge",
                    "The lighthouse keeper", "The apple harvest", "The night watch"]
_FILLER_VERBS = ["was discussed at length", "drew a small crowd", "needed repairs again",
                 "changed little over the years", "kept its usual schedule",
                 "surprised nobody in town", "was postponed until spring",
                 "remained a favorite topic", "carried on as before",
                 "made the rounds of local gossip"]
_FILLER_TAILS = ["despite the weather.", "according to the elders.",
                 "as recorded in the town ledger.", "much to everyone's relief.",
                 "though opinions differed.", "before the season turned.",
                 "with little fanfare.", "as it had for decades."]


@dataclass
class Task:
    family: str
    prompt: str
    gold: str
    task_id: str
    meta: dict = field(default_factory=dict)

    def check(self, generated: str) -> bool:
        return self.gold.lower() in generated.lower()


def _filler_paragraph(rng: random.Random, n_sentences: int = 4) -> str:
    return " ".join(
        f"{rng.choice(_FILLER_SUBJECTS)} {rng.choice(_FILLER_VERBS)} {rng.choice(_FILLER_TAILS)}"
        for _ in range(n_sentences))


def _assemble(rng: random.Random, facts: list[str], question: str,
              count_tokens, target_tokens: int,
              answer_hint: str = "Answer with a single word.") -> str:
    """Interleave facts with filler until the prompt reaches target length.

    answer_hint is part of the rendered prompt so the length guarantee holds
    for the text actually used; stripping it afterwards would leave the prompt
    short of target.
    """
    segments: list[str] = []
    # start with a filler, then alternate fact / filler
    segments.append(_filler_paragraph(rng))
    for f in facts:
        segments.append(f)
        segments.append(_filler_paragraph(rng))

    def render():
        tail = f"\n{answer_hint}" if answer_hint else ""
        return ("Read the following account carefully.\n\n"
                + "\n\n".join(segments)
                + f"\n\nQuestion: {question}{tail}")

    # pad with more filler (inserted at random interior gaps, never after the
    # last fact's trailing filler) until long enough
    while count_tokens(render()) < target_tokens:
        pos = rng.randrange(0, len(segments) - 1)
        segments.insert(pos, _filler_paragraph(rng))
    return render()


def multi_entity(rng: random.Random, count_tokens, target_tokens: int,
                 n_entities: int = 6, task_id: str = "") -> Task:
    names = rng.sample(NAMES, n_entities)
    items = rng.sample(ITEMS, n_entities)
    places = rng.sample(PLACES, min(n_entities, len(PLACES)))
    facts = [f"{n} keeps the {i} in the {places[k % len(places)]}."
             for k, (n, i) in enumerate(zip(names, items))]
    rng.shuffle(facts)
    q_idx = rng.randrange(n_entities)
    question = f"Which item does {names[q_idx]} keep?"
    prompt = _assemble(rng, facts, question, count_tokens, target_tokens)
    return Task("multi_entity", prompt, items[q_idx], task_id,
                {"n_entities": n_entities})


def multi_hop(rng: random.Random, count_tokens, target_tokens: int,
              hops: int = 3, task_id: str = "") -> Task:
    chain = rng.sample(NAMES, hops + 1)
    item = rng.choice(ITEMS)
    facts = [f"{chain[i]} handed the {item} to {chain[i + 1]}."
             for i in range(hops)]
    question = f"Who ended up holding the {item}?"
    prompt = _assemble(rng, facts, question, count_tokens, target_tokens)
    return Task("multi_hop", prompt, chain[-1], task_id, {"hops": hops})


def coreference(rng: random.Random, count_tokens, target_tokens: int,
                task_id: str = "") -> Task:
    name = rng.choice(NAMES)
    prof = rng.choice(PROFESSIONS)
    alias = rng.choice(ALIASES)
    place = rng.choice(PLACES)
    facts = [
        f"{name}, who works as a {prof}, lives near the {place}.",
        f"Around town, {name} is better known as {alias}.",
    ]
    question = f"What is the profession of the person known as {alias}?"
    prompt = _assemble(rng, facts, question, count_tokens, target_tokens)
    return Task("coreference", prompt, prof, task_id, {})


def longform(rng: random.Random, count_tokens, target_tokens: int,
             task_id: str = "") -> Task:
    """Long-form continuation: no gold answer, used for longer decode traces."""
    names = rng.sample(NAMES, 3)
    facts = [f"{names[0]} met {names[1]} at the {rng.choice(PLACES)} to plan "
             f"the {rng.choice(ITEMS)} exhibition.",
             f"{names[2]} promised to bring the {rng.choice(ITEMS)}."]
    question = ("Summarize the account above in two or three sentences, "
                "mentioning every person named.")
    prompt = _assemble(rng, facts, question, count_tokens, target_tokens,
                       answer_hint="")
    return Task("longform", prompt, "", task_id, {})


def reasoning(rng: random.Random, count_tokens, target_tokens: int,
              task_id: str = "") -> Task:
    """Multi-step arithmetic over facts scattered in context.

    Requires an explicit worked answer, so decode traces are long — the regime
    the proposal flags as dominating cost and deciding outcomes earliest.
    Answer-critical tokens are separated by filler, so a budget that globally
    evicts any one of them breaks the chain.
    """
    people = rng.sample(NAMES, 3)
    counts = [rng.randrange(3, 20) for _ in people]
    item = rng.choice(ITEMS)
    facts = [f"{p} brought {c} copies of the {item} to the exchange."
             for p, c in zip(people, counts)]
    rng.shuffle(facts)
    total = sum(counts)
    question = (f"Work through it step by step, then state how many copies of "
                f"the {item} were brought in total by {people[0]}, "
                f"{people[1]}, and {people[2]}.")
    prompt = _assemble(rng, facts, question, count_tokens, target_tokens,
                       answer_hint="End with the total as a number.")
    return Task("reasoning", prompt, str(total), task_id,
                {"counts": counts, "total": total})


FAMILIES = {
    "multi_entity": multi_entity,
    "multi_hop": multi_hop,
    "coreference": coreference,
    "reasoning": reasoning,
    "longform": longform,
}


def make_tasks(count_tokens, target_tokens: int, per_family: int = 3,
               families=None, seed: int = 0) -> list[Task]:
    families = families or list(FAMILIES)
    tasks = []
    for fam in families:
        for i in range(per_family):
            rng = random.Random(hash((seed, fam, i, target_tokens)) & 0xFFFFFFFF)
            t = FAMILIES[fam](rng, count_tokens, target_tokens,
                              task_id=f"{fam}-{target_tokens}-{i}")
            tasks.append(t)
    return tasks
