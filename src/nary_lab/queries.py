"""Layer 2a — what can each representation be *asked*?

Fidelity measures what a representation stores. This measures what it can
answer. The two come apart: a representation can hold enough information in
principle and still make a question awkward or impossible to express.

Five question types, chosen because each isolates a different capability. Each
is phrased in plain biological terms first, then formally.

``co_membership``
    *Which other proteins sit in a complex together with both A and B?*
    The basic higher-order question. A pairwise graph can only answer the
    weaker "which proteins are connected to both A and B", which is not the
    same thing and generally over-answers.

``exclusion``
    *How many complexes contain A but not B?*
    Requires complexes to be countable objects.

``group_size``
    *How many complexes have at least k subunits?*
    Requires the group to have an inspectable size.

``exact_composition``
    *Is there a complex made of exactly these proteins?*
    Requires group identity, not just connectivity.

``group_lookup``
    *Which complexes is protein A part of?*
    Requires groups to be addressable at all.

Scoring rules
-------------
1. Each answerer may use only what its own representation contains. No
   borrowing from the source data.
2. A question a representation cannot express scores **0** and is reported.
   "You cannot ask this" is a finding, not a missing value.
3. Precision and recall are reported separately, because a lossy
   representation nearly always *over*-answers, and over-answering and
   under-answering carry different risks.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .model import GroupCollection
from .representations import Reified, build_dissolved, build_reified

__all__ = ["QueryResult", "run_query_suite", "QUERY_FAMILIES"]

INEXPRESSIBLE = None


@dataclass
class QueryResult:
    family: str
    representation: str
    precision: float
    recall: float
    f1: float
    expressible: bool
    n_instances: int
    note: str = ""


def _prf(pred: set[str] | None, gold: set[str]) -> tuple[float, float, float]:
    if pred is None:
        return 0.0, 0.0, 0.0
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return p, r, f


def _sample_pairs(coll: GroupCollection, rng: np.random.Generator,
                  n: int, min_size: int = 3) -> list[tuple[str, str]]:
    """Pairs drawn from inside real groups, so gold answers are non-trivial.

    Falls back to size-2 groups when nothing larger exists. Data in which every
    group is a pair is the degenerate case for this whole comparison, and the
    suite should still run on it and report the (uninteresting) result rather
    than failing.
    """
    cands = [g for g in coll if g.size >= min_size]
    if not cands:
        cands = [g for g in coll if g.size >= 2]
    if not cands:
        return []
    out = []
    for _ in range(n):
        g = cands[int(rng.integers(len(cands)))]
        a, b = rng.choice(np.array(sorted(g.members), dtype=object), size=2, replace=False)
        out.append((str(a), str(b)))
    return out


# --------------------------------------------------------------------------- #
# Families
# --------------------------------------------------------------------------- #
def q_co_membership(coll, G, R, rng, n) -> list[QueryResult]:
    rows = []
    acc = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
    pairs = _sample_pairs(coll, rng, n)
    for a, b in pairs:
        gold: set[str] = set()
        for gid in coll.groups_of(a) & coll.groups_of(b):
            gold |= coll.groups[gid].members
        gold -= {a, b}

        # pairwise graph: common neighbours -- the standard approximation
        if a in G and b in G:
            pred_d = (set(G.neighbors(a)) & set(G.neighbors(b))) - {a, b}
        else:
            pred_d = set()
        # hub node: exact, by intersecting group memberships
        pred_r: set[str] = set()
        for gid in R.groups_of(a) & R.groups_of(b):
            pred_r |= R.members_of(gid)
        pred_r -= {a, b}
        # hyperedge: exact, by construction
        pred_n = set(gold)

        for key, pred in (("dissolved", pred_d), ("reified", pred_r), ("native", pred_n)):
            p, r, f = _prf(pred, gold)
            acc[key][0] += p
            acc[key][1] += r
            acc[key][2] += f
    for key, (p, r, f) in acc.items():
        n_i = max(len(pairs), 1)
        rows.append(QueryResult("co_membership", key, p / n_i, r / n_i, f / n_i,
                                True, len(pairs)))
    return rows


def q_exclusion(coll, G, R, rng, n) -> list[QueryResult]:
    rows = []
    pairs = _sample_pairs(coll, rng, n)
    hits = {"reified": 0, "native": 0}
    for a, b in pairs:
        gold = len([gid for gid in coll.groups_of(a)
                    if b not in coll.groups[gid].members])
        pred_r = len([gid for gid in R.groups_of(a) if b not in R.members_of(gid)])
        hits["reified"] += int(pred_r == gold)
        hits["native"] += 1  # exact by construction
    n_i = max(len(pairs), 1)
    rows.append(QueryResult("exclusion", "dissolved", 0.0, 0.0, 0.0, False, len(pairs),
                            "complexes are not objects in a pairwise graph"))
    for k in ("reified", "native"):
        acc = hits[k] / n_i
        rows.append(QueryResult("exclusion", k, acc, acc, acc, True, len(pairs)))
    return rows


def q_group_size(coll, G, R, rng, n, k: int = 4) -> list[QueryResult]:
    gold = sum(1 for g in coll if g.size >= k)
    pred_r = sum(1 for gid in R.group_nodes if len(R.members_of(gid)) >= k)
    rows = [QueryResult("group_size", "dissolved", 0.0, 0.0, 0.0, False, 1,
                        "no object whose size could be inspected")]
    for key, pred in (("reified", pred_r), ("native", gold)):
        acc = 1.0 if pred == gold else 0.0
        rows.append(QueryResult("group_size", key, acc, acc, acc, True, 1))
    return rows


def q_exact_composition(coll, G, R, rng, n) -> list[QueryResult]:
    """Does a complex with exactly this composition exist?

    Half the probes are real compositions, half are perturbed ones that should
    *not* be found — otherwise a representation that answers 'yes' to everything
    would score perfectly.
    """
    truth = coll.unique_member_sets()
    groups = [g for g in coll if g.size >= 3] or [g for g in coll if g.size >= 2]
    members = coll.members()
    probes: list[tuple[frozenset, bool]] = []
    if not groups:
        return [QueryResult("exact_composition", r, 0.0, 0.0, 0.0, False, 0,
                            "no groups available to probe")
                for r in ("dissolved", "reified", "native")]
    for _ in range(n // 2):
        g = groups[int(rng.integers(len(groups)))]
        probes.append((g.members, True))
        swap = set(g.members)
        if swap:
            swap.pop()
        swap.add(str(rng.choice(np.array(members, dtype=object))))
        probes.append((frozenset(swap), frozenset(swap) in truth))

    recovered_r = {frozenset(R.members_of(gid)) for gid in R.group_nodes}
    correct = {"reified": 0, "native": 0}
    for comp, expected in probes:
        correct["reified"] += int((comp in recovered_r) == expected)
        correct["native"] += int((comp in truth) == expected)
    n_i = max(len(probes), 1)
    rows = [QueryResult("exact_composition", "dissolved", 0.0, 0.0, 0.0, False, len(probes),
                        "a pairwise graph cannot distinguish a real composition from "
                        "any set whose members happen to be mutually connected")]
    for k in ("reified", "native"):
        acc = correct[k] / n_i
        rows.append(QueryResult("exact_composition", k, acc, acc, acc, True, len(probes)))
    return rows


def q_group_lookup(coll, G, R, rng, n) -> list[QueryResult]:
    members = coll.members()
    picks = [str(x) for x in rng.choice(np.array(members, dtype=object),
                                        size=min(n, len(members)), replace=False)]
    correct = {"reified": 0, "native": 0}
    for m in picks:
        gold = coll.groups_of(m)
        pred_r = {g.replace(Reified.GROUP_PREFIX, "") for g in R.groups_of(m)}
        correct["reified"] += int(pred_r == gold)
        correct["native"] += 1
    n_i = max(len(picks), 1)
    rows = [QueryResult("group_lookup", "dissolved", 0.0, 0.0, 0.0, False, len(picks),
                        "there is no complex to return")]
    for k in ("reified", "native"):
        acc = correct[k] / n_i
        rows.append(QueryResult("group_lookup", k, acc, acc, acc, True, len(picks)))
    return rows


QUERY_FAMILIES = {
    "co_membership": q_co_membership,
    "exclusion": q_exclusion,
    "group_size": q_group_size,
    "exact_composition": q_exact_composition,
    "group_lookup": q_group_lookup,
}


def run_query_suite(coll: GroupCollection, n_instances: int = 200, seed: int = 0,
                    families: Sequence[str] | None = None) -> list[QueryResult]:
    rng = np.random.default_rng(seed)
    G = build_dissolved(coll)
    R = build_reified(coll)
    fams = list(families) if families else list(QUERY_FAMILIES)
    out: list[QueryResult] = []
    for f in fams:
        out.extend(QUERY_FAMILIES[f](coll, G, R, rng, n_instances))
    return out
