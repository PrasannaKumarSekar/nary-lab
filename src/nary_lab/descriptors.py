"""Layer 0 — describe the data before measuring anything.

Six statistics, all cheap, that between them anticipate what the later layers
will find. Computing them first is what makes this a reusable tool rather than
a one-off study: you can point Layer 0 at any group-structured subgraph and get
a prediction in seconds, then decide whether the expensive layers are worth
running.

The descriptors
---------------
``mean_group_size``
    Average number of members per group. A group of two *is* a pair, so a
    collection dominated by pairs has little for any group-preserving
    representation to preserve.

``frac_above_pair``
    Share of groups with three or more members — the portion of the data where
    the question even applies.

``hub_concentration``
    Share of all (group, member) incidences contributed by the ten most
    frequently occurring members. High values mean a few members appear
    everywhere, which is what causes distinct groups to blur together when
    dissolved into pairs.

``recurrence``
    Distinct member sets divided by number of groups. Low values mean the same
    composition recurs many times, which gives set-level methods a pattern to
    learn from.

``mean_overlap``
    Average Jaccard similarity between pairs of groups that share at least one
    member. Measures how ambiguous a pairwise decomposition would be.

``conformality``
    Fraction of groups that are exactly recoverable as maximal cliques of the
    pairwise graph. A collection is *conformal* when every maximal clique is a
    real group; conformal data survives dissolution intact. This is the one
    descriptor that is not cheap on very large inputs, so it is optional.

Predicted verdict
-----------------
:func:`predict` turns the descriptors into an expected outcome. It is a
heuristic, stated explicitly so it can be checked against the measured result
and corrected in future iterations. When prediction and measurement disagree,
that disagreement is the finding.
"""

from __future__ import annotations

import collections
import itertools
from dataclasses import asdict, dataclass, field

import numpy as np

from .model import GroupCollection

__all__ = ["Descriptors", "describe", "predict"]


@dataclass
class Descriptors:
    n_groups: int
    n_members: int
    n_incidences: int
    mean_group_size: float
    median_group_size: float
    max_group_size: int
    frac_above_pair: float
    frac_pairs_only: float
    hub_concentration: float
    top_member_share: float
    members_in_one_group: float
    recurrence: float
    mean_overlap: float
    overlapping_group_pairs: int
    overlap_members_capped: int
    implied_pairs: int
    pair_collision: float
    conformality: float | None = None
    size_histogram: dict[int, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def describe(coll: GroupCollection, compute_conformality: bool = False,
             conformality_cap: int = 25, overlap_degree_cap: int = 200,
             overlap_sample: int = 2_000_000, seed: int = 0) -> Descriptors:
    """Compute the Layer 0 descriptors.

    ``overlap_degree_cap`` bounds the pairwise-overlap computation. Enumerating
    every pair of groups that share a member is quadratic in a member's degree,
    which is fine for curated data where members appear in a handful of groups
    and impossible where a single member appears in tens of thousands. Members
    above the cap are sampled rather than enumerated, and
    ``overlap_members_capped`` records how many were affected so the number is
    never read as exact.
    """
    sizes = coll.sizes()
    members = coll.members()

    incid = collections.Counter()
    for g in coll:
        incid.update(g.members)
    counts = np.array(sorted(incid.values(), reverse=True), dtype=float)
    total_inc = counts.sum() if counts.size else 1.0

    # Group overlap, over pairs of groups that share a member. Quadratic in
    # member degree, so high-degree members are sampled (see docstring).
    rng = np.random.default_rng(seed)
    by_member = collections.defaultdict(list)
    for g in coll:
        for m in g.members:
            by_member[m].append(g.id)
    gmap = coll.groups
    shared = collections.Counter()
    capped = 0
    budget = overlap_sample
    for gids in by_member.values():
        uniq = sorted(set(gids))
        if len(uniq) > overlap_degree_cap:
            capped += 1
            uniq = [str(x) for x in rng.choice(np.array(uniq, dtype=object),
                                               size=overlap_degree_cap, replace=False)]
        n_pairs = len(uniq) * (len(uniq) - 1) // 2
        if n_pairs > budget:
            continue
        budget -= n_pairs
        for a, b in itertools.combinations(uniq, 2):
            shared[(a, b)] += 1
    jac = [n / len(gmap[a].members | gmap[b].members) for (a, b), n in shared.items()]

    implied = coll.implied_pairs()
    asserted = int(sum(s * (s - 1) // 2 for s in sizes))

    d = Descriptors(
        n_groups=len(coll),
        n_members=len(members),
        n_incidences=coll.n_incidences(),
        mean_group_size=float(sizes.mean()) if sizes.size else 0.0,
        median_group_size=float(np.median(sizes)) if sizes.size else 0.0,
        max_group_size=int(sizes.max()) if sizes.size else 0,
        frac_above_pair=float((sizes > 2).mean()) if sizes.size else 0.0,
        frac_pairs_only=float((sizes == 2).mean()) if sizes.size else 0.0,
        hub_concentration=float(counts[:10].sum() / total_inc) if counts.size else 0.0,
        top_member_share=float(counts[0] / total_inc) if counts.size else 0.0,
        members_in_one_group=float((counts == 1).mean()) if counts.size else 0.0,
        recurrence=len(coll.unique_member_sets()) / max(len(coll), 1),
        mean_overlap=float(np.mean(jac)) if jac else 0.0,
        overlapping_group_pairs=len(shared),
        overlap_members_capped=capped,
        implied_pairs=len(implied),
        pair_collision=1.0 - (len(implied) / asserted) if asserted else 0.0,
        size_histogram={int(k): int(v) for k, v in
                        sorted(collections.Counter(sizes.tolist()).items())},
    )

    if compute_conformality:
        from .representations import build_dissolved, recover_from_dissolved
        sub, _ = coll.filter_size(2, conformality_cap)
        G = build_dissolved(sub)
        cliques = set(recover_from_dissolved(G))
        truth = sub.unique_member_sets()
        d.conformality = len(truth & cliques) / max(len(truth), 1)
    return d


# --------------------------------------------------------------------------- #
# Heuristic prediction
# --------------------------------------------------------------------------- #
#: Thresholds are stated as constants rather than buried in the function so
#: that a future iteration can tune them against accumulated evidence.
PAIR_FLOOR = 3.0          # mean group size below which grouping has little to add
HUB_HIGH = 0.15           # hub concentration above which distinct groups blur
RECURRENCE_LOW = 0.6      # below this, compositions repeat enough to learn from


def predict(d: Descriptors) -> dict[str, str]:
    """Anticipated findings, to be compared against the measured ones."""
    out: dict[str, str] = {}

    if d.frac_above_pair == 0.0:
        out["scope"] = (
            f"Every group has exactly {d.median_group_size:.0f} members. This is the "
            "degenerate case: a group of two IS an ordinary edge, so there is no "
            "higher-order structure for any group-preserving representation to "
            "preserve. Expect no fidelity difference. Whatever the grouping is doing "
            "here, it is not solving an n-ary participant problem, and the comparison "
            "should move to qualifiers, multiplicity and cost."
        )
    elif d.mean_group_size < PAIR_FLOOR:
        out["scope"] = (
            f"Mean group size {d.mean_group_size:.2f} is near the pair boundary. "
            f"Only {d.frac_above_pair:.0%} of groups have three or more members, so "
            "expect a modest overall effect and report results split by group size."
        )
    else:
        out["scope"] = (
            f"Mean group size {d.mean_group_size:.2f} with {d.frac_above_pair:.0%} of "
            "groups above pair size: the comparison is in scope."
        )

    if d.hub_concentration >= HUB_HIGH:
        out["failure_mode"] = (
            f"Hub concentration {d.hub_concentration:.1%} is high. A few members appear "
            "in very many groups, so dissolving into pairs should mainly **invent** "
            "groups that were never asserted."
        )
    else:
        out["failure_mode"] = (
            f"Hub concentration {d.hub_concentration:.1%} is low, so the pairwise graph "
            "stays sparse. Dissolving should mainly cause smaller groups to **vanish** "
            "inside larger ones rather than inventing new ones."
        )

    if d.recurrence <= RECURRENCE_LOW:
        out["learnability"] = (
            f"Recurrence {d.recurrence:.2f}: compositions repeat often, giving "
            "set-level methods a pattern to exploit."
        )
    else:
        out["learnability"] = (
            f"Recurrence {d.recurrence:.2f}: nearly every group has a distinct "
            "composition, so set-level prediction has little regularity to learn."
        )

    if d.conformality is not None:
        out["recoverability"] = (
            f"Conformality {d.conformality:.2f}: that share of groups is exactly "
            "recoverable from the pairwise graph; the rest is not."
        )
    return out
