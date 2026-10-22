"""Layer 1 — what does each representation lose?

Method: build the representation, then try to read the original groups back
out, then compare. Comparing by round trip rather than by argument means the
answer is measured, and the same procedure applies to every representation
including the ones expected to be lossless.

Two kinds of error, reported separately because they have different causes and
different consequences:

**Invented groups** — the reconstruction produces a group that was never
asserted. In a pairwise graph this happens when members co-occur in *different*
groups: if A–B, B–C and A–C are each recorded from separate facts, nothing
distinguishes that from one fact containing all three. An invented group looks
exactly like a real finding, which makes this the more dangerous error.

**Vanished groups** — a real group is not recovered. In a pairwise graph this
happens when a group sits entirely inside a larger one: the smaller set is no
longer a *maximal* clique, so clique enumeration never returns it. The fact is
silently absent rather than wrong.

Everything is additionally reported **stratified by group size**, because a
single headline number is misleading whenever a collection mixes pairs with
larger assemblies.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

from .model import GroupCollection
from .representations import (
    CAPABILITIES,
    build_dissolved,
    build_native,
    build_reified,
    native_dedup_loss,
    recover_from_dissolved,
    recover_from_native,
    recover_from_reified,
)

__all__ = ["FidelityResult", "assess", "assess_all", "size_stratified"]


@dataclass
class FidelityResult:
    representation: str
    n_true_groups: int
    n_recovered: int
    exact: int
    precision: float
    recall: float
    f1: float
    invented: int
    vanished: int
    lossless: bool
    claims_lossless: bool
    claim_holds: bool
    by_size: dict[int, dict[str, float]] = field(default_factory=dict)
    notes: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["by_size"] = {str(k): v for k, v in self.by_size.items()}
        return d


def _compare(truth: set[frozenset[str]], recovered: list[frozenset[str]]) -> dict:
    rec = set(recovered)
    exact = len(truth & rec)
    precision = exact / max(len(rec), 1)
    recall = exact / max(len(truth), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "n_recovered": len(rec), "exact": exact,
        "precision": precision, "recall": recall, "f1": f1,
        "invented": len(rec - truth), "vanished": len(truth - rec),
    }


def size_stratified(truth: set[frozenset[str]], recovered: list[frozenset[str]],
                    min_count: int = 10) -> dict[int, dict[str, float]]:
    """Recovery rate broken down by group size."""
    rec = set(recovered)
    by_size_truth = collections.Counter(len(s) for s in truth)
    by_size_hit = collections.Counter(len(s) for s in truth & rec)
    out: dict[int, dict[str, float]] = {}
    tail_t = tail_h = 0
    for k in sorted(by_size_truth):
        if by_size_truth[k] >= min_count:
            out[k] = {"n_true": by_size_truth[k], "n_recovered": by_size_hit[k],
                      "rate": by_size_hit[k] / by_size_truth[k]}
        else:
            tail_t += by_size_truth[k]
            tail_h += by_size_hit[k]
    if tail_t:
        out[-1] = {"n_true": tail_t, "n_recovered": tail_h, "rate": tail_h / tail_t,
                   "label": "remaining larger sizes pooled"}
    return out


def assess(coll: GroupCollection, representation: str,
           clique_cap: int | None = None) -> FidelityResult:
    """Round-trip one representation and score the result."""
    truth = coll.unique_member_sets()
    cap = CAPABILITIES[representation]
    notes = ""

    if representation == "dissolved":
        G = build_dissolved(coll)
        recovered = recover_from_dissolved(G, node_cap=clique_cap)
        notes = ("Reconstruction by maximal-clique enumeration, the strongest "
                 "decoder available to a consumer of a pairwise graph.")
    elif representation == "reified":
        recovered = recover_from_reified(build_reified(coll))
    elif representation == "native":
        H = build_native(coll, backend="sets")
        recovered = recover_from_native(H)
        dl = native_dedup_loss(coll)
        if dl["merged_away"]:
            notes = (f"{dl['merged_away']} groups share a member set with another and "
                     "merge into one hyperedge: set semantics preserve membership but "
                     "not group identity.")
    else:  # pragma: no cover
        raise ValueError(representation)

    m = _compare(truth, recovered)
    lossless = (m["invented"] == 0 and m["vanished"] == 0)
    # A representation claiming losslessness must never lose anything. One that
    # makes no such claim may still be lossless on favourable input: dissolution
    # is exactly recoverable when the data is conformal, i.e. when every maximal
    # clique of the pairwise graph happens to be a real group. So the claim is
    # violated only in one direction.
    claim_holds = lossless if cap.losslessly_invertible else True
    return FidelityResult(
        representation=representation,
        n_true_groups=len(truth),
        lossless=lossless,
        claims_lossless=cap.losslessly_invertible,
        claim_holds=claim_holds,
        by_size=size_stratified(truth, recovered),
        notes=notes,
        **m,
    )


def assess_all(coll: GroupCollection, clique_cap: int | None = None,
               representations: Sequence[str] = ("dissolved", "reified", "native"),
               ) -> list[FidelityResult]:
    return [assess(coll, r, clique_cap=clique_cap) for r in representations]


# --------------------------------------------------------------------------- #
# Independent-evidence coverage
# --------------------------------------------------------------------------- #
def coverage_by_independent_pairs(coll: GroupCollection,
                                  observed_pairs: set[tuple]) -> dict:
    """How much group co-membership an *independently sourced* pairwise layer covers.

    Distinct from :func:`assess`. There, the pairwise graph is derived from the
    groups, so the comparison isolates the cost of dissolution. Here the pairwise
    layer comes from separate experiments, so the comparison answers a different
    and more practical question: if a query relies on the interaction layer that
    actually exists in the graph, how much of the group structure does it see?
    """
    implied = coll.implied_pairs()
    hit = implied & observed_pairs
    per_group = []
    for g in coll:
        ps = set(g.pairs())
        if ps:
            per_group.append(len(ps & observed_pairs) / len(ps))
    per_group_arr = np.array(per_group) if per_group else np.array([0.0])
    fully = float((per_group_arr == 1.0).mean())
    none = float((per_group_arr == 0.0).mean())
    return {
        "implied_pairs": len(implied),
        "observed_pairs_total": len(observed_pairs),
        "implied_pairs_observed": len(hit),
        "pair_coverage": len(hit) / max(len(implied), 1),
        "mean_group_coverage": float(per_group_arr.mean()),
        "groups_fully_covered": fully,
        "groups_with_no_coverage": none,
    }
