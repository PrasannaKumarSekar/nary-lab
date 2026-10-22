"""Tests for the comparison framework.

The most important tests here are the ones that check a representation's
*declared* capabilities against its *measured* behaviour. A docstring that
drifts away from what the code does is worse than no docstring, so the claims
in ``representations.CAPABILITIES`` are treated as assertions under test.
"""

import collections

import numpy as np
import pytest

from nary_lab.analytics import (
    candidate_triples,
    community_benchmark,
    verify_operator_equivalence,
)
from nary_lab.descriptors import describe, predict
from nary_lab.fidelity import assess, assess_all, coverage_by_independent_pairs
from nary_lab.implementation import footprint
from nary_lab.model import Group, GroupCollection
from nary_lab.pubmed import attach_time, describe_basis, ordinal_proxy
from nary_lab.queries import run_query_suite
from nary_lab.representations import (
    CAPABILITIES,
    build_dissolved,
    build_native,
    build_reified,
    native_dedup_loss,
    recover_from_native,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def simple() -> GroupCollection:
    """Three groups, no nesting, no shared-pair ambiguity."""
    return GroupCollection([
        Group("g1", frozenset("abc"), evidence=("100",)),
        Group("g2", frozenset("de"), evidence=("200",)),
        Group("g3", frozenset("fgh"), evidence=("300",)),
    ], "simple")


def nested() -> GroupCollection:
    """A pair sitting inside a triple: the pair cannot be a maximal clique."""
    return GroupCollection([
        Group("big", frozenset("abc")),
        Group("small", frozenset("ab")),
    ], "nested")


def ambiguous() -> GroupCollection:
    """Three pairs that pairwise connect a, b, c without any group holding all three."""
    return GroupCollection([
        Group("p1", frozenset("ab")),
        Group("p2", frozenset("bc")),
        Group("p3", frozenset("ac")),
    ], "ambiguous")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def test_group_basics():
    g = Group("x", frozenset("abcd"))
    assert g.size == 4
    assert len(list(g.pairs())) == 6
    assert not g.has_roles()


def test_duplicate_group_id_rejected():
    with pytest.raises(KeyError):
        GroupCollection([Group("a", frozenset("xy")), Group("a", frozenset("yz"))])


def test_membership_index_and_implied_pairs():
    c = simple()
    assert c.groups_of("a") == {"g1"}
    assert len(c.implied_pairs()) == 3 + 1 + 3
    assert c.n_incidences() == 8


def test_filter_size_reports_what_it_dropped():
    c = GroupCollection([Group("a", frozenset("x")), Group("b", frozenset("xy")),
                         Group("c", frozenset("xyzw"))])
    kept, dropped = c.filter_size(2, 3)
    assert len(kept) == 1
    assert dropped == {"below_min_size": 1, "above_max_size": 1}


def test_restrict_members_drops_undersized_groups():
    c = simple().restrict_members({"a", "b", "d"}, min_size=2)
    assert {g.id for g in c} == {"g1", "g2"} - {"g2"} | {"g1"}  # only g1 keeps 2 members
    assert len(c) == 1


def test_rows_roundtrip():
    c = simple()
    back = GroupCollection.from_rows(c.to_rows(), "back")
    assert {g.members for g in back} == {g.members for g in c}


# --------------------------------------------------------------------------- #
# Representations and their declared claims
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("coll", [simple(), nested(), ambiguous()])
def test_group_preserving_forms_are_lossless(coll):
    for rep in ("reified", "native"):
        r = assess(coll, rep)
        # native merges identical member sets, which is set semantics not loss
        if rep == "native" and native_dedup_loss(coll)["merged_away"]:
            continue
        assert r.lossless, f"{rep} lost information on {coll.name}"
        assert r.f1 == pytest.approx(1.0)


def test_declared_capabilities_match_measurement():
    """CAPABILITIES is an assertion, not documentation."""
    coll = simple()
    for r in assess_all(coll):
        cap = CAPABILITIES[r.representation]
        assert r.claim_holds, (
            f"{r.representation} claims losslessly_invertible="
            f"{cap.losslessly_invertible} but measured lossless={r.lossless}"
        )


def test_lossy_representation_may_still_be_lossless_on_conformal_data():
    """Dissolution loses nothing when no group nests inside another and no
    pair-triangle spans separate groups. The claim check must allow this."""
    r = assess(simple(), "dissolved")
    assert r.lossless and not r.claims_lossless and r.claim_holds


def test_dissolved_invents_a_group_when_pairs_come_from_different_facts():
    r = assess(ambiguous(), "dissolved")
    assert r.invented == 1          # {a,b,c} looks like one group but is three pairs
    assert r.vanished == 3          # none of the three real pairs survive as maximal
    assert r.f1 == 0.0


def test_dissolved_loses_a_group_nested_inside_another():
    r = assess(nested(), "dissolved")
    assert r.vanished == 1          # {a,b} is not maximal inside {a,b,c}
    assert r.invented == 0


def test_dissolved_is_lossless_on_disjoint_groups():
    r = assess(simple(), "dissolved")
    assert r.lossless


def test_reified_keeps_node_kinds_separate():
    R = build_reified(simple())
    assert len(R.group_nodes) == 3
    assert len(R.member_nodes) == 8
    assert R.members_of("g1") == set("abc")
    assert R.co_members("a") == set("bc")


def test_native_merges_identical_member_sets():
    c = GroupCollection([Group("x", frozenset("ab")), Group("y", frozenset("ab"))])
    loss = native_dedup_loss(c)
    assert loss["merged_away"] == 1
    assert len(recover_from_native(build_native(c, backend="sets"))) == 2  # list keeps both
    assert len(set(recover_from_native(build_native(c, backend="sets")))) == 1


def test_dissolved_weights_count_cooccurrences():
    c = GroupCollection([Group("x", frozenset("ab")), Group("y", frozenset("abc"))])
    G = build_dissolved(c)
    assert G["a"]["b"]["weight"] == 2
    assert G["a"]["c"]["weight"] == 1


# --------------------------------------------------------------------------- #
# Descriptors
# --------------------------------------------------------------------------- #
def test_descriptors_basic_counts():
    d = describe(simple())
    assert d.n_groups == 3 and d.n_members == 8
    assert d.mean_group_size == pytest.approx(8 / 3)
    assert d.recurrence == 1.0


def test_hub_concentration_responds_to_a_shared_member():
    spread = GroupCollection([Group(f"g{i}", frozenset({f"a{i}", f"b{i}"}))
                              for i in range(20)])
    hubby = GroupCollection([Group(f"g{i}", frozenset({"HUB", f"b{i}"}))
                             for i in range(20)])
    assert describe(hubby).hub_concentration > describe(spread).hub_concentration


def test_prediction_returns_all_expected_keys():
    p = predict(describe(simple()))
    assert {"scope", "failure_mode", "learnability"} <= set(p)


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
def test_query_suite_marks_inexpressible_families_as_zero():
    coll = GroupCollection([Group(f"g{i}", frozenset({f"p{i}", f"q{i}", f"r{i}"}))
                            for i in range(30)])
    rows = run_query_suite(coll, n_instances=20, seed=0)
    by = collections.defaultdict(dict)
    for r in rows:
        by[r.representation][r.family] = r
    for fam in ("exclusion", "group_size", "exact_composition", "group_lookup"):
        assert by["dissolved"][fam].expressible is False
        assert by["dissolved"][fam].f1 == 0.0
        assert by["reified"][fam].f1 == pytest.approx(1.0)
        assert by["native"][fam].f1 == pytest.approx(1.0)


def test_co_membership_is_exact_for_group_preserving_forms():
    coll = GroupCollection([Group("g1", frozenset("abcd")), Group("g2", frozenset("cdef"))])
    rows = run_query_suite(coll, n_instances=10, seed=1, families=["co_membership"])
    by = {r.representation: r for r in rows}
    assert by["reified"].f1 == pytest.approx(1.0)
    assert by["native"].f1 == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #
def test_hypergraph_laplacian_equals_size_weighted_pairwise():
    coll = GroupCollection([Group(f"g{i}", frozenset({f"p{i%7}", f"q{i%5}", f"r{i%3}"}))
                            for i in range(40)])
    eq = verify_operator_equivalence(coll)
    assert eq["equivalent"]
    assert eq["max_abs_diff_vs_size_weighted_pairwise"] < 1e-10


def test_hypergraph_laplacian_differs_from_unweighted_pairwise():
    coll = GroupCollection([Group("a", frozenset("abcdef")), Group("b", frozenset("fgh")),
                            Group("c", frozenset("hij"))])
    eq = verify_operator_equivalence(coll)
    assert eq["distinct_from_unweighted"]


def test_candidate_triples_excludes_already_closed_ones():
    closed = GroupCollection([Group("g", frozenset("abc"))])
    triples, _ = candidate_triples(closed)
    assert triples == set()
    triples, _ = candidate_triples(ambiguous())
    assert triples == {frozenset("abc")}


def test_community_benchmark_recovers_clean_planted_structure():
    groups, labels = [], {}
    for m in range(4):
        members = [f"m{m}_{i}" for i in range(6)]
        for x in members:
            labels[x] = f"mod{m}"
        for j in range(5):
            groups.append(Group(f"g{m}_{j}", frozenset(members[j % 3:j % 3 + 3])))
    rows = community_benchmark(GroupCollection(groups), labels, seed=0)
    assert {r["method"] for r in rows} == {
        "pairwise_unweighted", "pairwise_size_weighted", "hypergraph_laplacian"}
    assert all(r["nmi"] > 0.8 for r in rows)


# --------------------------------------------------------------------------- #
# Time handling
# --------------------------------------------------------------------------- #
def test_ordinal_proxy_is_monotone_in_pmid():
    p = ordinal_proxy(["300", "100", "200"])
    assert p["100"] < p["200"] < p["300"]


def test_attach_time_uses_proxy_and_says_so(tmp_path):
    c = attach_time(simple(), tmp_path / "missing.json")
    assert all(g.timestamp is None for g in c)
    assert all(g.order_key is not None for g in c)
    assert "ORDINAL PROXY" in describe_basis(c)


def test_split_by_time_reports_its_basis():
    c = attach_time(simple(), "does/not/exist.json")
    early, late, cut, basis = c.split_by_time(0.6)
    assert len(early) + len(late) == len(c)
    assert "proxy" in basis


# --------------------------------------------------------------------------- #
# Implementation layer
# --------------------------------------------------------------------------- #
def test_footprint_shows_node_increase_and_edge_decrease():
    coll = GroupCollection([
        Group(f"g{i}", frozenset({f"p{i}a", f"p{i}b", f"p{i}c", f"p{i}d"}))
        for i in range(10)])
    fp = {r["representation"]: r for r in footprint(coll)}
    # 10 disjoint groups of 4: dissolution makes 10*6=60 pairs, a hub makes 10*4=40 edges
    assert fp["reified"]["extra_nodes"] == 10
    assert fp["reified"]["nodes"] > fp["dissolved"]["nodes"]
    assert fp["reified"]["edges"] < fp["dissolved"]["edges"]   # k edges vs k(k-1)/2 pairs
    assert fp["native"]["edges"] == 10


def test_independent_coverage_metric():
    coll = simple()
    observed = {("a", "b")}
    cov = coverage_by_independent_pairs(coll, observed)
    assert cov["implied_pairs_observed"] == 1
    assert 0 < cov["pair_coverage"] < 1
    assert cov["groups_with_no_coverage"] > 0


# --------------------------------------------------------------------------- #
# Formation prediction: significance machinery
# --------------------------------------------------------------------------- #
def _timed_collection(n_modules=12, per=14, seed=0):
    """Groups with real timestamps and some genuinely predictable closure."""
    rng = np.random.default_rng(seed)
    groups = []
    gid = 0
    for m in range(n_modules):
        pool = [f"m{m}_p{i}" for i in range(8)]
        for _ in range(per):
            size = int(rng.integers(2, 4))
            members = frozenset(rng.choice(np.array(pool, dtype=object),
                                           size=size, replace=False).tolist())
            year = float(rng.integers(1995, 2021))
            groups.append(Group(f"g{gid}", members, timestamp=year))
            gid += 1
    return GroupCollection(groups, "timed")


def test_formation_prediction_reports_null_distribution_and_pvalues():
    from nary_lab.analytics import formation_prediction

    res = formation_prediction(_timed_collection(), quantile=0.5, seed=0, n_null=200)
    assert res["basis"] == "timestamp"
    assert res["quantile"] == 0.5
    nd = res["null_distribution"]
    assert nd["draws"] == 200
    assert nd["p95_lift"] >= nd["mean_lift"]
    for row in res["scores"]:
        assert 0.0 <= row["p_value"] <= 1.0
        assert row["beats_chance"] == (row["p_value"] < 0.05)
    # sorted best-first
    aps = [r["average_precision"] for r in res["scores"]]
    assert aps == sorted(aps, reverse=True)


def test_formation_prediction_flags_underpowered_results():
    from nary_lab.analytics import formation_prediction

    res = formation_prediction(_timed_collection(n_modules=3, per=6), quantile=0.5,
                               seed=0, n_null=100)
    assert res["underpowered"] is True


def test_random_labels_produce_no_significant_scorer():
    """A control: when closure is unrelated to structure, nothing should clear
    the null. This is what stops the task reporting noise as a finding."""
    from nary_lab.analytics import formation_prediction

    rng = np.random.default_rng(7)
    groups = []
    members = [f"p{i}" for i in range(60)]
    for i in range(400):
        sz = int(rng.integers(2, 4))
        m = frozenset(rng.choice(np.array(members, dtype=object), size=sz,
                                 replace=False).tolist())
        groups.append(Group(f"r{i}", m, timestamp=float(rng.integers(1990, 2021))))
    res = formation_prediction(GroupCollection(groups), quantile=0.5, seed=0, n_null=500)
    if res["n_positives"] > 0:
        assert res["n_beating_chance"] <= 1   # allow one false positive at alpha=0.05


def test_pubmed_dates_csv_is_preferred_over_json_cache(tmp_path):
    from nary_lab.pubmed import load_dates_csv, load_year_map

    csv_path = tmp_path / "pmid_dates.csv"
    csv_path.write_text("pmid,year,month\n111,2001,3\n222,,\n")
    m = load_dates_csv(csv_path)
    assert m == {"111": 2001, "222": None}

    json_path = tmp_path / "cache.json"
    json_path.write_text('{"333": 1999}')
    got, src = load_year_map(csv_path, json_path)
    assert got["111"] == 2001 and "pmid_dates" in src

    got, src = load_year_map(tmp_path / "absent.csv", json_path)
    assert got == {"333": 1999} and "json" in src


def test_real_years_take_precedence_over_ordinal_proxy(tmp_path):
    from nary_lab.pubmed import attach_time, describe_basis

    coll = GroupCollection([Group("g1", frozenset("ab"), evidence=("111",)),
                            Group("g2", frozenset("cd"), evidence=("222",))])
    csv_path = tmp_path / "pmid_dates.csv"
    csv_path.write_text("pmid,year\n111,2001\n222,2010\n")
    out = attach_time(coll, tmp_path / "none.json", csv_path)
    assert {g.timestamp for g in out} == {2001.0, 2010.0}
    assert "publication year resolved" in describe_basis(out)
