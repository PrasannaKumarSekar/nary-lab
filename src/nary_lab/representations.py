"""The three ways to store a higher-order fact in a graph.

Take one fact: *proteins A, B, C and D form a complex.* There are three places
you can put it.

1. ``dissolved`` — **pairwise graph.** Record A–B, A–C, A–D, B–C, B–D, C–D.
   The group itself is not stored anywhere; it only survives as the pattern its
   pairs leave behind. This is what a plain protein-interaction network looks
   like.

2. ``reified`` — **hub node in a property graph.** Create a node standing for
   the complex and link each protein to it. The group is preserved, and it is
   still an ordinary property graph, so existing tooling works unchanged.

3. ``native`` — **hyperedge.** One object whose members are A, B, C and D. A
   *hyperedge* is simply an edge that may connect any number of nodes rather
   than exactly two; a *hypergraph* is a collection of them.

The first differs from the other two in **what is stored**. The second and
third differ only in **how it is stored and what tooling can act on it** — they
carry identical information, which this framework verifies rather than assumes.

Each representation declares a :class:`Capability` record up front. The test
suite checks those declarations against measurement, so a claim that drifts
from behaviour fails CI rather than quietly misleading a reader.
"""

from __future__ import annotations

import collections
from collections.abc import Callable
from dataclasses import dataclass

import networkx as nx

from .model import GroupCollection

__all__ = [
    "Capability", "CAPABILITIES", "REPRESENTATIONS",
    "build_dissolved", "build_reified", "build_native",
    "recover_from_dissolved", "recover_from_reified", "recover_from_native",
    "Reified",
]


# --------------------------------------------------------------------------- #
# Declared capabilities
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Capability:
    """What a representation claims it can do. Verified in ``tests/``."""

    key: str
    display: str
    preserves_group: bool
    preserves_roles: bool
    group_addressable: bool        # can a query refer to the group as an object?
    losslessly_invertible: bool
    runs_in_property_graph: bool
    note: str


CAPABILITIES: dict[str, Capability] = {
    "dissolved": Capability(
        "dissolved", "Pairwise graph", False, False, False, False, True,
        "Group survives only as the pattern of pairs it leaves behind. Cheapest "
        "to query, and the only one of the three that loses information.",
    ),
    "reified": Capability(
        "reified", "Hub node (property graph)", True, True, True, True, True,
        "A node stands for the group. Lossless, queryable with standard graph "
        "tooling, at the cost of extra nodes and one extra hop per traversal.",
    ),
    "native": Capability(
        "native", "Hyperedge (hypergraph)", True, True, True, True, False,
        "One object per group. Same information as the hub node; different "
        "tooling, and set-level operations become direct rather than derived.",
    ),
}


# --------------------------------------------------------------------------- #
# 1. Dissolved: the pairwise graph
# --------------------------------------------------------------------------- #
def build_dissolved(coll: GroupCollection, weighted: bool = True) -> nx.Graph:
    """Connect every pair of co-members.

    ``weighted=True`` records how many groups each pair co-occurs in, which is
    the most a pairwise graph can retain about group structure. Even this is
    not enough to recover the groups: a weight of 3 on A–B says the pair was
    seen three times, not which three groups, nor who else was present.
    """
    G = nx.Graph()
    G.add_nodes_from(coll.members())
    for g in coll:
        for a, b in g.pairs():
            if weighted and G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)
    return G


# --------------------------------------------------------------------------- #
# 2. Reified: hub node in a property graph
# --------------------------------------------------------------------------- #
class Reified:
    """A bipartite property graph: member nodes, group nodes, incidence edges.

    Held as an explicit object rather than a bare ``nx.Graph`` so that the two
    node kinds stay distinguishable. That distinction is exactly what makes a
    reified graph queryable — and exactly what makes its queries longer, since
    every member-to-member question must route through a group node.
    """

    GROUP_PREFIX = "grp:"

    def __init__(self, coll: GroupCollection) -> None:
        self.graph = nx.Graph()
        self.group_nodes: set[str] = set()
        self.member_nodes: set[str] = set(coll.members())
        self.graph.add_nodes_from(self.member_nodes, kind="member")
        for g in coll:
            gid = self.GROUP_PREFIX + g.id
            self.group_nodes.add(gid)
            self.graph.add_node(gid, kind="group", group_type=g.group_type,
                                evidence=";".join(g.evidence),
                                timestamp=g.timestamp, order_key=g.order_key)
            rm = g.role_map
            for m in g.members:
                self.graph.add_edge(gid, m, role=rm.get(m, ""))

    # -- size ---------------------------------------------------------------#
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    def n_edges(self) -> int:
        return self.graph.number_of_edges()

    # -- queries ------------------------------------------------------------#
    def members_of(self, group_id: str) -> set[str]:
        gid = group_id if group_id.startswith(self.GROUP_PREFIX) else self.GROUP_PREFIX + group_id
        return {n for n in self.graph.neighbors(gid)} if gid in self.graph else set()

    def groups_of(self, member: str) -> set[str]:
        if member not in self.graph:
            return set()
        return {n for n in self.graph.neighbors(member) if n in self.group_nodes}

    def co_members(self, member: str) -> set[str]:
        out: set[str] = set()
        for g in self.groups_of(member):
            out |= self.members_of(g)
        out.discard(member)
        return out


def build_reified(coll: GroupCollection) -> Reified:
    return Reified(coll)


# --------------------------------------------------------------------------- #
# 3. Native: the hypergraph
# --------------------------------------------------------------------------- #
def build_native(coll: GroupCollection, backend: str = "hypergraphx"):
    """Build a hypergraph.

    ``backend="hypergraphx"`` returns a ``hypergraphx.Hypergraph``; the library
    infers its node set from the edge list and **deduplicates identical member
    sets**, so two complexes with the same membership collapse into one
    hyperedge. That is correct set semantics but it loses group identity, so
    the count is reported alongside.

    ``backend="sets"`` returns a plain list of frozensets, used by tests and as
    a dependency-free fallback.
    """
    edges = [tuple(sorted(g.members)) for g in coll]
    if backend == "sets":
        return [frozenset(e) for e in edges]
    from hypergraphx import Hypergraph
    return Hypergraph(edges)


def native_dedup_loss(coll: GroupCollection) -> dict[str, int]:
    """How many distinct groups share a member set and would be merged."""
    counts = collections.Counter(coll.member_sets())
    return {
        "groups_in": len(coll),
        "unique_member_sets": len(counts),
        "merged_away": len(coll) - len(counts),
    }


# --------------------------------------------------------------------------- #
# Recovery: try to get the original groups back out
# --------------------------------------------------------------------------- #
def recover_from_dissolved(G: nx.Graph, max_clique_size: int | None = None,
                           node_cap: int | None = None) -> list[frozenset[str]]:
    """Best-effort reconstruction of groups from a pairwise graph.

    Enumerates maximal cliques — sets of nodes in which every pair is connected
    and which cannot be extended. This is the *strongest* reconstruction a
    consumer of a pairwise graph can perform, so using anything weaker would
    measure the decoder rather than the encoding.

    Two failure modes follow directly from the method and are quantified in
    :mod:`nary_lab.fidelity`:

    * a clique may correspond to no real group (members that pairwise co-occur
      in *different* groups look identical to one group containing all of them);
    * a real group contained inside a larger one is not maximal, so it never
      appears in the output at all.
    """
    cliques = []
    for c in nx.find_cliques(G):
        if node_cap is not None and len(c) > node_cap:
            continue
        if max_clique_size is not None and len(c) > max_clique_size:
            continue
        cliques.append(frozenset(c))
    return cliques


def recover_from_reified(R: Reified) -> list[frozenset[str]]:
    """Exact inverse: read each group node's neighbours."""
    return [frozenset(R.members_of(g)) for g in sorted(R.group_nodes)]


def recover_from_native(H) -> list[frozenset[str]]:
    """Exact inverse: the hyperedges are the groups."""
    if isinstance(H, list):
        return [frozenset(e) for e in H]
    return [frozenset(e) for e in H.get_edges()]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
REPRESENTATIONS: dict[str, dict[str, Callable]] = {
    "dissolved": {"build": build_dissolved, "recover": recover_from_dissolved},
    "reified": {"build": build_reified, "recover": recover_from_reified},
    "native": {"build": build_native, "recover": recover_from_native},
}
