"""Layer 3 — what does each representation cost to build, store and use?

Fidelity and analytics do not settle an architecture decision on their own. Two
representations can hold identical information and still differ in storage
footprint, in how long a query is to write, in which algorithms are available,
and in whether they run in the database you already operate.

The hub-node and hyperedge forms are informationally identical, so Layer 1 will
score them the same. This layer is where they actually differ, and it is
therefore the layer that decides between them.

A note on footprint
-------------------
"Adding a node per group bloats the graph" is a common expectation and is only
half right. Hub nodes increase the node count, but they *reduce* the edge count
relative to a pairwise graph, because dissolving a group of size k produces
k(k-1)/2 pairs while a hub produces k edges. Which effect dominates depends
entirely on the group-size distribution, so it should be measured per subgraph
rather than assumed. :func:`footprint` reports both.
"""

from __future__ import annotations

import time

from .model import GroupCollection
from .representations import CAPABILITIES, build_dissolved, build_native, build_reified

__all__ = ["footprint", "build_timing", "TRAVERSAL_COST", "ALGORITHM_AVAILABILITY",
           "TOOLING_NOTES", "summarise_tradeoffs"]


def footprint(coll: GroupCollection) -> list[dict]:
    """Nodes and edges materialised by each representation."""
    n_members = len(coll.members())
    n_groups = len(coll)
    n_incidences = coll.n_incidences()
    n_pairs = len(coll.implied_pairs())
    rows = [
        {"representation": "dissolved", "nodes": n_members, "edges": n_pairs,
         "extra_nodes": 0, "stored_cells": n_pairs},
        {"representation": "reified", "nodes": n_members + n_groups, "edges": n_incidences,
         "extra_nodes": n_groups, "stored_cells": n_incidences},
        {"representation": "native", "nodes": n_members, "edges": n_groups,
         "extra_nodes": 0, "stored_cells": n_incidences},
    ]
    base_nodes, base_edges = rows[0]["nodes"], max(rows[0]["edges"], 1)
    for r in rows:
        r["node_ratio_vs_dissolved"] = r["nodes"] / max(base_nodes, 1)
        r["edge_ratio_vs_dissolved"] = r["edges"] / base_edges
    return rows


def build_timing(coll: GroupCollection, repeats: int = 1) -> list[dict]:
    rows = []
    for name, fn in (("dissolved", build_dissolved), ("reified", build_reified)):
        t0 = time.perf_counter()
        for _ in range(repeats):
            fn(coll)
        rows.append({"representation": name,
                     "build_seconds": (time.perf_counter() - t0) / repeats})
    t0 = time.perf_counter()
    for _ in range(repeats):
        build_native(coll, backend="sets")
    rows.append({"representation": "native",
                 "build_seconds": (time.perf_counter() - t0) / repeats})
    return rows


#: Traversal cost for the canonical question "what else is in a complex with X?"
#: Derived from the encoding, not timed: hop count and intermediate cardinality
#: dominate query latency in a graph store, and the hub node is where a reified
#: graph pays for its correctness.
TRAVERSAL_COST = [
    {"representation": "dissolved", "hops": 1, "intermediate": "none",
     "cypher_sketch": "(x)-[:INTERACTS]-(y)",
     "note": "one hop, but returns anything connected to X rather than anything "
             "in a complex with X"},
    {"representation": "reified", "hops": 2, "intermediate": "complex node",
     "cypher_sketch": "(x)<-[:HAS_SUBUNIT]-(c:Complex)-[:HAS_SUBUNIT]->(y)",
     "note": "correct; every question about members must route through the "
             "complex node, and frequently-reused proteins make those nodes hot"},
    {"representation": "native", "hops": 1, "intermediate": "hyperedge",
     "cypher_sketch": "n/a - not a property graph",
     "note": "the group is the object; member lookup is direct"},
]


#: Which operations are available directly in each representation. "via
#: projection" means the operation exists but only after converting to a
#: pairwise graph first, which reintroduces that graph's assumptions.
ALGORITHM_AVAILABILITY = [
    {"operation": "shortest path / connectivity", "dissolved": "native",
     "reified": "native", "native": "via projection"},
    {"operation": "degree / betweenness centrality", "dissolved": "native",
     "reified": "native (inflated by hub nodes)", "native": "via projection"},
    {"operation": "community detection", "dissolved": "native",
     "reified": "native", "native": "native"},
    {"operation": "count groups by size", "dissolved": "unavailable",
     "reified": "native", "native": "native"},
    {"operation": "set membership / exclusion query", "dissolved": "unavailable",
     "reified": "native", "native": "native"},
    {"operation": "group-level prediction", "dissolved": "unavailable",
     "reified": "possible, must reassemble members", "native": "native"},
    {"operation": "higher-order motif analysis", "dissolved": "unavailable",
     "reified": "unavailable", "native": "native"},
    {"operation": "runs in existing property-graph store", "dissolved": "yes",
     "reified": "yes", "native": "no"},
]


#: Observed friction using the hypergraph tooling, recorded because it is a
#: real and recurring cost, not an abstraction. Each entry is something that
#: cost time or produced a wrong intermediate result during this study.
TOOLING_NOTES = [
    {"area": "tensor centralities",
     "observation": "CEC / ZEC centrality require a uniform hypergraph, i.e. every "
                    "group the same size. Complex data spans sizes 2 to 129, so these "
                    "are unavailable without slicing the data by size first.",
     "impact": "medium"},
    {"area": "spectral clustering",
     "observation": "The library's spectral routine drops isolated nodes before "
                    "clustering, so its output rows no longer align with the node list "
                    "returned by the hypergraph object. Aligning them incorrectly "
                    "produces a plausible but wrong score.",
     "impact": "high - silent wrong answer",
     "mitigation": "obtain the index-to-node map from the incidence matrix, and "
                   "check for zero-degree nodes before interpreting any output"},
    {"area": "projections",
     "observation": "clique_projection returns a networkx graph; bipartite_projection "
                    "returns a tuple. The asymmetry is easy to miss.",
     "impact": "low"},
    {"area": "group identity",
     "observation": "Constructing a hypergraph from an edge list silently merges "
                    "groups with identical membership, since a hyperedge is a set. "
                    "Distinct groups that happen to share a composition are lost.",
     "impact": "medium - affects counts"},
    {"area": "ecosystem",
     "observation": "No managed hypergraph store comparable to a production property "
                    "graph. Persistence, access control and bulk import must be built.",
     "impact": "high for production use"},
]


def summarise_tradeoffs(coll: GroupCollection) -> dict:
    fp = footprint(coll)
    by = {r["representation"]: r for r in fp}
    return {
        "footprint": fp,
        "traversal": TRAVERSAL_COST,
        "algorithms": ALGORITHM_AVAILABILITY,
        "tooling": TOOLING_NOTES,
        "capabilities": {k: v.__dict__ for k, v in CAPABILITIES.items()},
        "headline": (
            f"Hub nodes add {by['reified']['extra_nodes']} nodes "
            f"({by['reified']['node_ratio_vs_dissolved'] - 1:+.0%} vs pairwise) but "
            f"{by['reified']['edge_ratio_vs_dissolved'] - 1:+.0%} edges; "
            f"hyperedges store the same information in {by['native']['edges']} objects."
        ),
    }
