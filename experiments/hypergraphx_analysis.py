"""Working with the complexes as a genuine hypergraph, using ``hypergraphx``.

The main study builds its own lightweight structures so that it has no hard
dependency on a hypergraph library. This script does the opposite: it hands the
data to ``hypergraphx`` and exercises the operations that only exist once the
group is a first-class object, including one -- higher-order motif analysis --
that has no counterpart in a pairwise graph at all.

It also records where the tooling gets in the way. Those observations are part
of the result: a representation is only as usable as the software available for
it, and that cost belongs in the comparison rather than in a footnote.

Run:  python experiments/hypergraphx_analysis.py --data-dir data
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import numpy as np

from nary_lab.io import load_complexes
from nary_lab.scorecard import save_json


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-size", type=int, default=25)
    ap.add_argument("--motif-order", type=int, default=3)
    ap.add_argument("--motif-runs", type=int, default=10)
    args = ap.parse_args()

    try:
        from hypergraphx import Hypergraph
        from hypergraphx.linalg import binary_incidence_matrix
        from hypergraphx.motifs import compute_motifs
        from hypergraphx.representations.projections import clique_projection
    except ImportError:
        print("hypergraphx is not installed. Install with: pip install 'nary_lab[hypergraph]'")
        return 1

    findings: dict = {"tooling_observations": []}

    def note(area: str, text: str, impact: str = "medium") -> None:
        findings["tooling_observations"].append(
            {"area": area, "observation": text, "impact": impact})
        print(f"  [tooling:{area}] {text}")

    # ------------------------------------------------------------------ build
    coll, report = load_complexes(args.data_dir, member_column=":END_ID",
                                  min_size=2, max_size=args.max_size)
    edges = [tuple(sorted(g.members)) for g in coll]
    H = Hypergraph(edges)
    print(f"built hypergraph: {H.num_nodes()} proteins, {H.num_edges()} complexes")
    findings["build"] = {"groups_passed_in": len(edges),
                         "hyperedges_stored": H.num_edges(),
                         "nodes": H.num_nodes()}
    if H.num_edges() < len(edges):
        note("group identity",
             f"{len(edges) - H.num_edges()} complexes share a subunit set with another "
             "and were merged: a hyperedge is a set, so two complexes with identical "
             "membership become one object.", "medium")

    # -------------------------------------------------------------- incidence
    inc, mapping = binary_incidence_matrix(H, return_mapping=True)
    node_order = [mapping[i] for i in range(inc.shape[0])]
    degrees = np.asarray(inc.astype(float).sum(1)).ravel()
    findings["incidence"] = {"shape": list(inc.shape), "nonzeros": int(inc.nnz),
                             "isolated_nodes": int((degrees == 0).sum()),
                             "first_nodes": node_order[:3]}
    note("index alignment",
         "Row order of the incidence matrix must be obtained via return_mapping. "
         "Library routines that silently drop zero-degree nodes will otherwise "
         "return vectors that no longer line up with the node list, which produces "
         "a plausible but wrong answer rather than an error.", "high")

    # ------------------------------------------------------------- projection
    t0 = time.perf_counter()
    G = clique_projection(H)
    findings["clique_projection"] = {
        "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    print(f"clique projection: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ------------------------------------------------- size (order) structure
    sizes = collections.Counter(len(e) for e in H.get_edges())
    findings["size_histogram"] = {int(k): int(v) for k, v in sorted(sizes.items())}
    note("uniformity",
         f"Complex sizes span {min(sizes)}-{max(sizes)}. Tensor eigenvector "
         "centralities (CEC/ZEC) require a uniform hypergraph -- every group the "
         "same size -- so they cannot be applied to this data as a whole and would "
         "need the data sliced by size first.", "medium")

    # --------------------------------------------------------------- motifs
    # Higher-order motifs ask which small configurations of proteins occur more
    # often than chance, where "configuration" distinguishes three proteins in
    # one complex from three proteins linked only pairwise. A pairwise graph
    # cannot express that distinction, so this analysis has no graph counterpart.
    print(f"computing order-{args.motif_order} motifs "
          f"({args.motif_runs} null-model runs; this is the slow step)...")
    t0 = time.perf_counter()
    try:
        m = compute_motifs(H, order=args.motif_order, runs_config_model=args.motif_runs)
        elapsed = time.perf_counter() - t0
        observed = [(str(p), int(c)) for p, c in m["observed"]]
        deltas = [(str(p), float(d)) for p, d in m["norm_delta"]]
        findings["motifs"] = {
            "order": args.motif_order, "null_runs": args.motif_runs,
            "seconds": round(elapsed, 1),
            "observed": observed,
            "normalised_difference_vs_null": deltas,
        }
        print(f"  done in {elapsed:.1f}s")
        print("  pattern            observed    vs null")
        for (p, c), (_, d) in zip(observed, deltas, strict=False):
            print(f"  {p:<18}{c:>10,}{d:>11.3f}")
    except Exception as exc:  # pragma: no cover - depends on library internals
        note("motifs", f"compute_motifs failed: {type(exc).__name__}: {exc}", "high")

    out = Path(args.out_dir) / "hypergraphx_analysis.json"
    save_json(findings, out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
