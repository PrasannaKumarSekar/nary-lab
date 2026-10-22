"""Layer 2b — does keeping the group help you *compute* anything better?

Two tasks.

**Community detection.** Partition proteins into functional modules and score
against curated labels. Run under three operators:

* the pairwise graph's normalised Laplacian;
* the same graph with each group weighted ``1/(size-1)``, so a large group
  contributes less to each of its pairs;
* the hypergraph Laplacian.

A result worth knowing before reading the numbers: the hypergraph Laplacian
used here (Zhou et al. 2006) is algebraically the normalised Laplacian of the
pairwise graph weighted by ``1/size``. The three operators are therefore not
independent, and close agreement between them is the expected outcome rather
than a disappointing one. This task is included precisely to check that
expectation on real data. :func:`verify_operator_equivalence` tests it directly.

**Group-formation prediction.** Take proteins that are already known to pair
with one another but have never been reported in a single complex. Do they
later turn out to form one? Split chronologically: fit on complexes published
up to a cut-off, evaluate on those published afterwards.

Protocol note that matters: whole complexes are held out, never individual
subunits. Holding out one subunit of a complex that remains in the data leaves
a direct pairwise edge from every other subunit to the answer, and every method
then scores near-perfectly on an artefact of the split.

Scores are reported as area under the precision-recall curve **relative to
random**, following the convention for this task, because the positive rate is
very low and raw accuracy would be uninformative.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from .model import GroupCollection

__all__ = [
    "community_benchmark", "verify_operator_equivalence",
    "formation_prediction", "candidate_triples",
]


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #
def incidence(coll: GroupCollection, nodes: Sequence[str] | None = None):
    nodes = list(nodes) if nodes is not None else coll.members()
    idx = {n: i for i, n in enumerate(nodes)}
    rows, cols = [], []
    for j, g in enumerate(coll):
        for m in g.members:
            if m in idx:
                rows.append(idx[m])
                cols.append(j)
    H = sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                      shape=(len(nodes), len(coll)), dtype=float)
    return H, nodes


def _norm_laplacian(A: sp.spmatrix) -> sp.csr_matrix:
    d = np.asarray(A.sum(axis=1)).ravel()
    inv = sp.diags(np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0))
    return (sp.identity(A.shape[0], format="csr") - inv @ A @ inv).tocsr()


def pairwise_adjacency(coll, nodes=None, weight_by_size: bool = False):
    H, nodes = incidence(coll, nodes)
    if weight_by_size:
        delta = np.asarray(H.sum(axis=0)).ravel()
        w = np.where(delta > 1, 1.0 / np.maximum(delta - 1.0, 1.0), 0.0)
        A = (H @ sp.diags(w) @ H.T).tocsr()
    else:
        A = (H @ H.T).tocsr()
        A.data = np.ones_like(A.data)
    A.setdiag(0)
    A.eliminate_zeros()
    return A.tocsr(), nodes


def hypergraph_laplacian(coll, nodes=None):
    """Zhou normalised hypergraph Laplacian."""
    H, nodes = incidence(coll, nodes)
    dv = np.asarray(H.sum(axis=1)).ravel()
    de = np.asarray(H.sum(axis=0)).ravel()
    iDv = sp.diags(np.where(dv > 0, 1.0 / np.sqrt(np.maximum(dv, 1e-12)), 0.0))
    iDe = sp.diags(np.where(de > 0, 1.0 / np.maximum(de, 1e-12), 0.0))
    return (sp.identity(len(nodes), format="csr") - iDv @ H @ iDe @ H.T @ iDv).tocsr(), nodes


def verify_operator_equivalence(coll: GroupCollection, tol: float = 1e-10) -> dict:
    """Check the hypergraph Laplacian against its weighted-pairwise counterpart."""
    L_hyper, nodes = hypergraph_laplacian(coll)
    H, _ = incidence(coll, nodes)
    de = np.asarray(H.sum(axis=0)).ravel()
    iDe = sp.diags(np.where(de > 0, 1.0 / np.maximum(de, 1e-12), 0.0))
    A_equiv = (H @ iDe @ H.T).tocsr()          # self-loops retained, deliberately
    L_equiv = _norm_laplacian(A_equiv)
    A_unweighted, _ = pairwise_adjacency(coll, nodes, weight_by_size=False)
    L_unweighted = _norm_laplacian(A_unweighted)
    d_equiv = float(abs(L_hyper - L_equiv).max())
    d_unw = float(abs(L_hyper - L_unweighted).max())
    return {
        "max_abs_diff_vs_size_weighted_pairwise": d_equiv,
        "max_abs_diff_vs_unweighted_pairwise": d_unw,
        "equivalent": bool(d_equiv < tol),
        "distinct_from_unweighted": bool(d_unw > 1e-3),
    }


# --------------------------------------------------------------------------- #
# Community detection
# --------------------------------------------------------------------------- #
def _spectral_labels(L: sp.csr_matrix, k: int, seed: int = 0) -> np.ndarray:
    """Spectral clustering with a seeded starting vector.

    ``eigsh`` picks a random starting vector unless given one, which makes the
    eigenvectors -- and therefore the cluster assignments and the reported
    scores -- vary slightly between runs. Passing a seeded ``v0`` makes the
    whole layer reproducible.
    """
    from sklearn.cluster import KMeans
    n = L.shape[0]
    k_eff = max(2, min(k, n - 2))
    v0 = np.random.default_rng(seed).normal(size=n)
    vals, vecs = spl.eigsh(L.tocsc(), k=k_eff, sigma=-1e-5, which="LM", v0=v0)
    V = vecs[:, np.argsort(vals)]
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(V)


def community_benchmark(coll: GroupCollection, labels: dict[str, str],
                        seed: int = 0) -> list[dict]:
    """Cluster members; score against curated labels with NMI and ARI.

    ``labels`` maps member id -> ground-truth label. Members without a label,
    or with an ambiguous one, must be excluded by the caller: scoring against
    partial labels silently penalises every method equally and hides the signal.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    nodes = [m for m in coll.members() if m in labels]
    gold = np.unique([labels[m] for m in nodes], return_inverse=True)[1]
    k = len(set(gold))

    A_unw, _ = pairwise_adjacency(coll, nodes, weight_by_size=False)
    A_w, _ = pairwise_adjacency(coll, nodes, weight_by_size=True)
    L_hyper, _ = hypergraph_laplacian(coll, nodes)

    variants = {
        "pairwise_unweighted": _norm_laplacian(A_unw),
        "pairwise_size_weighted": _norm_laplacian(A_w),
        "hypergraph_laplacian": L_hyper,
    }
    rows = []
    for name, L in variants.items():
        pred = _spectral_labels(L, k, seed=seed)
        rows.append({
            "method": name, "k": k, "n_nodes": len(nodes),
            "nmi": float(normalized_mutual_info_score(gold, pred)),
            "ari": float(adjusted_rand_score(gold, pred)),
        })
    return rows


# --------------------------------------------------------------------------- #
# Group-formation prediction
# --------------------------------------------------------------------------- #
def candidate_triples(coll: GroupCollection, cap: int | None = None
                      ) -> tuple[set[frozenset[str]], nx.Graph]:
    """Protein triples whose three pairs are all known but which never co-occur.

    These are the open cases: every pair has been seen inside some complex, yet
    no single complex contains all three. A prediction task asks which of them
    later close.
    """
    G = nx.Graph()
    for g in coll:
        for a, b in g.pairs():
            if G.has_edge(a, b):
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, weight=1)
    closed = {frozenset(t) for g in coll if g.size >= 3
              for t in itertools.combinations(sorted(g.members), 3)}
    out: set[frozenset[str]] = set()
    for v in G:
        nbrs = sorted(G.neighbors(v))
        if cap and len(nbrs) > cap:
            continue
        for a, b in itertools.combinations(nbrs, 2):
            if G.has_edge(a, b):
                t = frozenset((v, a, b))
                if t not in closed:
                    out.add(t)
    return out, G


def _scores(triples: Sequence[frozenset[str]], G: nx.Graph,
            coll: GroupCollection) -> dict[str, np.ndarray]:
    """Matched-capacity scorers: all are simple local statistics."""
    harm, geom, arith, common, pref = [], [], [], [], []
    deg = dict(G.degree())
    for t in triples:
        a, b, c = sorted(t)
        w = [G[x][y]["weight"] for x, y in ((a, b), (a, c), (b, c))]
        harm.append(3.0 / sum(1.0 / max(x, 1e-9) for x in w))
        geom.append(float(np.prod(w)) ** (1 / 3))
        arith.append(float(np.mean(w)))
        nb = [set(G.neighbors(x)) for x in (a, b, c)]
        common.append(len(nb[0] & nb[1] & nb[2]))
        pref.append(deg[a] * deg[b] * deg[c])
    return {
        "tie_strength_harmonic": np.array(harm),
        "tie_strength_geometric": np.array(geom),
        "tie_strength_arithmetic": np.array(arith),
        "common_neighbours": np.array(common, dtype=float),
        "preferential_attachment": np.array(pref, dtype=float),
    }


def formation_prediction(coll: GroupCollection, quantile: float = 0.5,
                         seed: int = 0, n_null: int = 2000) -> dict:
    """Chronological prediction of which open triples later form a complex.

    **Why the split defaults to the median rather than 80/20.** An 80/20 split
    is conventional, but it is the wrong default when positives are scarce: the
    later period must contain enough newly formed groups for any score to be
    distinguishable from chance, and pushing the cut-off late leaves too few.
    A median split keeps the task honest at the cost of training on less
    history. Both remain available via ``quantile``.

    **Why lift alone is not reported.** With a handful of positives, a scorer
    that ranks at random still produces a wide spread of average precision, so
    a lift of two or three over the base rate can easily be noise. This function
    estimates the null distribution directly -- ``n_null`` random rankings of
    the same labels -- and reports an empirical p-value per scorer. A result
    that does not clear the null is reported as indistinguishable from chance,
    not as a weak positive.
    """
    from sklearn.metrics import average_precision_score

    early, late, cut, basis = coll.split_by_time(quantile)
    triples, G = candidate_triples(early)
    later_closed = {frozenset(t) for g in late if g.size >= 3
                    for t in itertools.combinations(sorted(g.members), 3)}
    triples = sorted(triples, key=lambda s: tuple(sorted(s)))
    y = np.array([1 if t in later_closed else 0 for t in triples])
    base_rate = float(y.mean()) if y.size else 0.0

    rows: list[dict] = []
    null_summary: dict = {}
    if y.sum() > 0:
        rng = np.random.default_rng(seed)
        null_aps = np.array([average_precision_score(y, rng.random(len(y)))
                             for _ in range(n_null)])
        p95 = float(np.percentile(null_aps, 95))
        null_summary = {
            "draws": n_null,
            "mean_lift": float(null_aps.mean() / base_rate) if base_rate else float("nan"),
            "p95_lift": float(p95 / base_rate) if base_rate else float("nan"),
            "p99_lift": float(np.percentile(null_aps, 99) / base_rate)
            if base_rate else float("nan"),
        }
        for name, s in _scores(triples, G, early).items():
            ap = float(average_precision_score(y, s))
            pval = float((null_aps >= ap).mean())
            rows.append({
                "scorer": name, "average_precision": ap,
                "lift_over_random": ap / base_rate if base_rate else float("nan"),
                "p_value": pval,
                "beats_chance": bool(pval < 0.05),
            })
        rows.sort(key=lambda r: -r["average_precision"])
    return {
        "basis": basis, "cut": cut, "quantile": quantile,
        "n_early_groups": len(early), "n_late_groups": len(late),
        "n_candidate_triples": len(triples),
        "n_positives": int(y.sum()), "base_rate": base_rate,
        "null_distribution": null_summary,
        "scores": rows,
        "n_beating_chance": sum(1 for r in rows if r["beats_chance"]),
        "underpowered": bool(y.sum() < 50),
    }
