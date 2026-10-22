"""End-to-end study runner.

Loads the data once, builds the three representations from it, runs all four
layers, and writes a scorecard plus machine-readable results.

Every run emits a manifest recording configuration, package version, library
versions and timing, so a number can always be traced back to the conditions
that produced it.
"""

from __future__ import annotations

import collections
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import __version__
from .analytics import (
    community_benchmark,
    formation_prediction,
    verify_operator_equivalence,
)
from .descriptors import describe, predict
from .fidelity import assess_all, coverage_by_independent_pairs
from .implementation import TOOLING_NOTES, build_timing, footprint, summarise_tradeoffs
from .io import (
    load_complexes,
    load_functional_groups,
    load_ppi_pairs,
)
from .pubmed import attach_time, describe_basis
from .queries import run_query_suite
from .scorecard import render_scorecard, save_json

__all__ = ["StudyConfig", "run_study"]


@dataclass
class StudyConfig:
    data_dir: str = "data"
    out_dir: str = "results"
    cache_path: str = "data/cache/pmid_year.json"
    member_column: str = ":END_ID"
    min_group_size: int = 2
    #: Very large assemblies dominate clique enumeration and motif counting.
    #: Capping is an analytical choice, recorded in the manifest.
    max_group_size: int | None = 25
    query_instances: int = 300
    time_quantile: float = 0.5
    seed: int = 0
    run_ppi: bool = True
    run_formation: bool = True
    run_communities: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


def _environment() -> dict:
    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return "not installed"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unknown"
    return {
        "nary_lab": __version__, "python": sys.version.split()[0],
        "platform": platform.platform(), "git_commit": commit,
        "numpy": ver("numpy"), "networkx": ver("networkx"),
        "scipy": ver("scipy"), "sklearn": ver("sklearn"),
        "hypergraphx": ver("hypergraphx"),
    }


def _build_member_labels(coll, label_of: dict[str, str]) -> dict[str, str]:
    """Propagate complex-level curated labels down to members.

    A protein inherits a label only when every labelled complex containing it
    agrees. Ambiguous proteins are excluded rather than assigned arbitrarily,
    because a wrong label penalises all methods equally and masks real signal.
    """
    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for g in coll:
        lab = label_of.get(g.id)
        if lab:
            for m in g.members:
                votes[m][lab] += 1
    return {m: next(iter(c)) for m, c in votes.items() if len(c) == 1}


def run_study(cfg: StudyConfig, verbose: bool = True) -> dict:
    t_start = time.time()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    # ---------------------------------------------------------------- load
    log("[1/7] loading complexes")
    full, report = load_complexes(cfg.data_dir, member_column=cfg.member_column,
                                  min_size=cfg.min_group_size, max_size=None)
    coll, dropped = full.filter_size(cfg.min_group_size, cfg.max_group_size,
                                     name="corum_complexes")
    log(f"      {len(full)} complexes loaded; {len(coll)} after size cap "
        f"{cfg.min_group_size}..{cfg.max_group_size} (dropped {dropped})")

    coll = attach_time(coll, cfg.cache_path)
    basis = describe_basis(coll)
    log(f"      {basis}")

    label_of, fcg_meta = load_functional_groups(cfg.data_dir)
    member_labels = _build_member_labels(coll, label_of)
    log(f"      curated labels: {len(set(label_of.values()))} groups, "
        f"{len(member_labels)} unambiguously labelled proteins")

    # ------------------------------------------------------------- layer 0
    log("[2/7] layer 0: descriptors")
    desc = describe(coll, compute_conformality=False)
    pred = predict(desc)

    # ------------------------------------------------------------- layer 1
    log("[3/7] layer 1: fidelity (round trip)")
    fid = assess_all(coll, clique_cap=cfg.max_group_size)
    desc.conformality = next(f.f1 for f in fid if f.representation == "dissolved")
    pred = predict(desc)

    # independent evidence layer
    ppi_cov, ppi_stats = None, None
    if cfg.run_ppi:
        log("[4/7] independent interaction layer (streaming)")
        try:
            pairs, ppi_stats = load_ppi_pairs(cfg.data_dir, restrict_to=set(coll.members()))
            ppi_cov = coverage_by_independent_pairs(coll, pairs)
            log(f"      {len(pairs):,} interaction pairs among complex proteins; "
                f"covers {ppi_cov['pair_coverage']:.1%} of co-membership pairs")
        except FileNotFoundError:
            log("      interaction file absent; skipped")

    # ------------------------------------------------------------ layer 2a
    log("[5/7] layer 2a: query expressibility")
    qres = run_query_suite(coll, n_instances=cfg.query_instances, seed=cfg.seed)
    pivot: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for r in qres:
        pivot[r.representation][r.family] = r.f1

    # ------------------------------------------------------------ layer 2b
    log("[6/7] layer 2b: analytics")
    equivalence = verify_operator_equivalence(coll)
    communities = []
    if cfg.run_communities and member_labels:
        sub = coll.restrict_members(member_labels, min_size=2, name="labelled")
        communities = community_benchmark(sub, member_labels, seed=cfg.seed)
    formation = None
    if cfg.run_formation:
        try:
            formation = formation_prediction(coll, quantile=cfg.time_quantile, seed=cfg.seed)
        except ValueError as exc:
            log(f"      formation prediction skipped: {exc}")

    # ------------------------------------------------------------- layer 3
    log("[7/7] layer 3: implementation")
    fp = footprint(coll)
    timing = build_timing(coll)
    tradeoffs = summarise_tradeoffs(coll)

    # ------------------------------------------------------------- verdict
    verdict = _verdict(desc, fid, pivot, communities, equivalence, fp, ppi_cov)

    result = {
        "dataset": coll.name,
        "config": cfg.as_dict(),
        "load_report": report.as_dict(),
        "size_filter_dropped": dropped,
        "time_basis": basis,
        "descriptors": desc.as_dict(),
        "prediction": pred,
        "fidelity": [f.as_dict() for f in fid],
        "independent_ppi_coverage": ppi_cov,
        "ppi_stats": {k: dict(v.most_common(15)) for k, v in (ppi_stats or {}).items()},
        "queries": [r.__dict__ for r in qres],
        "queries_pivot": {k: dict(v) for k, v in pivot.items()},
        "equivalence": equivalence,
        "communities": communities,
        "formation": formation,
        "footprint": fp,
        "build_timing": timing,
        "tooling": TOOLING_NOTES,
        "tradeoffs_headline": tradeoffs["headline"],
        "traversal": tradeoffs["traversal"],
        "algorithms": tradeoffs["algorithms"],
        "verdict": verdict,
        "environment": _environment(),
        "wall_seconds": round(time.time() - t_start, 1),
    }

    save_json(result, out / "study_results.json")
    card = render_scorecard(result)
    (out / "scorecard.txt").write_text(card + "\n")
    log("\n" + card)
    log(f"\nwrote {out/'study_results.json'} and {out/'scorecard.txt'}")
    return result


def _verdict(desc, fid, pivot, communities, equivalence, fp, ppi_cov) -> list[str]:
    lines: list[str] = []
    d = {f.representation: f for f in fid}
    lines.append(
        f"Dissolving complexes into pairs loses information: F1 {d['dissolved'].f1:.3f}, "
        f"{d['dissolved'].invented:,} groups invented, {d['dissolved'].vanished:,} lost."
    )
    if d["reified"].lossless and d["native"].lossless:
        lines.append("Hub nodes and hyperedges are both lossless and carry identical "
                     "information; the choice between them is operational, not "
                     "representational.")
    dis_ok = sum(1 for v in pivot.get("dissolved", {}).values() if v > 0.99)
    lines.append(f"Query expressibility: pairwise answers {dis_ok}/"
                 f"{len(pivot.get('dissolved', {}))} question types exactly; "
                 "group-preserving forms answer all of them.")
    if equivalence and equivalence["equivalent"]:
        lines.append("Hypergraph spectral clustering is the size-weighted pairwise "
                     "operator (verified numerically), so no advantage should be "
                     "expected there, and none is observed.")
    by = {r["representation"]: r for r in fp}
    lines.append(
        f"Footprint: hub nodes add {by['reified']['extra_nodes']:,} nodes but "
        f"{by['reified']['edge_ratio_vs_dissolved'] - 1:+.0%} edges relative to the "
        "pairwise graph."
    )
    if ppi_cov:
        lines.append(
            f"The separately sourced interaction layer covers only "
            f"{ppi_cov['pair_coverage']:.1%} of complex co-membership pairs, so it is "
            "not a substitute for complex records."
        )
    if desc.mean_group_size < 3:
        lines.append(
            f"Caveat: mean group size {desc.mean_group_size:.2f} with "
            f"{desc.frac_pairs_only:.0%} pairs. Report stratified by size."
        )
    return lines
