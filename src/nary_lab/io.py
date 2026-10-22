"""Loaders: CORUM Neo4j-style CSVs -> the canonical :class:`GroupCollection`.

The export consists of six files. Only two are strictly required to build the
core object; the rest supply labels, metadata and the independent
protein-interaction layer.

============================================================  ===================
file                                                          used for
============================================================  ===================
``protein-complex_protein_association_neo4j.csv``             the groups
``protein-complex_node_neo4j.csv``                            complex metadata
``functional-complex-group_protein-complex_association*.csv`` evaluation labels
``functional-complex-group_node_neo4j.csv``                   label names
``protein_node_neo4j.csv``                                    protein metadata
``protein_protein_association_neo4j.csv``                     independent PPI layer
============================================================  ===================

Identifier policy
-----------------
Complex membership is read from ``:START_ID`` / ``:END_ID``, i.e. the graph's
own resolved node identifiers, so that measurements describe the knowledge graph
as it actually exists. The source database's own accession is preserved in
``corum_subunit_uniprot_id`` and can be selected instead with
``member_column="corum_subunit_uniprot_id:string"`` for comparability with
published analyses of the source. The two differ slightly: the resolution step
is many-to-one, so a handful of distinct source accessions land on the same
graph node. :func:`load_complexes` reports that collapse rather than hiding it.
"""

from __future__ import annotations

import collections
import csv
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .model import Group, GroupCollection

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

__all__ = [
    "read_csv_rows", "load_complexes", "load_functional_groups",
    "load_complex_metadata", "load_ppi_pairs", "LoadReport",
]

SUBUNIT_FILE = "protein-complex_protein_association_neo4j.csv"
COMPLEX_NODE_FILE = "protein-complex_node_neo4j.csv"
FCG_EDGE_FILE = "functional-complex-group_protein-complex_association_neo4j.csv"
FCG_NODE_FILE = "functional-complex-group_node_neo4j.csv"
PROTEIN_NODE_FILE = "protein_node_neo4j.csv"
PPI_FILE = "protein_protein_association_neo4j.csv"

NULLS = {"", "none", "null", "nan", "na"}


def _clean(v: str | None) -> str:
    v = (v or "").strip()
    return "" if v.lower() in NULLS else v


def read_csv_rows(path: Path | str) -> Iterator[dict]:
    """Stream a CSV as dicts.

    ``utf-8-sig`` matters: several files in this export carry a byte-order mark,
    which otherwise attaches an invisible ``\\ufeff`` to the first column name
    and makes lookups on it fail for that file only.
    """
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        yield from csv.DictReader(fh)


@dataclass
class LoadReport:
    """What the loader saw and what it discarded. Always printed, never hidden."""

    rows: int = 0
    groups_built: int = 0
    members_distinct: int = 0
    source_accessions: int = 0
    accession_collapse: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    evidence_values: int = 0
    groups_without_evidence: int = 0

    def as_dict(self) -> dict:
        return {**self.__dict__, "dropped": dict(self.dropped)}


def load_complexes(
    data_dir: Path | str,
    member_column: str = ":END_ID",
    min_size: int = 2,
    max_size: int | None = None,
    name: str = "corum",
) -> tuple[GroupCollection, LoadReport]:
    """Build the group collection from ``has_subunit`` edges.

    ``max_size=None`` keeps everything. Capping is a real analytical choice
    (very large assemblies dominate any enumeration over member subsets), so it
    is left to the caller and recorded in the report when used.
    """
    path = Path(data_dir) / SUBUNIT_FILE
    members: dict[str, set[str]] = collections.defaultdict(set)
    evidence: dict[str, set[str]] = collections.defaultdict(set)
    src_acc: set[str] = set()
    node_ids: set[str] = set()
    rep = LoadReport()

    for row in read_csv_rows(path):
        rep.rows += 1
        gid = _clean(row[":START_ID"])
        mid = _clean(row[member_column])
        if not gid or not mid:
            rep.dropped["blank_id"] = rep.dropped.get("blank_id", 0) + 1
            continue
        members[gid].add(mid)
        node_ids.add(_clean(row.get(":END_ID", "")))
        acc = _clean(row.get("corum_subunit_uniprot_id:string", ""))
        if acc:
            src_acc.add(acc)
        pm = _clean(row.get("pmid:string", ""))
        if pm:
            for p in pm.replace(";", ",").split(","):
                p = p.strip()
                if p:
                    evidence[gid].add(p)

    rep.source_accessions = len(src_acc)
    rep.accession_collapse = len(src_acc) - len({n for n in node_ids if n})

    groups: list[Group] = []
    for gid, ms in members.items():
        if len(ms) < min_size:
            rep.dropped["below_min_size"] = rep.dropped.get("below_min_size", 0) + 1
            continue
        if max_size is not None and len(ms) > max_size:
            rep.dropped["above_max_size"] = rep.dropped.get("above_max_size", 0) + 1
            continue
        ev = tuple(sorted(evidence[gid]))
        groups.append(Group(id=gid, members=frozenset(ms), group_type="ProteinComplex",
                            evidence=ev))
    coll = GroupCollection(groups, name)
    rep.groups_built = len(coll)
    rep.members_distinct = len(coll.members())
    rep.evidence_values = len({e for g in coll for e in g.evidence})
    rep.groups_without_evidence = sum(1 for g in coll if not g.evidence)
    return coll, rep


def load_functional_groups(data_dir: Path | str) -> tuple[dict[str, str], dict[str, dict]]:
    """Evaluation labels: complex -> functional group, plus group metadata.

    These are curated groupings of complexes by shared biological function.
    They are used *only* to score clustering results — never as input to any
    method — so that the evaluation stays independent of the representation.
    """
    data_dir = Path(data_dir)
    label_of: dict[str, str] = {}
    for row in read_csv_rows(data_dir / FCG_EDGE_FILE):
        label_of[_clean(row[":END_ID"])] = _clean(row[":START_ID"])
    meta: dict[str, dict] = {}
    fcg_nodes = data_dir / FCG_NODE_FILE
    if fcg_nodes.exists():
        for row in read_csv_rows(fcg_nodes):
            meta[_clean(row["id:ID"])] = {
                "name": _clean(row.get("name:string", "")),
                "category": _clean(row.get("functional_category:string", "")),
            }
    return label_of, meta


def load_complex_metadata(data_dir: Path | str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in read_csv_rows(Path(data_dir) / COMPLEX_NODE_FILE):
        out[_clean(row["id:ID"])] = {
            "name": _clean(row.get("name:string", "")),
            "organism": _clean(row.get("organism:string", "")),
            "context": _clean(row.get("experimental_context:string", "")),
        }
    return out


def load_ppi_pairs(
    data_dir: Path | str,
    restrict_to: set[str] | None = None,
    methods_exclude: set[str] | None = None,
) -> tuple[set[tuple[str, str]], dict[str, collections.Counter]]:
    """Stream the independent protein-interaction layer.

    This is *not* a projection of the complex data — it is separately sourced
    evidence. Keeping it separate is the point: it lets the study ask how much
    of complex co-membership the interaction layer actually covers, which is a
    different question from how much a projection would lose.

    Returns the pair set and per-field counters (detection method, source
    database) computed over the retained rows.
    """
    path = Path(data_dir) / PPI_FILE
    pairs: set[tuple[str, str]] = set()
    stats = {"method": collections.Counter(), "source": collections.Counter(),
             "method_kept": collections.Counter(), "source_kept": collections.Counter()}
    excl = {m.lower() for m in (methods_exclude or set())}
    for row in read_csv_rows(path):
        a, b = _clean(row[":START_ID"]), _clean(row[":END_ID"])
        method = _clean(row.get("interaction_detection_method:string", ""))
        source = _clean(row.get("source_database:string", ""))
        stats["method"][method or "(blank)"] += 1
        stats["source"][source or "(blank)"] += 1
        if not a or not b or a == b:
            continue
        if restrict_to is not None and (a not in restrict_to or b not in restrict_to):
            continue
        if method.lower() in excl:
            continue
        pairs.add(tuple(sorted((a, b))))
        stats["method_kept"][method or "(blank)"] += 1
        stats["source_kept"][source or "(blank)"] += 1
    return pairs, stats
