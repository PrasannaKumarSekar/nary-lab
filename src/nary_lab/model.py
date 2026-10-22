"""The canonical data model.

Everything in this framework is built from one structure: a **group-incidence
table**. One row per (group, member) pair.

    group_id   group_type   role     member_id   evidence   timestamp
    CORUM:1    Complex      subunit  A0A510GDE0  11929873   2002

This is deliberately the same shape as a Neo4j relationship-import CSV in which
the group is the ``:START_ID``. A pipeline that already emits reified CSVs can
produce it with a projection rather than a rewrite.

Why one canonical form
----------------------
The comparison this framework performs is between *representations* of the same
facts. If each representation were loaded from its own source file, any measured
difference could be a difference in the data rather than in the encoding. Making
every representation a pure function of one ``GroupCollection`` removes that
confound by construction.

``GroupCollection`` is intentionally dumb: it holds groups and members and knows
nothing about proteins, complexes or biology. Domain loaders live in
:mod:`nary_lab.io`.
"""

from __future__ import annotations

import collections
import itertools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np

__all__ = ["Group", "GroupCollection"]


@dataclass(frozen=True)
class Group:
    """One higher-order fact: a set of members asserted to belong together.

    Parameters
    ----------
    id:
        Stable identifier for the group (e.g. ``CORUM:1``).
    members:
        Frozen set of member identifiers. A set, not a list: two groups with
        the same members are the same group as far as structure is concerned,
        even if they carry different identifiers.
    group_type:
        Free-form label (``Complex``, ``Reaction``, ...). Kept so that a single
        collection can hold several kinds of fact.
    roles:
        Optional mapping ``member_id -> role``. Empty for flat sets such as
        complex membership, populated for facts whose participants play
        distinguishable parts (a reaction's substrate versus its product).
    evidence:
        Provenance identifiers, e.g. PubMed IDs.
    timestamp:
        Numeric time (year) when known, else ``None``.
    order_key:
        Ordinal stand-in for time, used when real timestamps are unavailable.
        Kept separate from ``timestamp`` so that an approximation can never be
        silently mistaken for a measurement.
    """

    id: str
    members: frozenset[str]
    group_type: str = "Group"
    roles: tuple[tuple[str, str], ...] = ()
    evidence: tuple[str, ...] = ()
    timestamp: float | None = None
    order_key: float | None = None

    @property
    def size(self) -> int:
        """Number of members. Referred to throughout as *group size*."""
        return len(self.members)

    @property
    def role_map(self) -> dict[str, str]:
        return dict(self.roles)

    def pairs(self) -> Iterator[tuple[str, str]]:
        """Every unordered pair of members — what dissolution would produce."""
        yield from itertools.combinations(sorted(self.members), 2)

    def has_roles(self) -> bool:
        return len(self.roles) > 0


class GroupCollection:
    """An ordered collection of :class:`Group` objects plus derived indices."""

    def __init__(self, groups: Iterable[Group], name: str = "collection") -> None:
        self.name = name
        self._groups: dict[str, Group] = {}
        for g in groups:
            if g.id in self._groups:
                raise KeyError(f"duplicate group id {g.id!r}")
            self._groups[g.id] = g
        self._by_member: dict[str, set[str]] = collections.defaultdict(set)
        for g in self._groups.values():
            for m in g.members:
                self._by_member[m].add(g.id)

    # -- access ------------------------------------------------------------ #
    @property
    def groups(self) -> dict[str, Group]:
        return self._groups

    def __len__(self) -> int:
        return len(self._groups)

    def __iter__(self) -> Iterator[Group]:
        return iter(self._groups.values())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        s = self.sizes()
        return (f"<GroupCollection {self.name}: {len(self._groups)} groups, "
                f"{len(self.members())} members, mean size "
                f"{s.mean():.2f}>" if len(s) else f"<GroupCollection {self.name}: empty>")

    def members(self) -> list[str]:
        return sorted(self._by_member)

    def groups_of(self, member: str) -> set[str]:
        return self._by_member.get(member, set())

    def sizes(self) -> np.ndarray:
        return np.array([g.size for g in self._groups.values()], dtype=int)

    def member_sets(self) -> list[frozenset[str]]:
        return [g.members for g in self._groups.values()]

    def unique_member_sets(self) -> set[frozenset[str]]:
        return set(self.member_sets())

    def implied_pairs(self) -> set[tuple[str, str]]:
        """All pairs implied by co-membership — the dissolved edge set."""
        out: set[tuple[str, str]] = set()
        for g in self._groups.values():
            out.update(g.pairs())
        return out

    def n_incidences(self) -> int:
        return int(self.sizes().sum())

    def has_roles(self) -> bool:
        return any(g.has_roles() for g in self._groups.values())

    def has_timestamps(self) -> bool:
        return any(g.timestamp is not None for g in self._groups.values())

    # -- filtering ---------------------------------------------------------- #
    def filter_size(self, min_size: int = 2, max_size: int | None = None,
                    name: str | None = None) -> tuple[GroupCollection, dict[str, int]]:
        """Restrict to a group-size window.

        Returns the filtered collection *and* a count of what was dropped, so
        that a filtering decision is always visible in the results rather than
        buried in a script.
        """
        kept, dropped = [], collections.Counter()
        for g in self._groups.values():
            if g.size < min_size:
                dropped["below_min_size"] += 1
            elif max_size is not None and g.size > max_size:
                dropped["above_max_size"] += 1
            else:
                kept.append(g)
        label = name or f"{self.name}[{min_size}..{max_size}]"
        return GroupCollection(kept, label), dict(dropped)

    def subset(self, group_ids: Iterable[str], name: str | None = None) -> GroupCollection:
        ids = set(group_ids)
        return GroupCollection([g for g in self._groups.values() if g.id in ids],
                               name or f"{self.name}[subset]")

    def restrict_members(self, allowed: Iterable[str], min_size: int = 2,
                         name: str | None = None) -> GroupCollection:
        """Keep only the listed members; drop groups that fall below ``min_size``."""
        allow = set(allowed)
        out = []
        for g in self._groups.values():
            m = frozenset(g.members & allow)
            if len(m) >= min_size:
                roles = tuple((k, v) for k, v in g.roles if k in m)
                out.append(Group(g.id, m, g.group_type, roles, g.evidence,
                                 g.timestamp, g.order_key))
        return GroupCollection(out, name or f"{self.name}[restricted]")

    def split_by_time(self, quantile: float = 0.8
                      ) -> tuple[GroupCollection, GroupCollection, float, str]:
        """Chronological split for prediction tasks.

        Uses real ``timestamp`` when present, otherwise ``order_key``. The
        returned basis string records which was used so a downstream report can
        never present an ordinal proxy as a calendar date.
        """
        if self.has_timestamps():
            key, basis = (lambda g: g.timestamp), "timestamp"
        else:
            key, basis = (lambda g: g.order_key), "order_key (ordinal proxy)"
        usable = [g for g in self._groups.values() if key(g) is not None]
        if not usable:
            raise ValueError("no groups carry a timestamp or order_key")
        cut = float(np.quantile([key(g) for g in usable], quantile))
        early = GroupCollection([g for g in usable if key(g) <= cut], f"{self.name}[early]")
        late = GroupCollection([g for g in usable if key(g) > cut], f"{self.name}[late]")
        return early, late, cut, basis

    # -- serialisation ------------------------------------------------------ #
    def to_rows(self) -> list[dict]:
        rows = []
        for g in self._groups.values():
            rm = g.role_map
            for m in sorted(g.members):
                rows.append({
                    "group_id": g.id, "group_type": g.group_type,
                    "role": rm.get(m, ""), "member_id": m,
                    "evidence": ";".join(g.evidence),
                    "timestamp": g.timestamp if g.timestamp is not None else "",
                    "order_key": g.order_key if g.order_key is not None else "",
                })
        return rows

    @classmethod
    def from_rows(cls, rows: Iterable[dict], name: str = "collection") -> GroupCollection:
        acc: dict[str, dict] = {}
        for r in rows:
            gid = r["group_id"]
            e = acc.setdefault(gid, {
                "members": set(), "roles": {},
                "type": r.get("group_type", "Group"),
                "evidence": r.get("evidence", ""),
                "ts": r.get("timestamp", ""), "ok": r.get("order_key", ""),
            })
            e["members"].add(r["member_id"])
            if r.get("role"):
                e["roles"][r["member_id"]] = r["role"]
        groups = []
        for gid, e in acc.items():
            ts_raw, ok_raw = str(e["ts"]).strip(), str(e["ok"]).strip()
            ts = float(ts_raw) if ts_raw not in ("", "None") else None
            ok = float(ok_raw) if ok_raw not in ("", "None") else None
            ev = tuple(x for x in str(e["evidence"]).split(";") if x)
            groups.append(Group(gid, frozenset(e["members"]), e["type"],
                                tuple(e["roles"].items()), ev, ts, ok))
        return cls(groups, name)
