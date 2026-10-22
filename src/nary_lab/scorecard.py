"""Rendering: turn layer outputs into one readable page.

The scorecard is the framework's product. It puts the predicted verdict next to
the measured one so that disagreement between them is visible, which is how the
prediction heuristics get corrected in later iterations.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

__all__ = ["render_scorecard", "save_json", "to_markdown_table"]


def _d(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    return obj


def save_json(obj: Any, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=lambda o: _d(o) if _d(o) is not o else str(o))
    return path


def to_markdown_table(rows: list[dict], columns: list[str] | None = None,
                      floatfmt: str = "{:.3f}") -> str:
    if not rows:
        return "_(no rows)_"
    cols = columns or list(rows[0].keys())
    def fmt(v):
        if isinstance(v, float):
            return floatfmt.format(v)
        if isinstance(v, bool):
            return "yes" if v else "no"
        return "" if v is None else str(v)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def render_scorecard(result: dict[str, Any], width: int = 84) -> str:
    """One-page plain-text summary of a completed study run."""
    L: list[str] = []
    bar = "=" * width
    d = result["descriptors"]
    name = result.get("dataset", "dataset")

    L.append(bar)
    L.append(f"REPRESENTATION SCORECARD  |  {name}")
    L.append(bar)

    L.append("")
    L.append("LAYER 0  STRUCTURE (cheap descriptors, computed before any experiment)")
    L.append(f"  groups {d['n_groups']:,}   members {d['n_members']:,}   "
             f"incidences {d['n_incidences']:,}")
    L.append(f"  group size: mean {d['mean_group_size']:.2f}  median "
             f"{d['median_group_size']:.0f}  max {d['max_group_size']}")
    L.append(f"  pairs only {d['frac_pairs_only']:.0%} | three or more members "
             f"{d['frac_above_pair']:.0%}")
    L.append(f"  hub concentration {d['hub_concentration']:.1%} | recurrence "
             f"{d['recurrence']:.3f} | mean overlap {d['mean_overlap']:.3f}")
    if d.get("conformality") is not None:
        L.append(f"  conformality {d['conformality']:.3f}")
    for k, v in result.get("prediction", {}).items():
        L.append(f"  PREDICTED [{k}]: {v}")

    L.append("")
    L.append("LAYER 1  REPRESENTATIONAL FIDELITY (round trip, measured)")
    for f in result["fidelity"]:
        tag = "LOSSLESS" if f["lossless"] else "LOSSY"
        L.append(f"  {f['representation']:<10} {tag:<9} F1 {f['f1']:.3f}  "
                 f"invented {f['invented']:>6,}  vanished {f['vanished']:>6,}  "
                 f"claim holds: {'yes' if f['claim_holds'] else 'NO'}")
    bs = next((f["by_size"] for f in result["fidelity"]
               if f["representation"] == "dissolved"), {})
    if bs:
        parts = [f"{k}:{v['rate']:.2f}" for k, v in sorted(bs.items(), key=lambda x: int(x[0]))
                 if int(k) > 0]
        L.append("  dissolved recovery by group size -> " + "  ".join(parts))

    L.append("")
    L.append("LAYER 2  ANALYTICAL CAPABILITY")
    q = result.get("queries_pivot", {})
    for rep, fams in q.items():
        ok = sum(1 for v in fams.values() if v > 0.99)
        L.append(f"  queries  {rep:<10} {ok}/{len(fams)} answered exactly   " +
                 "  ".join(f"{k}:{v:.2f}" for k, v in fams.items()))
    for c in result.get("communities", []):
        L.append(f"  community  {c['method']:<24} NMI {c['nmi']:.3f}  ARI {c['ari']:.3f}")
    eq = result.get("equivalence")
    if eq:
        L.append(f"  operator equivalence: hypergraph vs size-weighted pairwise "
                 f"max|diff| {eq['max_abs_diff_vs_size_weighted_pairwise']:.2e} "
                 f"-> {'CONFIRMED' if eq['equivalent'] else 'NOT CONFIRMED'}")
    fp = result.get("formation")
    if fp:
        L.append(f"  formation prediction ({fp['basis']}): {fp['n_candidate_triples']:,} "
                 f"candidates, {fp['n_positives']} positives "
                 f"(base rate {fp['base_rate']:.2%})"
                 + ("  [UNDERPOWERED]" if fp["underpowered"] else ""))
        nd = fp.get("null_distribution") or {}
        if nd:
            L.append(f"      null distribution ({nd['draws']} random rankings): "
                     f"mean x{nd['mean_lift']:.2f}, 95th pct x{nd['p95_lift']:.2f} "
                     "- a scorer must clear this to mean anything")
        for s in fp.get("scores", []):
            flag = "beats chance" if s.get("beats_chance") else "within noise"
            L.append(f"      {s['scorer']:<26} AP {s['average_precision']:.4f}  "
                     f"lift x{s['lift_over_random']:.2f}  p={s.get('p_value', float('nan')):.3f}  "
                     f"{flag}")

    L.append("")
    L.append("LAYER 3  IMPLEMENTATION TRADE-OFFS")
    for r in result["footprint"]:
        L.append(f"  {r['representation']:<10} nodes {r['nodes']:>8,}  edges {r['edges']:>8,}"
                 f"   (nodes {r['node_ratio_vs_dissolved']:.2f}x, "
                 f"edges {r['edge_ratio_vs_dissolved']:.2f}x vs pairwise)")
    L.append(f"  tooling issues logged: {len(result.get('tooling', []))}")

    L.append("")
    L.append("VERDICT")
    for line in result.get("verdict", []):
        L.append(f"  {line}")
    L.append(bar)
    return "\n".join(L)
