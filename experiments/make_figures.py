"""Figures for the study write-up. Reads ``results/study_results.json``.

Four panels, each tied to one claim in the report:

1. complex size distribution -- how much of the data is higher-order at all
2. recovery rate by complex size -- where dissolution fails
3. storage footprint by representation -- nodes versus edges
4. query expressibility -- which questions each representation can answer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = {"dissolved": "#c1666b", "reified": "#4f6d7a", "native": "#48a9a6"}
LABEL = {"dissolved": "Pairwise graph", "reified": "Hub node", "native": "Hyperedge"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results/study_results.json")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args()

    R = json.loads(Path(args.results).read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    # 1 -- size distribution -------------------------------------------------
    ax = axes[0, 0]
    hist = {int(k): v for k, v in R["descriptors"]["size_histogram"].items()}
    ks = sorted(hist)
    ax.bar(ks, [hist[k] for k in ks], color="#4f6d7a")
    ax.set_yscale("log")
    ax.set_xlabel("subunits per complex")
    ax.set_ylabel("number of complexes (log)")
    ax.set_title(f"A. Complex sizes (mean {R['descriptors']['mean_group_size']:.2f})")
    ax.axvline(2.5, color="#c1666b", ls="--", lw=1)
    ax.text(2.7, max(hist.values()) * 0.5,
            f"{R['descriptors']['frac_above_pair']:.0%} have\n3 or more",
            fontsize=8, color="#c1666b")
    ax.grid(alpha=.25, axis="y")

    # 2 -- recovery by size --------------------------------------------------
    ax = axes[0, 1]
    dis = next(f for f in R["fidelity"] if f["representation"] == "dissolved")
    bs = {int(k): v for k, v in dis["by_size"].items() if int(k) > 0}
    ks = sorted(bs)
    ax.plot(ks, [bs[k]["rate"] for k in ks], "o-", color="#c1666b", label="Pairwise graph")
    ax.axhline(1.0, color="#48a9a6", lw=2, label="Hub node / hyperedge")
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("subunits per complex")
    ax.set_ylabel("fraction recovered exactly")
    ax.set_title(f"B. Recovery after dissolving into pairs (overall F1 {dis['f1']:.2f})")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=.25)

    # 3 -- footprint ---------------------------------------------------------
    ax = axes[1, 0]
    fp = {r["representation"]: r for r in R["footprint"]}
    reps = ["dissolved", "reified", "native"]
    x = np.arange(len(reps))
    ax.bar(x - 0.2, [fp[r]["nodes"] for r in reps], 0.4, label="nodes", color="#4f6d7a")
    ax.bar(x + 0.2, [fp[r]["edges"] for r in reps], 0.4, label="edges / hyperedges",
           color="#c1666b")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[r] for r in reps], fontsize=9)
    ax.set_ylabel("count")
    ax.set_title("C. Storage footprint")
    ax.legend(fontsize=8)
    ax.grid(alpha=.25, axis="y")
    for i, r in enumerate(reps):
        ax.text(i - 0.2, fp[r]["nodes"], f"{fp[r]['nodes']:,}", ha="center",
                va="bottom", fontsize=7)
        ax.text(i + 0.2, fp[r]["edges"], f"{fp[r]['edges']:,}", ha="center",
                va="bottom", fontsize=7)

    # 4 -- query expressibility ---------------------------------------------
    ax = axes[1, 1]
    piv = R["queries_pivot"]
    fams = list(next(iter(piv.values())).keys())
    x = np.arange(len(fams))
    w = 0.26
    for i, rep in enumerate(reps):
        ax.bar(x + (i - 1) * w, [piv[rep][f] for f in fams], w,
               label=LABEL[rep], color=PALETTE[rep])
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in fams], fontsize=7.5)
    ax.set_ylabel("answer accuracy (F1)")
    ax.set_ylim(0, 1.1)
    ax.set_title("D. Which questions can be answered")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=.25, axis="y")

    fig.suptitle("Protein complexes: pairwise graph vs group-preserving representations",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out / "study_overview.png"
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
