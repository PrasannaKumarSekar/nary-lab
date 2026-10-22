"""Command line interface.

    nary-lab profile        # Layer 0 only: descriptors and predicted verdict
    nary-lab fidelity       # Layer 1 round-trip table
    nary-lab pubmed         # resolve PubMed IDs to years (network required)
    nary-lab run            # full four-layer study + scorecard
"""
from __future__ import annotations

import argparse
import json
import sys

from .descriptors import describe, predict
from .fidelity import assess_all
from .io import load_complexes
from .pubmed import fetch_years
from .scorecard import to_markdown_table
from .study import StudyConfig, run_study


def _load(args):
    coll, rep = load_complexes(args.data_dir, member_column=args.member_column,
                               min_size=args.min_size, max_size=None)
    coll, dropped = coll.filter_size(args.min_size, args.max_size)
    return coll, rep, dropped


def cmd_profile(args) -> int:
    coll, rep, dropped = _load(args)
    d = describe(coll, compute_conformality=args.conformality)
    print(json.dumps(d.as_dict(), indent=2, default=float))
    print("\nPREDICTED:")
    for k, v in predict(d).items():
        print(f"  [{k}] {v}")
    print(f"\nload report: {rep.as_dict()}\nsize filter dropped: {dropped}")
    return 0


def cmd_fidelity(args) -> int:
    coll, _, _ = _load(args)
    rows = [f.as_dict() for f in assess_all(coll, clique_cap=args.max_size)]
    cols = ["representation", "n_true_groups", "exact", "precision", "recall", "f1",
            "invented", "vanished", "lossless", "claim_holds"]
    print(to_markdown_table(rows, cols))
    return 0


def cmd_pubmed(args) -> int:
    coll, _, _ = _load(args)
    pmids = {e for g in coll for e in g.evidence}
    print(f"{len(pmids)} distinct PubMed IDs")
    fetch_years(pmids, cache_path=args.cache, api_key=args.api_key,
                sleep=0.11 if args.api_key else 0.34)
    return 0


def cmd_run(args) -> int:
    cfg = StudyConfig(data_dir=args.data_dir, out_dir=args.out_dir,
                      member_column=args.member_column, min_group_size=args.min_size,
                      max_group_size=args.max_size, seed=args.seed)
    run_study(cfg)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nary-lab", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--data-dir", default="data")
        sp.add_argument(
            "--member-column", default=":END_ID",
            help="':END_ID' (graph node) or "
                 "'corum_subunit_uniprot_id:string' (source accession)")
        sp.add_argument("--min-size", type=int, default=2)
        sp.add_argument("--max-size", type=int, default=25)

    a = sub.add_parser("profile", help="Layer 0 descriptors")
    common(a)
    a.add_argument("--conformality", action="store_true")
    a.set_defaults(func=cmd_profile)

    b = sub.add_parser("fidelity", help="Layer 1 round trip")
    common(b)
    b.set_defaults(func=cmd_fidelity)

    c = sub.add_parser("pubmed", help="resolve PubMed IDs to years")
    common(c)
    c.add_argument("--cache", default="data/cache/pmid_year.json")
    c.add_argument("--api-key", default=None)
    c.set_defaults(func=cmd_pubmed)

    d = sub.add_parser("run", help="full study")
    common(d)
    d.add_argument("--out-dir", default="results")
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=cmd_run)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
