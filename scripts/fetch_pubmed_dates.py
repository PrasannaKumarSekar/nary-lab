#!/usr/bin/env python3
"""Fetch publication dates for the PubMed IDs referenced by the CORUM export.

Standalone by design: Python 3.9+ and the standard library only. No install, no
dependencies, nothing from this repository. Copy the file anywhere and run it.

    python fetch_pubmed_dates.py --data-dir /path/to/corum/csvs

Output lands in ``--out-dir`` (default ``pubmed_out/``):

    pmid_dates.csv       one row per PubMed ID -- this is the file to share back
    pmid_cache.json      raw responses, used to resume; not needed downstream
    fetch_report.json    coverage, failures, timing, and quality checks

Why a separate script
---------------------
The dates come from a live external service, so the fetch cannot be part of a
reproducible analysis run. Separating it means the analysis depends on a file
with a checksum rather than on whatever NCBI returned that afternoon.

Rate limits
-----------
NCBI allows 3 requests/second anonymously and 10 with a free API key. At the
default batch size of 200 identifiers per request, a few thousand IDs take
well under a minute either way. Get a key from your NCBI account settings and
pass ``--api-key`` if you plan to rerun this often.

Safe to interrupt and rerun: completed batches are cached.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
SUBUNIT_FILE = "protein-complex_protein_association_neo4j.csv"
PMID_COLUMN = "pmid:string"
TOOL = "nary_lab-pubmed-fetch"

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
SEASONS = {"spring": 4, "summer": 7, "fall": 10, "autumn": 10, "winter": 1}

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))


# --------------------------------------------------------------------------- #
# 1. Collect the identifiers
# --------------------------------------------------------------------------- #
def collect_pmids(data_dir: Path, column: str = PMID_COLUMN) -> tuple[list[str], dict]:
    """Read PubMed IDs out of the CORUM subunit edge file.

    The column holds one or more identifiers; separators seen in practice are
    commas and semicolons. Anything non-numeric is dropped and counted.
    """
    path = data_dir / SUBUNIT_FILE
    if not path.exists():
        raise SystemExit(f"could not find {path}\n"
                         f"point --data-dir at the directory holding {SUBUNIT_FILE}")

    pmids: dict[str, int] = {}
    stats = {"rows": 0, "values_seen": 0, "non_numeric": 0, "blank": 0}
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as fh:
        reader = csv.DictReader(fh)
        if column not in (reader.fieldnames or []):
            raise SystemExit(f"column {column!r} not in {path.name}; "
                             f"columns are {reader.fieldnames}")
        for row in reader:
            stats["rows"] += 1
            raw = (row.get(column) or "").strip()
            if not raw or raw.lower() in {"none", "null", "nan", "na"}:
                stats["blank"] += 1
                continue
            for token in re.split(r"[;,|\s]+", raw):
                token = token.strip()
                if not token:
                    continue
                stats["values_seen"] += 1
                if token.isdigit():
                    pmids[token] = pmids.get(token, 0) + 1
                else:
                    stats["non_numeric"] += 1
    stats["distinct_pmids"] = len(pmids)
    return sorted(pmids, key=int), stats


# --------------------------------------------------------------------------- #
# 2. Parse one esummary record
# --------------------------------------------------------------------------- #
def parse_record(rec: dict) -> dict:
    """Extract year and month from an esummary record.

    PubMed date fields are irregular. Observed forms include ``2002 Apr 15``,
    ``2015 Dec``, ``1998 Spring``, ``1976 Nov-Dec`` and bare ``2009``. There are
    also three different date fields, which can disagree:

    ``sortpubdate``  normalised ``YYYY/MM/DD``; the field NCBI itself sorts on
    ``pubdate``      the printed publication date, free-form
    ``epubdate``     electronic publication, often earlier than print

    We prefer ``sortpubdate`` for consistency and fall back in the order above.
    All three raw strings are kept in the output so the choice can be revisited
    without refetching.
    """
    out = {
        "year": None, "month": None, "date_source": None,
        "raw_sortpubdate": (rec.get("sortpubdate") or "").strip(),
        "raw_pubdate": (rec.get("pubdate") or "").strip(),
        "raw_epubdate": (rec.get("epubdate") or "").strip(),
        "journal": (rec.get("source") or "").strip(),
        "error": (rec.get("error") or "").strip(),
    }
    for field, key in (("sortpubdate", "raw_sortpubdate"),
                       ("pubdate", "raw_pubdate"),
                       ("epubdate", "raw_epubdate")):
        year, month = _parse_date_string(out[key])
        if year is not None:
            out["year"], out["month"], out["date_source"] = year, month, field
            return out
    return out


def _parse_date_string(value: str) -> tuple[int | None, int | None]:
    if not value:
        return None, None
    tokens = re.split(r"[\s/\-]+", value.strip())
    year = month = None
    for tok in tokens:
        low = tok.lower()
        if year is None and tok.isdigit() and 1800 <= int(tok) <= 2200:
            year = int(tok)
            continue
        if month is None:
            if low[:3] in MONTHS:
                month = MONTHS[low[:3]]
            elif low in SEASONS:
                month = SEASONS[low]
            elif tok.isdigit() and 1 <= int(tok) <= 12 and year is not None:
                month = int(tok)
    return year, month


# --------------------------------------------------------------------------- #
# 3. Fetch
# --------------------------------------------------------------------------- #
def _request(ids: list[str], api_key: str | None, email: str | None,
             timeout: int) -> dict:
    """One esummary call. POST so that large batches never hit URL limits."""
    params = {"db": "pubmed", "id": ",".join(ids), "retmode": "json", "tool": TOOL}
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(ESUMMARY, data=data,
                                 headers={"User-Agent": TOOL})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def fetch(pmids: list[str], cache: dict[str, dict], api_key: str | None,
          email: str | None, batch_size: int, sleep: float,
          max_retries: int, timeout: int, cache_path: Path,
          verbose: bool = True) -> dict:
    todo = [p for p in pmids if p not in cache]
    report = {"requested": len(pmids), "already_cached": len(pmids) - len(todo),
              "fetched": 0, "failed_batches": 0, "http_errors": []}
    if not todo:
        if verbose:
            print("all identifiers already cached; nothing to fetch")
        return report

    n_batches = (len(todo) + batch_size - 1) // batch_size
    if verbose:
        print(f"fetching {len(todo):,} identifiers in {n_batches} batch(es) "
              f"of up to {batch_size}")

    t0 = time.time()
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        bno = i // batch_size + 1
        payload = None
        for attempt in range(1, max_retries + 1):
            try:
                payload = _request(batch, api_key, email, timeout)
                break
            except urllib.error.HTTPError as exc:
                report["http_errors"].append({"batch": bno, "code": exc.code,
                                              "attempt": attempt})
                # 429 is rate limiting; back off harder and keep going
                wait = sleep * (2 ** attempt) + random.random()
                if verbose:
                    print(f"  batch {bno}: HTTP {exc.code}, retry {attempt}/"
                          f"{max_retries} in {wait:.1f}s")
                time.sleep(wait)
            except (urllib.error.URLError, json.JSONDecodeError, TimeoutError,
                    ConnectionError) as exc:
                wait = sleep * (2 ** attempt) + random.random()
                if verbose:
                    print(f"  batch {bno}: {type(exc).__name__}, retry {attempt}/"
                          f"{max_retries} in {wait:.1f}s")
                time.sleep(wait)

        if payload is None:
            report["failed_batches"] += 1
            if verbose:
                print(f"  batch {bno}: FAILED after {max_retries} attempts; "
                      "leaving these identifiers unfetched so a rerun retries them")
        else:
            result = payload.get("result", {})
            for pid in batch:
                rec = result.get(pid)
                cache[pid] = parse_record(rec) if isinstance(rec, dict) else {
                    "year": None, "month": None, "date_source": None,
                    "raw_sortpubdate": "", "raw_pubdate": "", "raw_epubdate": "",
                    "journal": "", "error": "not returned by esummary"}
                report["fetched"] += 1
            _write_json(cache, cache_path)

        time.sleep(sleep)
        if verbose and (bno % 5 == 0 or bno == n_batches):
            done = min(i + batch_size, len(todo))
            rate = done / max(time.time() - t0, 1e-9)
            print(f"  {done:,}/{len(todo):,}  ({rate:.0f} ids/s)")

    report["seconds"] = round(time.time() - t0, 1)
    return report


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# 4. Quality checks
# --------------------------------------------------------------------------- #
def quality_checks(cache: dict[str, dict], pmids: list[str]) -> dict:
    """Checks worth running before trusting the output.

    The ordering check is the important one. The analysis currently stands in
    for publication date by ranking PubMed identifiers numerically, on the
    assumption that identifiers are issued in roughly chronological order.
    Older articles were indexed retrospectively, so that assumption is known to
    break somewhere -- this measures how often, which tells us how much the
    placeholder distorted the earlier results.
    """
    resolved = {p: cache[p] for p in pmids if cache.get(p, {}).get("year")}
    years = sorted(v["year"] for v in resolved.values())
    checks: dict = {
        "requested": len(pmids),
        "resolved": len(resolved),
        "unresolved": len(pmids) - len(resolved),
        "coverage": round(len(resolved) / max(len(pmids), 1), 4),
    }
    if years:
        checks["year_min"] = years[0]
        checks["year_max"] = years[-1]
        checks["year_median"] = years[len(years) // 2]
        checks["implausible_years"] = sum(1 for y in years if y < 1950 or y > 2030)
        decade: dict[str, int] = {}
        for y in years:
            key = f"{(y // 10) * 10}s"
            decade[key] = decade.get(key, 0) + 1
        checks["by_decade"] = dict(sorted(decade.items()))

    # does PubMed ID order match publication order?
    ordered = sorted(resolved.items(), key=lambda kv: int(kv[0]))
    seq = [v["year"] for _, v in ordered]
    inversions = sum(1 for a, b in zip(seq, seq[1:], strict=False) if b < a)
    checks["pmid_order_inversions"] = inversions
    checks["pmid_order_inversion_rate"] = round(inversions / max(len(seq) - 1, 1), 4)
    far = [(p, v["year"]) for p, v in ordered if v["year"] and seq and
           abs(v["year"] - seq[len(seq) // 2]) > 40]
    checks["far_outliers_sample"] = far[:5]

    sources: dict[str, int] = {}
    for v in resolved.values():
        k = v.get("date_source") or "unknown"
        sources[k] = sources.get(k, 0) + 1
    checks["date_field_used"] = sources

    errs: dict[str, int] = {}
    for p in pmids:
        e = cache.get(p, {}).get("error") or ""
        if e:
            errs[e] = errs.get(e, 0) + 1
    checks["error_messages"] = errs
    return checks


# --------------------------------------------------------------------------- #
# 5. Output
# --------------------------------------------------------------------------- #
FIELDS = ["pmid", "year", "month", "date_source", "raw_sortpubdate",
          "raw_pubdate", "raw_epubdate", "journal", "error"]


def write_csv(cache: dict[str, dict], pmids: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for p in pmids:
            rec = cache.get(p, {})
            w.writerow({"pmid": p, **{k: rec.get(k, "") for k in FIELDS[1:]}})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data",
                    help=f"directory containing {SUBUNIT_FILE}")
    ap.add_argument("--out-dir", default="pubmed_out")
    ap.add_argument("--pmid-column", default=PMID_COLUMN)
    ap.add_argument("--api-key", default=os.environ.get("NCBI_API_KEY"),
                    help="NCBI API key; raises the rate limit from 3/s to 10/s")
    ap.add_argument("--email", default=os.environ.get("NCBI_EMAIL"),
                    help="contact address; NCBI asks for one on automated use")
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--sleep", type=float, default=None,
                    help="seconds between requests (default 0.11 with a key, else 0.35)")
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None,
                    help="only fetch the first N identifiers (for a quick trial run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="collect identifiers and report, but make no requests")
    args = ap.parse_args(argv)

    sleep = args.sleep if args.sleep is not None else (0.11 if args.api_key else 0.35)
    out_dir = Path(args.out_dir)
    cache_path = out_dir / "pmid_cache.json"

    print("=" * 72)
    print("CORUM PubMed date fetch")
    print("=" * 72)

    pmids, collect_stats = collect_pmids(Path(args.data_dir), args.pmid_column)
    print(f"rows read               : {collect_stats['rows']:,}")
    print(f"identifier values seen  : {collect_stats['values_seen']:,}")
    print(f"non-numeric discarded   : {collect_stats['non_numeric']:,}")
    print(f"distinct PubMed IDs     : {collect_stats['distinct_pmids']:,}")
    if args.limit:
        pmids = pmids[:args.limit]
        print(f"limited to              : {len(pmids):,}")
    rate = f"{1/sleep:.1f} req/s" if sleep > 0 else "unthrottled"
    print(f"rate                    : {rate} "
          f"({'with' if args.api_key else 'no'} API key)")

    if args.dry_run:
        print("\n--dry-run: stopping before any network request")
        return 0

    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if cache:
        print(f"resuming from cache     : {len(cache):,} already fetched")

    print()
    report = fetch(pmids, cache, args.api_key, args.email, args.batch_size,
                   sleep, args.max_retries, args.timeout, cache_path)
    _write_json(cache, cache_path)

    csv_path = out_dir / "pmid_dates.csv"
    write_csv(cache, pmids, csv_path)
    checks = quality_checks(cache, pmids)
    _write_json({"collect": collect_stats, "fetch": report, "checks": checks},
                out_dir / "fetch_report.json")

    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)
    print(f"resolved                : {checks['resolved']:,}/{checks['requested']:,} "
          f"({checks['coverage']:.1%})")
    if checks.get("year_min"):
        print(f"year range              : {checks['year_min']}-{checks['year_max']} "
              f"(median {checks['year_median']})")
        print(f"by decade               : {checks['by_decade']}")
        print(f"implausible years       : {checks['implausible_years']}")
    print(f"date field used         : {checks['date_field_used']}")
    print("PubMed-ID order vs publication order:")
    print(f"  out-of-order adjacent pairs: {checks['pmid_order_inversions']:,} "
          f"({checks['pmid_order_inversion_rate']:.1%})")
    if report.get("failed_batches"):
        print(f"\nWARNING: {report['failed_batches']} batch(es) failed. "
              "Rerun this script to retry only those.")
    if checks["unresolved"]:
        print(f"\n{checks['unresolved']:,} identifier(s) unresolved. "
              "Common causes: retracted or withdrawn records, or identifiers that "
              "are not PubMed IDs. See the 'error' column.")

    print(f"\nwrote {csv_path}")
    print(f"      {out_dir/'fetch_report.json'}")
    print(f"      {cache_path}  (resume cache; no need to share)")
    print(f"\nShare {csv_path.name} and fetch_report.json back for the analysis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
