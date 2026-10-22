"""Resolving PubMed IDs to publication years.

Complexes in this dataset carry the PubMed ID of the article that first
described them. Turning those into years gives the study a time axis, which in
turn makes a genuine prediction task possible: train on what was known up to a
cut-off year and test against what was published afterwards.

Two paths are provided.

**Preferred — NCBI E-utilities.** :func:`fetch_years` queries the public
``esummary`` endpoint in batches, caches every result on disk, and resumes
after interruption. This is the path to use for any reported result.

**Fallback — ordinal proxy.** When the network is unavailable,
:func:`ordinal_proxy` ranks PubMed IDs numerically and uses the rank as a
stand-in for chronology. PubMed identifiers are assigned in broadly increasing
order as records are added, so the *ordering* is usually about right even
though the absolute years are unknown.

The two are deliberately kept in separate fields on the model (``timestamp``
versus ``order_key``) so an approximation can never be silently reported as a
measurement. :func:`describe_basis` returns a sentence naming which was used,
for inclusion in any output.

Known limitations of the proxy, which must be stated wherever it is used:

* Older articles were added to PubMed retrospectively, so a small number of
  early papers carry high identifiers and sort far later than they belong.
* Identifier assignment tracks *record creation*, not publication, so records
  near a boundary can be out of order by months.
* It supports relative ordering only. Any statement of the form "before 2015"
  requires real dates.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path

__all__ = ["fetch_years", "load_cache", "load_year_map", "ordinal_proxy",
           "attach_time", "describe_basis"]

ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
DEFAULT_CACHE = Path("data/cache/pmid_year.json")
#: Output of ``scripts/fetch_pubmed_dates.py``. Preferred over the JSON cache
#: because it is the artefact that gets shared and checksummed.
DEFAULT_DATES_CSV = Path("data/cache/pmid_dates.csv")


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def load_cache(path: Path | str = DEFAULT_CACHE) -> dict[str, int | None]:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as fh:
        return json.load(fh)


def load_dates_csv(path: Path | str = DEFAULT_DATES_CSV) -> dict[str, int | None]:
    """Read ``pmid_dates.csv`` as produced by ``scripts/fetch_pubmed_dates.py``.

    Rows with no resolved year are kept with a value of ``None`` so that the
    distinction between "not looked up" and "looked up, no date available" is
    not lost.
    """
    import csv as _csv

    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, int | None] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = _csv.DictReader(fh)
        if "pmid" not in (reader.fieldnames or []):
            raise ValueError(f"{path} has no 'pmid' column; columns are "
                             f"{reader.fieldnames}")
        for row in reader:
            pid = (row.get("pmid") or "").strip()
            year = (row.get("year") or "").strip()
            if pid:
                out[pid] = int(year) if year.isdigit() else None
    return out


def load_year_map(dates_csv: Path | str = DEFAULT_DATES_CSV,
                  cache_path: Path | str = DEFAULT_CACHE) -> tuple[dict[str, int | None], str]:
    """Resolve the PubMed year map, preferring the shared CSV over the cache.

    Returns the map and a short string naming where it came from, so that any
    report can state its provenance.
    """
    csv_map = load_dates_csv(dates_csv)
    if csv_map:
        return csv_map, f"pmid_dates.csv ({Path(dates_csv).name})"
    json_map = load_cache(cache_path)
    if json_map:
        return json_map, f"json cache ({Path(cache_path).name})"
    return {}, "none"


def _save_cache(cache: dict[str, int | None], path: Path | str = DEFAULT_CACHE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(cache, fh)
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# NCBI E-utilities
# --------------------------------------------------------------------------- #
def _parse_year(rec: dict) -> int | None:
    """Prefer the print/electronic publication date, fall back to entry date."""
    for key in ("pubdate", "epubdate", "sortpubdate"):
        val = rec.get(key) or ""
        for token in str(val).replace("/", " ").replace("-", " ").split():
            if token.isdigit() and 1800 <= int(token) <= 2200:
                return int(token)
    return None


def fetch_years(
    pmids: Iterable[str],
    cache_path: Path | str = DEFAULT_CACHE,
    batch_size: int = 200,
    api_key: str | None = None,
    sleep: float = 0.34,
    max_retries: int = 4,
    timeout: int = 30,
    verbose: bool = True,
) -> dict[str, int | None]:
    """Resolve PubMed IDs to publication years, caching as it goes.

    NCBI permits three requests per second without an API key and ten with one;
    ``sleep`` defaults to the unauthenticated rate. Supply ``api_key`` (free,
    from an NCBI account) and lower ``sleep`` to 0.11 to go faster.

    Failures are recorded as ``None`` in the cache so a rerun does not retry
    them forever; delete those entries to force a retry.
    """
    cache = load_cache(cache_path)
    todo = sorted({str(p).strip() for p in pmids if str(p).strip().isdigit()} - set(cache))
    if verbose:
        print(f"[pubmed] {len(cache)} cached, {len(todo)} to fetch")

    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        params = {"db": "pubmed", "id": ",".join(batch), "retmode": "json"}
        if api_key:
            params["api_key"] = api_key
        url = f"{ESUMMARY}?{urllib.parse.urlencode(params)}"

        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp:
                    payload = json.load(resp)
                result = payload.get("result", {})
                for pid in batch:
                    cache[pid] = _parse_year(result.get(pid, {})) if pid in result else None
                break
            except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError,
                    TimeoutError) as exc:
                wait = sleep * (2 ** attempt)
                if verbose:
                    print(f"[pubmed] batch {i // batch_size} attempt {attempt + 1} "
                          f"failed ({type(exc).__name__}); retrying in {wait:.1f}s")
                time.sleep(wait)
        else:
            for pid in batch:
                cache.setdefault(pid, None)
            if verbose:
                print(f"[pubmed] batch {i // batch_size} gave up; marked unresolved")

        _save_cache(cache, cache_path)
        time.sleep(sleep)
        if verbose and (i // batch_size) % 5 == 0:
            done = min(i + batch_size, len(todo))
            print(f"[pubmed] {done}/{len(todo)}")

    resolved = sum(1 for p in cache.values() if p)
    if verbose:
        print(f"[pubmed] resolved {resolved}/{len(cache)}")
    return cache


# --------------------------------------------------------------------------- #
# Ordinal fallback
# --------------------------------------------------------------------------- #
def ordinal_proxy(pmids: Iterable[str]) -> dict[str, float]:
    """Rank PubMed IDs numerically; the rank stands in for chronology."""
    nums = sorted({int(p) for p in pmids if str(p).strip().isdigit()})
    return {str(v): float(i) for i, v in enumerate(nums)}


# --------------------------------------------------------------------------- #
# Attaching time to a collection
# --------------------------------------------------------------------------- #
def attach_time(coll, cache_path: Path | str = DEFAULT_CACHE,
                dates_csv: Path | str = DEFAULT_DATES_CSV):
    """Return a copy of ``coll`` with ``timestamp`` and/or ``order_key`` filled.

    A group's time is the **earliest** of its evidence articles: the point at
    which it first entered the literature.

    Real years are used when available from ``scripts/fetch_pubmed_dates.py``
    output (or the JSON cache); otherwise only ``order_key`` is populated and
    the caller can tell the difference.
    """
    from .model import Group, GroupCollection

    cache, _ = load_year_map(dates_csv, cache_path)
    all_pmids = {e for g in coll for e in g.evidence}
    proxy = ordinal_proxy(all_pmids)

    out = []
    for g in coll:
        years = [cache[e] for e in g.evidence if cache.get(e)]
        ranks = [proxy[e] for e in g.evidence if e in proxy]
        out.append(Group(
            id=g.id, members=g.members, group_type=g.group_type, roles=g.roles,
            evidence=g.evidence,
            timestamp=float(min(years)) if years else None,
            order_key=float(min(ranks)) if ranks else None,
        ))
    return GroupCollection(out, coll.name)


def describe_basis(coll) -> str:
    """One sentence naming the time basis, for inclusion in reports."""
    n = len(coll)
    real = sum(1 for g in coll if g.timestamp is not None)
    if real:
        yrs = [g.timestamp for g in coll if g.timestamp is not None]
        return (f"Time axis: publication year resolved for {real}/{n} complexes "
                f"({min(yrs):.0f}-{max(yrs):.0f}).")
    ordn = sum(1 for g in coll if g.order_key is not None)
    return (f"Time axis: ORDINAL PROXY (PubMed identifier rank) for {ordn}/{n} complexes. "
            "Relative ordering only; no calendar years. Replace by running the "
            "PubMed resolver.")
