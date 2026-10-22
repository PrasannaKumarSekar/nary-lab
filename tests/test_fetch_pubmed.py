"""Tests for ``scripts/fetch_pubmed_dates.py``.

The script talks to an external service, so the parts that matter are tested
without touching the network: date parsing against the irregular formats PubMed
actually returns, batching and resume behaviour against a stubbed request
function, and the quality checks.
"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_pubmed_dates.py"
spec = importlib.util.spec_from_file_location("fetch_pubmed_dates", SCRIPT)
fp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fp)


# --------------------------------------------------------------------------- #
# Date parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value,expected", [
    ("2002/04/15 00:00", (2002, 4)),
    ("2002 Apr 15", (2002, 4)),
    ("2015 Dec", (2015, 12)),
    ("1998 Spring", (1998, 4)),
    ("1976 Nov-Dec", (1976, 11)),
    ("2009", (2009, None)),
    ("1975 Jan 1", (1975, 1)),
    ("", (None, None)),
    ("   ", (None, None)),
    ("no date here", (None, None)),
])
def test_date_string_parsing(value, expected):
    assert fp._parse_date_string(value) == expected


def test_year_out_of_plausible_range_is_not_taken_as_year():
    assert fp._parse_date_string("3500 Apr") == (None, 4)


def test_prefers_sortpubdate_then_pubdate_then_epubdate():
    rec = {"sortpubdate": "2002/04/15 00:00", "pubdate": "2003 Jan", "epubdate": "2001 Dec"}
    out = fp.parse_record(rec)
    assert (out["year"], out["month"], out["date_source"]) == (2002, 4, "sortpubdate")

    out = fp.parse_record({"sortpubdate": "", "pubdate": "2003 Jan", "epubdate": "2001 Dec"})
    assert (out["year"], out["date_source"]) == (2003, "pubdate")

    out = fp.parse_record({"sortpubdate": "", "pubdate": "", "epubdate": "2001 Dec"})
    assert (out["year"], out["date_source"]) == (2001, "epubdate")


def test_raw_fields_are_always_preserved():
    rec = {"sortpubdate": "2002/04/15 00:00", "pubdate": "2002 Apr 15",
           "epubdate": "2002 Mar 1", "source": "J Cell Sci"}
    out = fp.parse_record(rec)
    assert out["raw_pubdate"] == "2002 Apr 15"
    assert out["raw_epubdate"] == "2002 Mar 1"
    assert out["journal"] == "J Cell Sci"


def test_error_record_yields_no_year_but_keeps_message():
    out = fp.parse_record({"uid": "999", "error": "cannot get document summary"})
    assert out["year"] is None
    assert "cannot get" in out["error"]


# --------------------------------------------------------------------------- #
# Collecting identifiers
# --------------------------------------------------------------------------- #
def _write_subunit_csv(tmp_path, pmid_values):
    """Write a minimal subunit edge file. Values are quoted, as a comma inside
    an unquoted field would be read as a column break rather than a separator."""
    import csv as _csv
    p = tmp_path / fp.SUBUNIT_FILE
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh, quoting=_csv.QUOTE_MINIMAL)
        w.writerow(["id", ":TYPE", ":START_ID", ":END_ID",
                    "corum_subunit_uniprot_id:string", "pmid:string"])
        for i, v in enumerate(pmid_values):
            w.writerow([f"u{i}", "has_subunit", f"CORUM:{i}", f"P{i}", f"Q{i}", v])
    return tmp_path


def test_collect_handles_multiple_separators_and_junk(tmp_path):
    # The live export carries exactly one numeric id per row; this covers the
    # defensive paths in case a future export packs several into one field.
    d = _write_subunit_csv(tmp_path, ["111", "222;333", "444,555", "666 777", "", "abc", "None"])
    pmids, stats = fp.collect_pmids(d)
    assert pmids == ["111", "222", "333", "444", "555", "666", "777"]
    assert stats["non_numeric"] == 1
    assert stats["blank"] == 2
    assert stats["distinct_pmids"] == 7


def test_collect_sorts_numerically_not_lexically(tmp_path):
    d = _write_subunit_csv(tmp_path, ["100", "99", "1000"])
    pmids, _ = fp.collect_pmids(d)
    assert pmids == ["99", "100", "1000"]


def test_collect_errors_clearly_on_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="could not find"):
        fp.collect_pmids(tmp_path)


def test_collect_errors_clearly_on_missing_column(tmp_path):
    p = tmp_path / fp.SUBUNIT_FILE
    p.write_text("a,b\n1,2\n")
    with pytest.raises(SystemExit, match="not in"):
        fp.collect_pmids(tmp_path)


# --------------------------------------------------------------------------- #
# Fetching, with the network stubbed out
# --------------------------------------------------------------------------- #
def _stub(responses, calls):
    def _req(ids, api_key, email, timeout):
        calls.append(list(ids))
        return {"result": {i: responses.get(i, {"uid": i, "error": "missing"})
                           for i in ids}}
    return _req


def test_fetch_batches_and_caches(tmp_path, monkeypatch):
    pmids = [str(i) for i in range(1, 11)]
    responses = {p: {"uid": p, "sortpubdate": f"20{10 + int(p)}/01/01 00:00"} for p in pmids}
    calls = []
    monkeypatch.setattr(fp, "_request", _stub(responses, calls))
    cache, cache_path = {}, tmp_path / "cache.json"

    report = fp.fetch(pmids, cache, None, None, batch_size=4, sleep=0,
                      max_retries=2, timeout=5, cache_path=cache_path, verbose=False)
    assert [len(c) for c in calls] == [4, 4, 2]
    assert report["fetched"] == 10
    assert cache["3"]["year"] == 2013
    assert json.loads(cache_path.read_text())["3"]["year"] == 2013


def test_fetch_skips_already_cached(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fp, "_request", _stub({}, calls))
    cache = {"1": {"year": 1999}}
    report = fp.fetch(["1"], cache, None, None, 10, 0, 2, 5,
                      tmp_path / "c.json", verbose=False)
    assert calls == []
    assert report["already_cached"] == 1


def test_fetch_retries_then_leaves_batch_unfetched(tmp_path, monkeypatch):
    attempts = []

    def always_fail(ids, api_key, email, timeout):
        attempts.append(1)
        raise ConnectionError("boom")

    monkeypatch.setattr(fp, "_request", always_fail)
    cache = {}
    report = fp.fetch(["1", "2"], cache, None, None, 10, 0, 3, 5,
                      tmp_path / "c.json", verbose=False)
    assert len(attempts) == 3
    assert report["failed_batches"] == 1
    # nothing cached, so a rerun retries exactly these
    assert cache == {}


def test_fetch_records_ids_the_service_omits(tmp_path, monkeypatch):
    monkeypatch.setattr(fp, "_request",
                        lambda ids, a, e, t: {"result": {}})
    cache = {}
    fp.fetch(["42"], cache, None, None, 10, 0, 2, 5, tmp_path / "c.json", verbose=False)
    assert cache["42"]["year"] is None
    assert "not returned" in cache["42"]["error"]


# --------------------------------------------------------------------------- #
# Quality checks
# --------------------------------------------------------------------------- #
def test_quality_checks_coverage_and_decades():
    cache = {"1": {"year": 1999, "date_source": "sortpubdate"},
             "2": {"year": 2005, "date_source": "sortpubdate"},
             "3": {"year": None, "error": "gone"}}
    c = fp.quality_checks(cache, ["1", "2", "3"])
    assert c["resolved"] == 2 and c["unresolved"] == 1
    assert c["coverage"] == pytest.approx(2 / 3, abs=1e-3)
    assert c["by_decade"] == {"1990s": 1, "2000s": 1}
    assert c["error_messages"] == {"gone": 1}


def test_quality_checks_detect_pmid_order_inversions():
    # identifiers ascend but years do not: a retrospectively indexed article
    cache = {"10": {"year": 2000}, "20": {"year": 2001}, "30": {"year": 1953}}
    c = fp.quality_checks(cache, ["10", "20", "30"])
    assert c["pmid_order_inversions"] == 1
    assert c["pmid_order_inversion_rate"] == pytest.approx(0.5)


def test_quality_checks_flag_implausible_years():
    cache = {"1": {"year": 1800}, "2": {"year": 2005}}
    c = fp.quality_checks(cache, ["1", "2"])
    assert c["implausible_years"] == 1


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def test_csv_has_one_row_per_requested_id(tmp_path):
    import csv as _csv
    cache = {"1": {"year": 2001, "month": 3, "date_source": "pubdate",
                   "raw_pubdate": "2001 Mar", "journal": "J"}}
    out = tmp_path / "pmid_dates.csv"
    fp.write_csv(cache, ["1", "2"], out)
    rows = list(_csv.DictReader(open(out)))
    assert [r["pmid"] for r in rows] == ["1", "2"]
    assert rows[0]["year"] == "2001" and rows[0]["month"] == "3"
    assert rows[1]["year"] == ""           # unresolved still appears
    assert list(rows[0].keys()) == fp.FIELDS
