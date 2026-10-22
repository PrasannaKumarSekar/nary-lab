# Input data

## Files

Put these in `data/`. They are Neo4j bulk-import CSVs exported from the knowledge graph.

| file | size | needed for |
|---|---|---|
| `protein-complex_protein_association_neo4j.csv` | 1.5 MB | **required** — the groups themselves, one row per (complex, subunit) |
| `functional-complex-group_protein-complex_association_neo4j.csv` | 0.2 MB | **required** — curated labels used to score clustering |
| `functional-complex-group_node_neo4j.csv` | 4 KB | label names |
| `protein-complex_node_neo4j.csv` | 0.5 MB | complex names and organism |
| `protein_protein_association_neo4j.csv` | 714 MB | optional — the independent interaction-coverage check |

The first four ship with this repository. The interaction file does not, because of its size; without it the study runs and simply skips that one check.

```bash
make data-check    # verifies what is present
```

## Identifiers

The subunit file carries two identifiers per subunit: `:END_ID` (the protein node in the graph) and `corum_subunit_uniprot_id:string` (the source database's own accession).

The default is `:END_ID`, so measurements describe the graph as it actually is. The mapping is not injective — 5,232 source accessions resolve onto 5,107 graph nodes — and the loader reports that collapse on every run. Pass `--member-column corum_subunit_uniprot_id:string` to use source accessions instead, which is the right choice when comparing against published analyses of the source database.

## Filters

Two, both reported in every run:

- **Single-member entries dropped** (307). A one-protein "complex" is a curation placeholder, not a group.
- **Groups above 25 members dropped** (25). The largest has 129. Enumerating over subsets of very large groups dominates runtime. Change with `--max-size`.

Leaves 5,222 groups over 4,777 members.

## Publication dates

Groups cite the paper describing them, but the export carries no dates. `data/cache/pmid_dates.csv` holds the resolved years and is committed, so the study is reproducible from a file rather than from a live service.

To regenerate it:

```bash
python scripts/fetch_pubmed_dates.py --data-dir data --out-dir data/cache
# or: make pubmed      (make pubmed-dry to preview without network calls)
```

The script is standalone — Python 3.9+ and the standard library only, nothing from this package — so it can be copied anywhere and run. It reads the PubMed IDs out of the subunit file, queries NCBI E-utilities in batches of 200, and writes:

| file | purpose |
|---|---|
| `pmid_dates.csv` | one row per PubMed ID — the artefact to keep |
| `fetch_report.json` | coverage, failures, timing, quality checks |
| `pmid_cache.json` | resume cache; not needed downstream |

3,656 identifiers means 19 requests, about a minute. Set `NCBI_API_KEY` to raise the rate limit from 3 to 10 per second. Safe to interrupt; completed batches are cached.

### What it stores and why

Per identifier: resolved `year` and `month`, which of the three PubMed date fields it came from, and all three raw date strings. PubMed dates are irregular (`2002 Apr 15`, `1998 Spring`, `1976 Nov-Dec`, bare `2009`) and the fields can disagree, so keeping the raw strings means the choice of field can be revisited without refetching. Preference order is `sortpubdate`, then `pubdate`, then `epubdate`.

### Checks it runs

- **Coverage** — how many resolved, with the error message for each failure.
- **Plausibility** — years outside 1950–2030.
- **Ordering** — how often PubMed ID order disagrees with publication order. This audits the fallback below.

### Fallback

Without `pmid_dates.csv`, the framework ranks PubMed identifiers numerically and uses the rank as a stand-in for chronology. Identifiers are issued in broadly increasing order, but older papers were indexed retrospectively, so it is approximate.

Real years and the proxy are held in separate fields (`timestamp` vs `order_key`), and `describe_basis()` names which was used, so an approximation cannot be reported as a measurement. Drop the CSV into `data/cache/` and everything switches over with no flag to set.

## Provenance

Complexes come from CORUM, a manually curated catalogue of mammalian protein complexes — every entry traces to a published experiment, and high-throughput screens are excluded. The interaction layer is assembled from STRING, IntAct, Reactome and SIGNOR, and is separately sourced evidence rather than a flattened view of the complexes.
