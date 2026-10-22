# data/

Inputs for the study. See `../docs/DATA.md` for provenance, identifier policy and filters.

Committed here:

| file | purpose |
|---|---|
| `protein-complex_protein_association_neo4j.csv` | the groups (complex -> subunit) |
| `functional-complex-group_protein-complex_association_neo4j.csv` | curated labels for scoring |
| `functional-complex-group_node_neo4j.csv` | label names |
| `protein-complex_node_neo4j.csv` | complex metadata |
| `cache/pmid_dates.csv` | resolved publication years |

Not committed, because of size (714 MB):

| file | effect if absent |
|---|---|
| `protein_protein_association_neo4j.csv` | the interaction-coverage check is skipped; everything else runs |

Run `make data-check` to see what is present.
