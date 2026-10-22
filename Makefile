.PHONY: help install data-check profile fidelity study hypergraph figures test lint clean all

PY ?= python3
DATA ?= data
OUT ?= results

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## editable install with dev extras
	$(PY) -m pip install -e ".[dev]"

data-check:  ## report which input files are present
	@$(PY) -c "from pathlib import Path; \
	req=['protein-complex_protein_association_neo4j.csv',\
	'functional-complex-group_protein-complex_association_neo4j.csv',\
	'functional-complex-group_node_neo4j.csv','protein-complex_node_neo4j.csv']; \
	opt=['protein_protein_association_neo4j.csv']; \
	d=Path('$(DATA)'); \
	miss=[f for f in req if not (d/f).exists()]; \
	print('required : ' + ('MISSING ' + str(miss) if miss else 'all present')); \
	print('optional : ' + ('present' if (d/opt[0]).exists() else 'absent (interaction-coverage check will be skipped)')); \
	print('dates    : ' + ('present' if (d/'cache'/'pmid_dates.csv').exists() else 'absent (ordinal proxy will be used)')); \
	raise SystemExit(1 if miss else 0)"

profile:  ## Layer 0 only: descriptors and predicted verdict
	$(PY) -m nary_lab.cli profile --data-dir $(DATA) --conformality

fidelity:  ## Layer 1 round-trip table
	$(PY) -m nary_lab.cli fidelity --data-dir $(DATA)

pubmed:  ## fetch publication dates for CORUM PMIDs (needs network; run locally)
	$(PY) scripts/fetch_pubmed_dates.py --data-dir $(DATA) --out-dir data/cache

pubmed-dry:  ## show what pubmed would fetch, without making any request
	$(PY) scripts/fetch_pubmed_dates.py --data-dir $(DATA) --dry-run

study:  ## full four-layer study -> results/
	$(PY) -m nary_lab.cli run --data-dir $(DATA) --out-dir $(OUT)

hypergraph:  ## native hypergraph analyses incl. higher-order motifs
	$(PY) experiments/hypergraphx_analysis.py --data-dir $(DATA) --out-dir $(OUT)

figures:  ## regenerate figures from results/
	$(PY) experiments/make_figures.py --results $(OUT)/study_results.json

test:  ## run the test suite
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src experiments tests

all: data-check study hypergraph figures  ## everything, in order

clean:
	rm -rf $(OUT)/*.json $(OUT)/*.txt $(OUT)/figures/*.png
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
