# nary-lab

*An **n-ary** relationship is one that connects more than two things. This is a lab for
measuring how best to store them in a graph.*

A small framework for answering one question: **when a fact involves more than two things, how should it be stored in a graph — and what does each choice cost?**

Graph edges join exactly two nodes. Plenty of facts don't fit that shape:

- a protein complex made of six subunits
- a biochemical reaction with three substrates, two products and an enzyme
- a combination therapy of four drugs for one indication
- a piece of evidence linking a gene, a disease, a study and a cohort

`nary-lab` calls any such fact a **group**. It builds three different representations of the same groups and measures them against each other, so the modelling decision can be made from numbers rather than from intuition.

---

## The three representations

| key | name | how one group is stored |
|---|---|---|
| `dissolved` | pairwise graph | one edge per pair of members; the group itself is not stored |
| `reified` | hub node | a node representing the group, with an edge to each member |
| `native` | hyperedge | a single object whose members are all of them |

The first differs from the other two in **what is stored**. The second and third differ only in **how it is stored and which tools can act on it** — the framework verifies that rather than assuming it.

---

## What it measures

Four layers. Each answers a different question, and each can be run on its own.

**Layer 0 — Describe.** Six cheap structural statistics, plus a *predicted* verdict. Runs in seconds and tells you whether the expensive layers are worth it for a given dataset.

| descriptor | what it tells you |
|---|---|
| mean group size | below about 3 there is little for grouping to preserve |
| share above pair size | how much of the data is in scope at all |
| hub concentration | whether dissolving will mainly **invent** groups or **lose** them |
| recurrence | whether set-level methods have any pattern to learn |
| mean overlap | how ambiguous a pairwise decomposition would be |
| conformality | what share of groups survive dissolution exactly |

**Layer 1 — Fidelity.** Build each representation, read the groups back out, compare. Two error types are reported separately because they have different consequences: groups **invented** (the reconstruction produces something never asserted) and groups **vanished** (a real group is not recovered). Always broken down by group size.

**Layer 2 — Capability.** What each representation can be *asked* (five query types; anything inexpressible scores zero and is reported rather than skipped) and what can be *computed* with it (clustering against known labels; chronological prediction of new groups, with significance testing against an explicit null).

**Layer 3 — Trade-offs.** Storage footprint, traversal cost, which algorithms are available natively versus only after projection, and a log of practical tooling friction.

---

## Why the prediction sits next to the measurement

Layer 0 emits a *predicted* verdict before any experiment runs; Layers 1–3 emit a *measured* one. Both appear on the scorecard.

When they disagree, that disagreement is the finding, and the prediction thresholds should change. That feedback loop is what makes this a tool that improves across studies rather than a script that accumulates results.

---

## Design decisions worth knowing

**One canonical input.** Everything derives from a *group-incidence table* — one row per (group, member) pair:

```
group_id   group_type   role     member_id   evidence   timestamp
```

All three representations are pure functions of that one structure, so no measured difference can be an artefact of loading them from different places. It is deliberately the same shape as a Neo4j relationship-import CSV with the group as `:START_ID`.

**Capability claims are tested, not documented.** Each representation declares what it preserves in `representations.CAPABILITIES`; a test checks those declarations against measured behaviour, so a claim that drifts from reality fails rather than quietly misleading a reader.

**Approximations are structurally separated from measurements.** Real timestamps and ordinal stand-ins live in different fields, and every report names which was used.

**Decoder strength matters.** Reconstruction from a pairwise graph uses maximal-clique enumeration — the strongest decoder available — so the measurement reflects the encoding rather than a weak reader.

**Rare positives get a null distribution, not a lift number.** When a prediction task has few positive cases, a random ranking can still score two or three times the base rate. Scores are therefore compared against an explicitly generated null and reported as beating chance or not.

---

## Layout

```
src/nary_lab/
  model.py            the canonical group-incidence structure
  io.py               loaders (CSV -> GroupCollection)
  representations.py  the three representations + capability declarations
  descriptors.py      Layer 0
  fidelity.py         Layer 1
  queries.py          Layer 2a
  analytics.py        Layer 2b
  implementation.py   Layer 3
  pubmed.py           publication-date handling
  scorecard.py        rendering
  study.py            orchestrator
  cli.py              command line

scripts/              standalone helpers (no install needed)
experiments/          optional extras beyond the core run
data/                 inputs (see docs/DATA.md)
results/              generated outputs
tests/                including the capability-claim checks
```

---

## Using it

```bash
pip install -e ".[dev]"
make data-check     # confirm inputs are in place
make study          # all four layers -> results/
make figures
```

Or step by step:

```bash
nary-lab profile  --data-dir data   # Layer 0 only
nary-lab fidelity --data-dir data   # Layer 1 table
nary-lab run      --data-dir data   # everything
```

Outputs land in `results/`: a one-page `scorecard.txt`, full numbers in `study_results.json`, and figures. Every run writes a manifest recording configuration, library versions and timing.

Optional extras:

```bash
python experiments/hypergraphx_analysis.py --data-dir data   # hypergraph-native operations
python scripts/fetch_pubmed_dates.py --data-dir data --out-dir data/cache
```

---

## Pointing it at your own data

Write one loader that returns a `GroupCollection`. Nothing else changes.

```python
from nary_lab.model import Group, GroupCollection

groups = [
    Group(id="RXN:1",
          members=frozenset({"glucose", "ATP", "G6P", "ADP"}),
          group_type="Reaction",
          roles=(("glucose", "substrate"), ("ATP", "substrate"),
                 ("G6P", "product"), ("ADP", "product")),
          evidence=("12345678",),
          timestamp=2004.0),
]
collection = GroupCollection(groups, "reactions")
```

Then run any layer against it. `io.load_complexes` is a worked example, including how to report what a loader dropped.

**Adding a representation:** write `build_x` and `recover_from_x`, add a `Capability` entry declaring what it preserves, register it, and add a branch in `fidelity.assess`. The capability test will police it from then on.

---

## Current state

A first iteration, built and validated on a single dataset. Known gaps:

- **Roles are supported but untested.** The dataset used so far has flat member sets, so the role machinery has had no exercise. Phenomena with distinguishable roles and direction — reactions, signalling — are the obvious next step.
- **Nested groups are not modelled**; a group inside a larger group is treated as unrelated.
- **Traversal cost is derived from the encoding, not benchmarked** against a live database.
- **No learned models.** Scorers are simple local statistics, chosen so that differences are attributable to the representation rather than to model capacity.

See `docs/DATA.md` for input setup and provenance.
