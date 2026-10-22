"""nary_lab - a framework for comparing ways of storing higher-order facts in a graph.

A *higher-order fact* is one that involves more than two things at once: a
protein complex made of five subunits, a biochemical reaction with three
substrates and two products, a combination therapy of four drugs. A graph edge
connects exactly two things, so such facts do not fit directly.

This package builds three representations of the same facts and measures them
against each other across four layers:

* ``descriptors``    - Layer 0, cheap structural statistics and a predicted verdict
* ``fidelity``       - Layer 1, what each representation loses on a round trip
* ``queries`` / ``analytics`` - Layer 2, what can be asked and what can be computed
* ``implementation`` - Layer 3, footprint, traversal cost, tooling

Entry point: ``nary_lab.study.run_study`` or the ``nary_lab`` command.
"""

__version__ = "1.0.0"

from .model import Group, GroupCollection  # noqa: E402
from .representations import CAPABILITIES  # noqa: E402

__all__ = ["__version__", "Group", "GroupCollection", "CAPABILITIES"]
