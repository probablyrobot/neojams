NeoJAMS: A JSON Annotated Music Specification
==================================

[![PyPI](https://img.shields.io/pypi/v/neojams.svg)](https://pypi.python.org/pypi/neojams)
[![License](https://img.shields.io/github/license/probablyrobot/neojams.svg)](https://github.com/probablyrobot/neojams/blob/main/LICENSE)
[![Tests](https://github.com/probablyrobot/neojams/actions/workflows/test.yml/badge.svg)](https://github.com/probablyrobot/neojams/actions/workflows/test.yml)
[![PyPI Deployment](https://github.com/probablyrobot/neojams/actions/workflows/publish.yml/badge.svg)](https://github.com/probablyrobot/neojams/actions/workflows/publish.yml)
[![Python Versions](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://www.python.org/)

A modernized Python package for working with the JAMS format, built on top of the original JAMS project. This version adds support for Python 3.12 and 3.13, improved type safety, and enhanced robustness.

JAMS are structured JSON annotations for music tracks that fully contain both the information and metadata for various types of annotations.

## Installation

You can install the most recent release of NeoJAMS from PyPI:

```bash
pip install neojams
```

Or with Poetry:

```bash
poetry add neojams
```

For development installation:

```bash
# Clone the repository
git clone https://github.com/probablyrobot/neojams.git
cd neojams

# Install with Poetry
poetry install --with dev
```

**NeoJAMS requires Python 3.12 or later.**

Please, refer to [documentation](http://jams.readthedocs.io/en/stable/) for a comprehensive
description of JAMS.

What
----
JAMS is a JSON-based music annotation format.

We provide:
* A formal JSON schema for generic annotations
* The ability to store multiple annotations per file
* Schema definitions for a wide range of annotation types (beats, chords, segments, tags, etc.)
* Error detection and validation for annotations
* A translation layer to interface with [mir eval](https://craffel.github.io/mir_eval)
    for evaluating annotations

Why
----
Music annotations are traditionally provided as plain-text files employing
simple formatting schema (comma or tab separated) when possible. However, as
the field of MIR has continued to evolve, such annotations have become
increasingly complex, and more often custom conventions are employed to
represent this information. And, as a result, these custom conventions can be
unwieldy and non-trivial to parse and use.

Therefore, JAMS provides a simple, structured, and sustainable approach to
representing rich information in a human-readable, language agnostic format.
Importantly, JAMS supports the following use-cases:
* multiple types annotations
* multiple annotations for a given task
* rich file level and annotation level metadata

How
----
This library is offered as a proof-of-concept, demonstrating the promise of a
JSON-based schema to meet the needs of the MIR community. To install, clone the
repository into a working directory and proceed thusly.

The full documentation can be found [here](http://jams.readthedocs.io/en/stable/).

Development
-----------
NeoJAMS is specifically designed for Python 3.12 and 3.13, with modern type hints and improved error handling.

For development setup and guidelines, see [DEVELOPMENT.md](DEVELOPMENT.md).

Who
----
NeoJAMS is a modernization of the original JAMS project, which was developed by the MARL@NYU team and contributors. This version is maintained by Igor Bogicevic (igor.bogicevic@gmail.com, GitHub: probablyrobot).

The original JAMS effort evolved out of internal needs at MARL@NYU, with feedback from LabROSA. This modernization builds upon their work while adding support for modern Python features and improved type safety.

If you want to get involved, do let us know!

Details
-------
JAMS is proposed in the following publication:

[1] Eric J. Humphrey, Justin Salamon, Oriol Nieto, Jon Forsyth, Rachel M. Bittner,
and Juan P. Bello, "[JAMS: A JSON Annotated Music Specification for Reproducible
MIR Research](http://marl.smusic.nyu.edu/papers/humphrey_jams_ismir2014.pdf)",
Proceedings of the 15th International Conference on Music Information Retrieval,
2014.

The JAMS schema and data representation used in the API were overhauled significantly between versions 0.1 (initial proposal) and 0.2 (overhauled), see the following technical report for details:

[2] B. McFee, E. J. Humphrey, O. Nieto, J. Salamon, R. Bittner, J. Forsyth, J. P. Bello, "[Pump Up The JAMS: V0.2 And Beyond](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/mcfee_jams_ismir_lbd2015.pdf)", Technical report, October 2015.
