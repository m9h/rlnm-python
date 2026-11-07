# rlnm-python

A Python API and CLI for robust lesion network mapping: functional projection,
spatial-null sensitivity, and selective specificity. This is an independent
research implementation of [pBFSLab/rLNM](https://github.com/pBFSLab/rLNM), pinned
to commit `059ade5fc2a332bb2a246ef7db0bd7c4217446e4`. It is not an official release
from the paper's authors.

## Install and run

```bash
git clone https://github.com/m9h/rlnm-python.git
cd rlnm-python
uv sync --locked
uv run rlnm demo --output outputs/demo
uv run rlnm run outputs/demo/specificity.json --output outputs/demo-result
```

Open `outputs/demo-result/report.html` for a self-contained QC report. The CLI
writes NIfTI statistics, intermediate arrays, the configuration, and SHA-256
provenance. It refuses to overwrite a nonempty output directory.

See [methods and limitations](docs/methods.md), [input formats](docs/inputs.md),
and [validation](docs/validation.md). Python 3.11+ is supported. No MATLAB license
is needed to run the Python package.

## Status

Version 0.1 implements the statistical and functional-mapping core. Structural
tractography, electrode reconstruction, electric-field simulation, registration,
clinical prediction, and a full interactive viewer are outside this release.
Precomputed connectivity maps can be imported; their modality and interpretation
must be appropriate to the specified analysis. A disconnectome probability is not
automatically interchangeable with a signed functional-connectivity map.

The pinned upstream's fixed-residual permutation method and selected-sign
sensitivity tail are explicitly identified in every analysis. Numerical agreement
does not establish that these inferential choices are calibrated for every study.

## Development

```bash
uv sync --locked
uv run pytest
uv run ruff check .
uv run python -m build
```

Apache-2.0. See [NOTICE](NOTICE) for attribution and adapted upstream components.

