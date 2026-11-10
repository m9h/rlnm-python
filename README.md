# rlnm-python

A Python API and CLI for robust lesion network mapping: functional projection,
spatial-null sensitivity, and selective specificity. This is an independent
research implementation of [pBFSLab/rLNM](https://github.com/pBFSLab/rLNM), pinned
to commit `059ade5fc2a332bb2a246ef7db0bd7c4217446e4`. It is not an official release
from the paper's authors.

[![tests](https://github.com/m9h/rlnm-python/actions/workflows/ci.yml/badge.svg)](https://github.com/m9h/rlnm-python/actions/workflows/ci.yml)

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

The [research and software comparison](docs/software-landscape.md) connects
disconnectome analysis, Horn's Lead-DBS ecosystem, Python alternatives, and EigenBrain,
with sources and priorities for further development.

The [ARC lab validation plan](docs/arc-lab-plan.md) specifies DGX Spark execution,
NAS storage, patient-outcome benchmarks, and a proposed longitudinal tractography
extension. These ARC workflows are planned and have not yet run.

## Status

Initial validation passed: 20 tests on Linux, macOS and Windows (Python 3.11/3.12),
plus execution of original MATLAB calculation blocks through Octave. On the
184-map upstream example, shared-permutation p-values matched exactly and the
largest numerical discrepancy across compared arrays was `1.42e-14`.
The Python sample pipeline also completed 10,000 permutations over 5,486 voxels.
[Recorded evidence and scope](docs/validation.md#recorded-initial-validation).

Version 0.2 adds optional molecular annotation and an experimental, nested
MRI/PET prediction benchmark. The original statistical calculations are unchanged.
Structural
tractography, electrode reconstruction, electric-field simulation, registration,
validated clinical prediction, and a full interactive viewer are outside this release.
Precomputed connectivity maps can be imported; their modality and interpretation
must be appropriate to the specified analysis. A disconnectome probability is not
automatically interchangeable with a signed functional-connectivity map.

The pinned upstream's fixed-residual permutation method and selected-sign
sensitivity tail are explicitly identified in every analysis. Numerical agreement
does not establish that these inferential choices are calibrated for every study.

## Molecular annotations and EigenBrain-style benchmarking

```bash
uv sync --locked --all-extras
uv run --all-extras rlnm demo-research --output outputs/research
uv run --all-extras rlnm annotate outputs/research/annotate.json --output outputs/annotation
uv run --all-extras rlnm benchmark-eigenbrain outputs/research/benchmark.json --output outputs/benchmark
```

These examples are synthetic. Each workflow writes a standalone `report.html`,
machine-readable results, and input/output hashes. Nothing downloads during the
synthetic demos. An explicit neuromaps query in an annotation config downloads
that reference map into the output directory's cache.

- **`annotate`:** continuous NIfTI map → common-domain parcel averages →
  annotation-side BrainSMASH spatial surrogates → two-sided correlations and BH
  correction across annotations. Includes coverage and variogram diagnostics.
- **`benchmark-eigenbrain`:** paired individual FC/PET arrays plus an independent
  reference FC matrix → BrainSpace gradients → nested leave-one-subject-out
  molecular reconstruction, compared with a training-only population atlas.

See [configuration, methods, and limitations](docs/molecular.md). This is an
independent FC-only experiment inspired by [EigenBrain](https://doi.org/10.64898/2026.10.27.758642),
not a reproduction of the complete paper or evidence of individual receptor density.
Map-comparison nulls complement the original lesion-location and symptom-label
nulls; they test different hypotheses. Optional dependencies and fetched atlases
retain their own licenses and attribution requirements.

## Development

```bash
uv sync --locked
uv run pytest
uv run ruff check .
uv run python -m build
```

Apache-2.0. See [NOTICE](NOTICE) for attribution and adapted upstream components.
