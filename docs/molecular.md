# Molecular annotation and experimental EigenBrain-style benchmarking

Version 0.2 adds two independent workflows. Neither changes rLNM sensitivity or
specificity calculations. Install extras with `uv sync --locked --all-extras`, or
install `rlnm-python[annotations,eigenbrain]` from a checkout with pip. Use
`uv run --all-extras ...` thereafter so uv keeps optional packages installed.
BrainSMASH 0.11 uses NumPy's deprecated `row_stack`; the annotations extra constrains
NumPy to `<2.5`. The core does not require BrainSMASH, neuromaps, BrainSpace, or VTK.

## Annotation inputs

Run `rlnm annotate config.json --output results`. Paths are relative to the config.

```json
{
  "map": "symptom-001_effect.nii.gz",
  "mask": "valid_target_domain.nii.gz",
  "atlas": "cortical_parcels.nii.gz",
  "distances": "parcel_distances.npz",
  "space": "MNI152",
  "min_coverage": 0.8,
  "n_nulls": 10000,
  "seed": 42,
  "null_model": {"kernel": "exp", "pv": 25, "nh": 25},
  "annotations": [
    {
      "id": "5HT1a-savli",
      "neuromaps": {
        "source": "savli2012", "desc": "way100635",
        "space": "MNI152", "res": "3mm"
      },
      "citation": "https://doi.org/10.1016/j.neuroimage.2012.07.001",
      "resample": true,
      "zero_is_missing": true
    }
  ]
}
```

An annotation can instead specify `"map": "local_annotation.nii.gz"`; supply an
`id` and the original data `citation` in either case. Local paths and neuromaps
queries are mutually exclusive. IDs must be unique. A query must identify one
MNI152 volume exactly. Surface annotations are not automatically transformed;
prepare them explicitly outside this workflow with the appropriate method.

The target should be a continuous, signed effect/connectivity map. Use the actual
analysis-domain mask: values outside it are not observations. A thresholded
significance map changes the question and creates selection artifacts. A cortical
receptor atlas cannot directly annotate white-matter disconnection probabilities
or electrode locations without an appropriate cortical projection first.

### Geometry and missing values

- Target, binary mask, and nonnegative integer-label atlas must share the same
  grid and declared millimetre units. Atlas label zero is background.
- Local annotations must share that grid unless `resample: true` is explicitly
  enabled. Resampling is linear within the declared template; it is not anatomical
  registration. The template string cannot distinguish or verify MNI variants.
- Neuromaps volumes undergo the same geometry checks. Fetching does not authorize
  implicit registration or resampling. Use only maps registered to the intended
  template variant.
- NaN/Inf values are missing. `zero_is_missing` defaults to false: zero is a valid
  observation unless explicitly declared background. When enabled, zero is made
  missing **before** interpolation. Interpolation outside the field of view
  produces NaN, never artificial zero-valued observations.
- Every annotation and the target use the intersection of finite voxels within
  the mask. Each parcel's coverage is measured against its full atlas volume.
  Parcels below `min_coverage` are excluded from every comparison. Thus changing
  the annotation set can change the common domain and all correlations.
- Parcel means have equal weight in the correlation; this is not a voxelwise or
  surface-area-weighted analysis. Report the atlas and coverage threshold.

### Distance archive and spatial hypothesis

`parcel_distances.npz` contains exactly:

| Array | Shape / meaning |
|---|---|
| `parcel_ids` | `(P,)` unique positive integer atlas labels, in array order |
| `distances` | `(P,P)` pairwise distances appropriate to the intended null |
| `blocks` | `(P,)` integer or string labels, usually hemisphere |

Parcel IDs must exactly cover the nonzero atlas labels; ordering is explicit and
need not be sorted. Within each block, distances must be finite, symmetric, zero
on the diagonal, and positive elsewhere. Cross-block distances are unused and may
be NaN. After coverage filtering each remaining block must have at least 20 parcels.
Provide cortical geodesic distances for cortical variogram nulls. Arbitrary label
numbers or Euclidean distances are not substitutes for those distances. The
synthetic example uses Euclidean distances on a synthetic grid only.

BrainSMASH fits the annotation's variogram within each block. Blocks have separate,
deterministic seeds. Its demeaned surrogates have the observed block mean restored,
so the null retains between-block mean differences. Cross-block covariance and
bilateral homology are **not**
preserved. This is a stated limitation of the selected null hypothesis. The same
draw number combines one surrogate from each block. Each generated annotation is
correlated with the unchanged target. The two-sided Monte Carlo p-value counts
`abs(r_null) >= abs(r_observed)`, includes ties, and adds one to numerator and
denominator. The minimum possible p is `1/(n_nulls+1)`. Constant maps/nulls fail.
BH correction spans **all requested annotations for this one target**; analyses
of multiple target maps need a separately defined multiple-testing family.

The report includes empirical and surrogate variograms with a 95% envelope and
relative RMSE from at most 100 evenly selected draws. These are fit diagnostics,
not an automatic calibration certificate or a universal acceptance threshold.
Inspect them before interpreting p-values. Run-specific draws depend on the
annotation order, seed, null parameters and dependency versions, all recorded.
The deliberately small synthetic grid can have poor variogram fits; its p-values
are pipeline demonstrations, not examples of established calibration. Each result
includes its worst block-relative RMSE to keep this diagnostic beside the scores.

Outputs: `associations.tsv`, `parcels.tsv` (including excluded parcels),
`parcel_data.npz`, `null_correlations.npz`, `null_diagnostics.json`, figures,
`summary.json`, `provenance.json`, and `report.html`. Fetched source files stay in
`cache/` with hashes. Surrogate maps themselves are not retained; null correlations,
fit diagnostics, settings, and dependency versions are retained for regeneration.

## Nested EigenBrain-style benchmark

Run `rlnm benchmark-eigenbrain config.json --output results`:

```json
{
  "dataset": "paired_mri_pet.npz",
  "reference": "independent_reference_fc.npz",
  "space": "Glasser360-fsLR32k",
  "marker": "tau",
  "reference_citation": "Study/accession and reference cohort definition",
  "components": [5, 10, 20, 40],
  "sparsity": 0.9,
  "seed": 42
}
```

The paired NPZ must contain exactly `subject_ids` (unique strings), `parcel_ids`
(unique positive integers), `fc` `(N,P,P)`, and `pet` `(N,P)`. At least four subjects
are required. One marker is analyzed per run. All values must be finite; use a
prespecified common parcel domain. PET is standardized within each participant
across parcels, so the outcome is spatial shape, not total tracer uptake.

The reference NPZ contains exactly `subject_ids` (the contributors to the
reference), `parcel_ids`, and a single group `fc` `(P,P)` matrix. IDs must be
disjoint from the paired cohort, with exactly matching parcel IDs and order. Use
namespaced participant identifiers across cohorts. Independence remains a user
declaration: the software cannot prove who contributed to a matrix. Never construct
the reference using the evaluated cohort. All FC matrices must use the same
processing and correlation/Fisher-z convention and must be finite and symmetric.

The implementation uses BrainSpace normalized-angle affinity, diffusion-map
embedding (`alpha=0.5`, `diffusion_time=0`), explicit row sparsity, and zeroed FC
diagonals. A disconnected embedding graph fails. For each candidate count K,
individual modes are aligned to the external reference with orthogonal Procrustes
and then standardized per column. Alignment is computed using MRI only, separately
for each K. It does not use the molecular outcome.

For each outer test participant:

1. Reserve that participant's PET map for evaluation only.
2. Among the remaining participants, use leave-one-out inner validation to choose
   K by mean prediction MSE. Each inner training target is the mean PET map of
   **inner-training participants only**. Ties select the smaller K.
3. Fit OLS with an intercept from reference gradients to the outer-training mean
   PET map. Transfer its weights to the outer-test participant's aligned gradients.
4. Compare the prediction with the held-out PET and with the simple outer-training
   mean PET atlas baseline. Report spatial correlations, MSE, MSE improvement, and
   the correlation between predicted and observed deviations from that baseline.

`deviation_r` is blank/null when either deviation is constant. Positive MSE
improvement means the model improves upon the atlas baseline. These are descriptive
out-of-sample scores, not p-values. Do not infer significance from treating the
overlapping LOSO training folds as independent experiments. Related participants,
repeated scans, or multi-site generalization require grouped/external splits beyond
this initial benchmark. Molecular specificity requires additional spatial-null
and paired-data validation; outcome prediction alone is not a mechanism test.

Outputs include `participants.tsv`, `folds.json` (training IDs, all candidate inner
scores, selected K), `predictions.npz` (observed/predicted/training-atlas arrays),
`bases.npz`, provenance, and a standalone report.

## Relationship to the paper and reuse

This is an independently written functional-connectivity branch, not a numerical
replication of [Wan et al., EigenBrain](https://doi.org/10.64898/2026.10.27.758642).
It does not implement their structural/geodesic model fusion, elbow selection,
clinical group analyses, or claim validation for any individual receptor system.
Its nested validation and atlas baseline deliberately evaluate a different,
stricter prediction question. The author's code was inspected at
`wanb-psych/EigenBrain_framework@11490822d8d885d89b2b2b3342d9513a84835bff`;
none of that repository's source or reference data is incorporated here.

The paper's eight group-template spatial-null successes do not imply that every
atlas in the same transmitter family is individually predictable. Likewise,
failure to reconstruct an atlas does not prevent using that measured reference
atlas in a properly tested spatial association analysis.

The new code is Apache-2.0. Optional external packages and downloaded maps retain
their own terms, including neuromaps' CC-BY-NC-SA license; optional installation
does not waive those terms. Cite the original map sources as well as software:

- [neuromaps](https://netneurolab.github.io/neuromaps/)
- [BrainSpace](https://brainspace.readthedocs.io/)
- [BrainSMASH](https://brainsmash.readthedocs.io/)
- [EigenBrain preprint](https://doi.org/10.64898/2026.10.27.758642)

No patient data, reference connectomes, or molecular atlases are distributed here.
