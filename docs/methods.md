# Methods and compatibility

The reference is **source commit `059ade5fc2a332bb2a246ef7db0bd7c4217446e4`** of
[pBFSLab/rLNM](https://github.com/pBFSLab/rLNM/tree/059ade5fc2a332bb2a246ef7db0bd7c4217446e4).
This is a source-level port of that snapshot, not a claim to reproduce every
result of every preprint version. The initial and revised papers use different
descriptions and defaults; do not mix them without specifying the version.

## Functional projection and sensitivity

The projection averages selected rows of a reference FC matrix, ignoring NaNs
per target voxel and returning zero if all selected entries are NaN (upstream
behaviour). Inf is rejected. This is **not** generally equivalent to correlating
a lesion-averaged BOLD time series with target time series. A supplied matrix's
preprocessing, population, and r/z scale remain the data provider's responsibility.

For each projected lesion map, MATLAB-style Hazen percentiles define the upper
and lower tails (25% each by default). Upper-tail values receive +1, then lower-
tail values receive -1. If thresholds coincide, the latter assignment wins.
At each voxel the more frequent sign determines the signed proportion; equal
positive and negative counts produce zero. These are percentile thresholds,
not fixed correlation magnitudes.

Spatial nulls translate lesions by integer voxel offsets within an explicit
placement mask. Shape, volume, and orientation are preserved; lesions may
overlap. The placement mask is not necessarily the FC target/FDR domain.
This release requires all observed lesion voxels to be inside the placement
mask and all placement voxels to have connectome rows. It refuses implicit
clipping or silent exclusion.

- `reposition` samples valid translations without frequency matching.
- `annealing` matches the distribution of voxelwise lesion coverage using the
  upstream sorted-frequency RMSE objective and simulated-annealing move rule.
  It does not preserve pairwise lesion overlaps or coverage at fixed locations.
  The RMSE target is an early-stop criterion; KS acceptance remains separate.
  SciPy's asymptotic two-sample KS implementation is used for acceptance; it
  has not been established to match MATLAB `kstest2` numerically. The KS check
  is a heuristic inherited from upstream, not an independent calibration proof.
  Failed candidates are retried to a finite configurable limit, then the run
  fails rather than accepting an invalid null.

Sensitivity testing counts equally or more extreme null statistics, including
ties, and uses `(extreme_count + 1) / (actual_null_count + 1)`.

- `upstream-sign` chooses the tail according to the observed sign. This matches
  the source's comparison rule and is not a conventional two-sided test.
- `two-sided` compares absolute values and is a deliberate alternative.

BH-FDR covers every target voxel in the FC index list. Retained maps additionally
exclude zero observed overlaps. Default sensitivity FDR is 0.005, as in the
pinned source. The synthetic demo explicitly chooses two-sided testing and 0.05.

## Selective specificity

The analysis domain is an explicit binary brain mask fixed before analysis.
Each symptom needs at least two independent lesions. For each lesion:

1. Average each symptom's signed maps, excluding the lesion from its own group.
2. Give those symptom averages equal weight in the shared LOLO regressor.
3. Regress the lesion's map on that regressor across domain voxels, with an
   intercept, and retain the signed OLS residual.

Each symptom is compared to all others using voxelwise Welch t statistics.
The reported effect is the target-minus-contrast mean divided by the
degrees-of-freedom-weighted pooled standard deviation. Upstream's zero-spread
convention is retained: t and effect are zero where their denominators vanish.

The shared display map averages positive-clipped symptom means and can be
masked by the intersection of explicit sensitivity masks. Neither that clipping
nor those sensitivity masks enters the regression or FDR domain. When masks
are omitted, the display is labelled **unmasked**; no sensitivity claim is made.
Unlike upstream's whole-grid display export, Python's display maps are restricted
to the declared analysis domain. Inferential outputs use the same domain.

**Permutation mode is fixed-residual label permutation, as in the pinned code.**
The label-dependent LOLO regression is not refitted for every permutation.
This procedure's calibration must be assessed for the intended study; numerical
agreement alone is not evidence of false-positive control. Repeated measures,
clustered sites, covariate-adjusted permutations, and full-pipeline refitting
are not implemented. Use independent observations and an appropriate design.

`sided="one"` tests greater target t; `"two"` tests absolute t. Both use the
plus-one correction. BH-FDR is applied across domain voxels **separately for
each symptom**, not jointly across all symptoms. Default retention is strict
`q < 0.05`; effect magnitude is not a second selection criterion.

## Reproducibility and intentional differences

- NumPy PCG64 uses `SeedSequence([seed, draw_id])`. Changing worker count or
  batch size does not change the sample assigned to a draw. These are not
  MATLAB's random sequences. Comparisons must use common saved permutations.
- Float64 is used for statistics and outputs. No GPU approximation is used.
- Shape, affine, coded forms, and spatial units are checked for every image.
  The `space` declaration still needs human verification: an affine cannot
  identify a template or establish successful anatomical registration.
- All arrays flatten in zero-based NIfTI array order with `order="F"`.
  MATLAB's one-based indices are not accepted as Python indices. FreeSurfer
  `MRIread().vol` axis conventions must also be resolved before importing the
  upstream C1 spreadsheet/matrix. This release does not automate that conversion.
- The C2 example importer explicitly repairs missing units to millimetres only
  after verifying the pinned example's geometry against its mm template.
  Originals are preserved and both original and imported hashes are recorded.
- Unknown parameters, duplicate IDs/paths, non-finite input maps, and implicit
  resampling are rejected. p/q maps use 1 outside the domain; effects use 0.
- Nulls are streamed. Full null maps are not retained; deterministic settings
  and per-draw diagnostics permit regeneration. No resume/cache feature yet.

## What this release does not establish

It does not reproduce the paper's epilepsy surgery outcome prediction, all
reported lesion cohorts, or structural-connectivity extension. It does not
validate clinical use, identify a causal treatment target, or claim improved
statistical power. Its contribution is an inspectable implementation and a
foundation for comparisons and subsequent validation.

