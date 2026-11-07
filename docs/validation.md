# Validation

Validation has separate levels. Passing one does not imply the others.

1. **Numerical unit tests:** Welch statistics against SciPy, BH adjustment against
   SciPy, OLS residuals against independent least-squares design matrices, and
   exact enumerated permutation tails.
2. **Reproducibility:** identical p/q arrays with different worker counts and
   batch sizes; shape and volume preservation under deterministic spatial nulls.
3. **Spatial/pipeline tests:** NIfTI round trips, every-subject affine checks,
   conflicting qform/sform rejection, unit mismatch rejection, duplicate inputs,
   unknown parameters, two end-to-end synthetic analyses, and output provenance.
4. **Original MATLAB source oracle:** the CI workflow runs pinned upstream S3 and
   S2 calculation blocks through Octave, alongside upstream Welch, effect, and BH
   functions. Explicit shared permutation rows isolate numerical differences from
   different RNG sequences. It compares LOLO regressors, residuals, statistics,
   effects, both permutation tails, BH results, projection, and signed overlap.
5. **Upstream data smoke run:** the supplied C2 sample is imported and processed
   with 10,000 permutations. This checks integration, not agreement with every
   published result or an executed end-to-end MATLAB pipeline.

## Run locally

```bash
uv run pytest -q
uv run python scripts/validate_reference.py \
  --upstream data/upstream --engine octave --output outputs/oracle.json
```

If a MATLAB interpreter and the required percentile functionality are available,
use `--engine matlab` to execute the same comparison in MATLAB. MATLAB Runtime
alone cannot execute the source oracle. The oracle errors if its selected engine
is missing; it never substitutes Python code for that engine.

The CI artifact `octave-reference-validation` contains the observed comparison
errors. Linux, macOS and Windows unit/pipeline jobs run on Python 3.11 and 3.12.

## Recorded initial validation

[CI run 42347166590](https://github.com/m9h/rlnm-python/actions/runs/42347166590)
passed all seven jobs at implementation commit
`2b2237aaad97714cd23102240168ee1f696fef89`:

- 20 tests, lint, and distribution builds on six OS/Python combinations.
- [Synthetic source comparison](validation-results/oracle.json): 18 maps,
  29 voxels, 83 shared permutation rows; largest error `6.66e-16`.
- [Upstream C2 source comparison](validation-results/example-oracle.json):
  184 maps, 5,486 voxels, 83 shared permutation rows; largest error `1.42e-14`.
  One- and two-sided permutation p-values matched exactly in both comparisons.
- [Python C2 integration run](validation-results/upstream-c2.json): 10,000
  permutations, seed 0, FDR 0.05. Significant voxel counts were 309 (Face),
  1,320 (Lan), and 1,397 (Spatial). These observations are reproducibility
  records for this implementation and dataset, not independent biological validation.

The recorded source oracle used Octave, not a licensed MATLAB interpreter.
It exercises the original MATLAB arithmetic but does not establish equivalence
of MATLAB's image I/O, RNG, or `kstest2`. The shared display map is not covered
by the oracle. Later documentation/whitespace-only commits do not change these
tested calculations.

## Remaining scientific validation

- False-positive calibration for the fixed-residual, label-dependent specificity
  procedure and for the selected-sign sensitivity option.
- Full-pipeline refitting and constrained exchangeability for more complex designs.
- Cross-engine annealing/KS equivalence and sensitivity to placement constraints.
- Full-resolution datasets, independently held-out cohorts, and outcome prediction.
- Structural-disconnection and stimulation-specific extensions.

No claims of superiority over BCBtoolkit, Lead-DBS, or the original rLNM are made.
