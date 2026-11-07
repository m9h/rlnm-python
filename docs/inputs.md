# Inputs and outputs

All paths in a configuration are relative to that JSON file. Map paths in a TSV
are relative to the TSV. Spaces in filenames work. Arrays and images are read
locally; there is no upload or service dependency.

## Specificity from precomputed maps

`participants.tsv` (tab separated):

```text
participant_id	symptom	map
sub-01	language	maps/sub-01.nii.gz
sub-02	language	maps/sub-02.nii.gz
sub-03	motor	maps/sub-03.nii.gz
sub-04	motor	maps/sub-04.nii.gz
```

IDs and paths must be unique. Every image must be finite, 3-D, and match the
mask's grid and spatial units. The mask contains only 0/1 and is nonempty.

```json
{
  "analysis": "specificity",
  "manifest": "participants.tsv",
  "brain_mask": "brain_mask.nii.gz",
  "space": "your-exact-template-id",
  "n_permutations": 10000,
  "seed": 42,
  "jobs": 4,
  "batch_size": 128,
  "sided": "one",
  "fdr": 0.05,
  "sensitivity_masks": {
    "language": "masks/language.nii.gz",
    "motor": "masks/motor.nii.gz"
  }
}
```

`sensitivity_masks` is optional. If specified, every symptom must have an
explicit matching binary mask. Empty significance masks are valid. The maps
affect the shared-network display only; see [methods](methods.md).

```bash
uv run rlnm run analysis.json --output outputs/study
```

Outputs:

- `symptom-001_{mean,t,effect,p,q,significant,specific}.nii.gz`, etc.
- `shared_sensitivity_masked.nii.gz`, or explicitly `shared_unmasked.nii.gz`.
- `intermediates.npz`: input domain maps, labels, regressors, residuals, statistics,
  and the exact full-volume index list. Load with `allow_pickle=False`.
- `summary.json`: symptom-to-filename mapping, counts, domain, inference settings.
- `provenance.json`: resolved settings, versions, geometry, input/output hashes,
  RNG policy, time, and completion/failure status.
- `report.html`: standalone embedded PNG views and run summary.

Reports are quality-control aids without anatomical underlays; review registration
using original images in an imaging viewer. No browser/network access is required.

## Sensitivity from lesion masks and a functional connectome

Use the same TSV with binary lesion masks; the symptom column is optional.
A connectome is a square `.npy` array plus a zero-based `.npy` index list and
an explicit spatial reference. It is memory-mapped when loaded.

`connectome.json`:

```json
{
  "matrix": "fc.npy",
  "indices": "indices.npy",
  "reference": "reference.nii.gz",
  "space": "your-exact-template-id",
  "index_order": "nifti-fortran-zero"
}
```

`indices[i]` is the full-volume voxel associated with both row and column `i`.
Build coordinates with `np.unravel_index(indices, shape, order="F")`. Indices must
be unique. The matrix need not include every voxel of the reference volume,
but it must include every possible lesion-placement voxel. All target columns
enter FDR; the placement mask does not restrict that multiple-testing family.

Analysis configuration:

```json
{
  "analysis": "sensitivity",
  "manifest": "lesions.tsv",
  "brain_mask": "placement_mask.nii.gz",
  "connectome": "connectome.json",
  "space": "your-exact-template-id",
  "n_permutations": 10000,
  "seed": 42,
  "jobs": 2,
  "fdr": 0.005,
  "tail_fraction": 0.25,
  "tail": "upstream-sign",
  "null_model": {
    "method": "annealing",
    "max_candidates": 100,
    "max_placement_trials": 1000,
    "optimization_steps": 1000,
    "target_rmse": 0.005,
    "temperature": 0.0001,
    "cooling": 0.997,
    "ks_alpha": 0.05
  }
}
```

The same output/provenance conventions apply. `null_diagnostics.json` records
each accepted draw's candidate count, objective values, optimization steps, and
KS p value. Failed draws abort the analysis and mark provenance as failed.

## Reproduce the supplied C2 example

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/pBFSLab/rLNM.git data/upstream
git -C data/upstream checkout 059ade5fc2a332bb2a246ef7db0bd7c4217446e4
uv run rlnm prepare-upstream --source data/upstream --output data/prepared
uv run rlnm run data/prepared/specificity.json --output outputs/upstream
```

The C2 example does not need the large C1 FC matrix, so skipping LFS avoids
that download. The importer converts the template's nonzero voxels into a
binary mask and creates copies of maps with explicitly declared mm units.
`import_provenance.json` records these changes and original hashes; the analysis
records that provenance file's hash. Raw example data are not bundled here.

## Array API

```python
from rlnm import specificity, sensitivity_from_nulls

# maps: lesions x domain voxels; labels: one symptom name per independent lesion
result = specificity(maps, labels, n_permutations=10000, seed=42, jobs=4)

# Works with a generator; does not store all null maps.
sensitivity = sensitivity_from_nulls(
    observed_overlap, null_map_generator, tail="two-sided", fdr=0.05
)
```

These names describe user-provided arrays; they are not preloaded variables.
The array API deliberately leaves geometry validation to its caller. The CLI
performs that validation and should be preferred for file-based analyses.

