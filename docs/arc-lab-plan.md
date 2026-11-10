# ARC validation on the DGX Spark and NAS

Updated 9 November 2026. ARC processing will run at the lab: the NAS will hold
versioned inputs and completed results, and the DGX Spark will execute analyses
using local scratch storage. Begin with the existing normalized lesions and
language scores; add longitudinal diffusion and cortical analysis after a pilot.

This is an execution plan. The ARC importer, continuous-outcome benchmark, and
calibration suite are still to be implemented. No ARC processing has been launched
on the Spark. Its saved SSH hostname did not resolve from the development machine
on this date, including a retry outside the local sandbox.

## Storage and execution

| Location | Contents and use |
|---|---|
| NAS source archive | Pinned ARC releases, author derivatives, reference connectomes, download manifests, and checksums. Preserve original files. |
| Spark local storage | Repository checkout, environment, staged participant inputs, temporary files, tractograms, and active job outputs. Verify that these paths resolve to local storage. |
| NAS derivatives | Completed, checksum-verified outputs in distinct directories per pipeline version and run. Include QC, exclusions, transforms, parcel order, and provenance. |
| GitHub | Code, configuration templates, methods, and aggregate validation records. Keep downloaded imaging and participant-level results in lab storage. |

Use `/data/datasets/rlnm-arc` as a proposed NAS project root **only after verifying
the mount**. Do not let a missing NAS silently turn this into a local directory.
Choose a local scratch path after checking free space and filesystem type; a home
directory or symlink may itself reside on the NAS. Avoid repeated random access
to large tractograms across NFS. Stage a participant's inputs locally, then copy
completed outputs back with checksum verification before removing local copies.

The current package uses CPU NumPy/SciPy calculations; it has no CUDA backend.
Start the pilot with conservative worker/thread counts and measure memory, runtime,
and temporary disk usage. Increase concurrency from those measurements. Verify
ARM64 support for each external executable or container before a cohort launch;
containerization alone does not establish architecture compatibility.

### First session at the lab

On the Spark, inspect the machine and intended storage locations:

```bash
hostname
uname -m
free -h
findmnt -rn -t nfs,nfs4 -o SOURCE,TARGET,OPTIONS
findmnt --target /data -o TARGET,SOURCE,FSTYPE,OPTIONS
df -h / /data
command -v uv
command -v docker
```

Inspect the selected scratch directory with `findmnt --target` and `df -h` as well.
Record the actual architecture, available resources, and mount paths in the lab
run record. After obtaining a checkout on verified local storage, run the existing
core environment and tests:

```bash
git rev-parse HEAD
uv sync --locked --python 3.12
uv run pytest -q
uv run rlnm demo --output outputs/lab-smoke-input
uv run rlnm run outputs/lab-smoke-input/specificity.json --output outputs/lab-smoke-result
```

Use fresh output directories on subsequent runs. These commands verify the
existing synthetic workflow; they do not process ARC. Test optional molecular
dependencies separately with `uv sync --locked --all-extras` and
`uv run --all-extras pytest -q` before scheduling molecular work.

## First patient benchmark

Use the [ARC dataset, OpenNeuro ds004884](https://openneuro.org/datasets/ds004884)
and the authors' [normalized derivatives and tables](https://github.com/neurolabusc/AphasiaRecoveryCohortDemo).
The [dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11384737/) describes
language scores, multimodal imaging, and repeated visits.

An initial repository inspection on 8 November found 228 individual normalized
lesion masks. Matching their identifiers to finite `wab_aq` values in
`Scripts/merged_jhu_participants.tsv` yielded 226 candidate participants. The table
contains 284 distinct identifiers, so its row count is not the imaging cohort size.
Recompute this audit against a pinned source commit before analysis; registration,
assessment timing, covariate completeness, and exclusions determine the final sample.

1. **Prepare and audit inputs.** Pin releases and commits, record checksums, join
   participant identifiers explicitly, review lesion/anatomical alignment, and
   record scan-to-assessment intervals. Keep an exclusion reason for each omitted
   record. Freeze outcome definitions and split assignments before model comparison.
2. **Establish baselines.** Predict continuous WAB-AQ with a training-set mean,
   lesion volume plus available covariates, and regional lesion burden. Use the same
   participants and splits for paired comparisons. The authors' own
   [benchmark discussion](https://github.com/neurolabusc/AphasiaRecoveryCohortDemo#running-the-deep-learning-script-on-this-dataset-to-predict-wab-aq)
   highlights the strength of age and lesion volume.
3. **Add network measurements.** Compare functional lesion-network features and
   structural disconnection features, then their combination. Identify and pin the
   healthy reference connectomes before generating features. Specify whether the
   functional projection averages correlation rows or uses lesion-averaged BOLD;
   these operations are not interchangeable. Preserve structural and functional
   feature semantics separately.
4. **Evaluate held-out predictions.** Keep every visit from a participant in the
   same split. Fit imputation, scaling, covariate adjustment, feature selection,
   outcome-derived network maps, and hyperparameters within training folds. Report
   MAE, out-of-sample R-squared, and paired improvement over simple baselines with
   uncertainty. A within-ARC holdout does not establish external-cohort performance.
5. **Calibrate inference.** Use real lesion geometry with simulated null, nuisance,
   local-effect, and network-effect outcomes. Compare fixed-residual and refitted
   procedures under their stated exchangeability assumptions. Record false-positive
   rates and uncertainty, power, and localization error. Freeze the simulation
   scenarios and acceptance tolerances before examining calibration results.

The current categorical specificity calculation remains a separately documented
method. Adding continuous-outcome prediction does not, by itself, validate its
permutation procedure.

## Longitudinal validation from the Pruckner work

Pruckner and colleagues studied 59 epilepsy surgery patients and related disrupted
pathways to subsequent cortical thinning and white-matter change. Their results
support network-mediated degeneration. The cohort-wide connectivity increases
seen with independent cross-sectional reconstructions were absent with the robust
longitudinal method; this does not establish that biological plasticity never
occurs. The surgical imaging data require an author request.
[Brain paper](https://doi.org/10.1093/brain/awaf344).

Their companion method constructs one within-subject tractogram with corresponding
streamlines across sessions, then estimates changes in density weights through
symmetric or differential SIFT2 optimization. The
[implementation and tutorial](https://github.com/ppruc/sift2_unbiased) include
phantom examples, while the
[appendix](https://github.com/ppruc/sift2_unbiased/blob/main/tutorial/part4_appendix.md)
describes integration with longitudinal FreeSurfer. This offers a proposed
additional validation target: whether inferred disconnection relates to later
measured change in connected, preserved tissue.

For an ARC pilot:

- Audit availability of at least two suitable T1/DWI visits per participant,
  gradient tables, distortion-correction metadata, acquisition changes, and visit
  intervals. Counts of total sessions do not establish usable longitudinal pairs.
- Run the upstream phantom benchmark first, then a small prespecified set of ARC
  participants spanning lesion sizes and acquisition conditions. Compare independent
  session processing with the longitudinal method before scaling up.
- Pin the exact MRtrix source commits and binaries. The tutorial currently requires
  separate `sift2diff` and `fixelcorrespondence` branches, in addition to MRtrix3,
  FreeSurfer, FSL, and ANTs. Ordinary released `tcksift2` should not be assumed to
  provide the required options. Follow the
  [tutorial prerequisites](https://github.com/ppruc/sift2_unbiased/blob/main/tutorial/README.md).
- Review lesion-aware tissue segmentation, surface placement, longitudinal alignment,
  and streamline endpoints. Use identical parcel definitions across visits, retain
  SIFT2 scaling information, and distinguish tractography weights from literal axon
  counts or disconnectome probabilities.
- Relate baseline disconnection to subsequent cortical/white-matter change while
  accounting for repeated observations, baseline tissue status, lesion burden, and
  elapsed time. Prespecify regions and multiple-testing families, including suitable
  comparison regions and sensitivity analyses for distance from the lesion.

ARC's chronic post-stroke visits are not pre-injury scans. Adapting the method can
test later changes after established injury; it cannot reproduce a pre-/post-surgery
experiment. If the objective is prospective prediction, construct predictors only
from information available at baseline. A template built from all visits can help
estimate retrospective change but would introduce future information into a
baseline-only predictor. Record those two workflows separately.

## Deliverables and completion criteria

The first lab milestone is a verified environment, a pinned ARC inventory, an
audited participant join, and baseline held-out predictions. The next milestone is
paired network-feature comparisons plus statistical calibration. Longitudinal
tractography and cortical measurements form a subsequent pilot with their own
phantom, QC, and repeatability evidence.

For each milestone preserve the source commit, environment or container digest,
dataset versions and hashes, inclusion criteria, split definitions, random seeds,
complete predictions/statistics, runtime and memory measurements, and a readable
report. Record negative findings. Publish claims only at the level supported by
those observations; the [existing validation](validation.md) remains the record of
what has already run.
