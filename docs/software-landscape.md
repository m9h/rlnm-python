# Python tools for lesion networks and DBS analysis

Research snapshot: 8 November 2026.

For DBS electric-field and pathway-activation modeling, OSS-DBSv2 is the preferred
Python starting point for this project. For functional lesion-network analysis,
Nilearn is the relevant foundation; DIPY supplies diffusion and streamline tools.
Lead-DBS spans electrode localization, registration, simulation, and cohort
analysis, so replacing its complete workflow requires several components.

The project direction is interoperability with established imaging and DBS tools,
with reproducible inference and molecular annotation in rlnm-python. The
recommendations below distinguish existing software capabilities from proposed
project extensions.

## How the scientific approaches connect

| Approach | Main question | Connection to this project |
|---|---|---|
| Michel Thiebaut de Schotten and disconnectome analysis | Which structural connections are disrupted by a lesion? | A complementary description of lesion effects based on white-matter pathways. |
| Robust lesion network mapping | Which functional networks show symptom-associated connectivity beyond the selected null hypotheses? | The statistical and functional-mapping core implemented here. |
| Andreas Horn and connectomic DBS analysis | Which local regions, pathways, or distributed networks engaged by stimulation relate to outcomes? | A complementary intervention-based way to investigate symptom networks. |
| neuromaps and EigenBrain | Which molecular and organizational patterns characterize those networks? | Reference-map annotation and an experimental molecular reconstruction benchmark. |

Structural disconnection, functional connectivity, and stimulation engagement are
different measurements. Combining them requires explicit spatial mapping and
appropriate statistical models. A tract-intersection probability is not equivalent
to a signed functional-connectivity value, and spatial agreement alone does not
establish a causal mechanism. See the [BCBtoolkit methods paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5863218/),
[rLNM preprint](https://doi.org/10.64898/2026.05.18.725812), and
[Horn and colleagues' circuit-mapping work](https://www.nature.com/articles/s41593-024-01570-1).

## Python options by task

| Task | Software | Scope and practical choice |
|---|---|---|
| DBS electric fields and pathway activation | [OSS-DBSv2](https://oss-dbsv2.readthedocs.io/en/latest/) | Standalone Python/CLI simulation with imaging, electrodes, and stimulation settings as inputs. The strongest fit for the simulation component. |
| Functional connectivity and seed-based analysis | [Nilearn](https://nilearn.github.io/stable/manipulating_images/masker_objects.html) | Signal extraction and functional-connectivity building blocks. A basis for a Python network-mapping workflow, rather than a complete Lead Mapper replacement. |
| Diffusion tractography and structural connectivity | [DIPY](https://docs.dipy.org/stable/reference/dipy.tracking.html) | Tractography, streamline operations, and connectivity matrices. A disconnectome workflow still needs its own reference data, lesion-intersection rules, and validation. |
| Registration and spatial transformations | [ANTsPy](https://antspy.readthedocs.io/en/latest/) | Python access to medical-image registration and related processing. Registration quality remains an explicit QC task. |
| Interactive anatomical review | [3D Slicer with SlicerNetstim](https://github.com/netstim/SlicerNetstim) | Python-enabled desktop modules for DBS-related anatomy and registration refinement; not an ordinary pip-only application. |
| Reference-map interpretation | [neuromaps](https://netneurolab.github.io/neuromaps/) | Brain annotations, coordinate transformations, and spatial comparison tools. |
| Connectivity gradients and spatial nulls | [BrainSpace](https://brainspace.readthedocs.io/) and [BrainSMASH](https://brainsmash.readthedocs.io/) | Gradient estimation and variogram-based spatial surrogates, respectively. Used by the optional molecular workflows. |

OSS-DBSv2 already integrates with Lead-DBS: Lead-DBS prepares anatomy, reconstructs
electrodes, and passes simulation inputs to the Python backend. Its
`leaddbs2ossdbs` converter can turn a Lead-DBS stimulation parameter file into a
standalone JSON configuration. This offers a practical route to Python batch
simulation while retaining the established anatomical preparation workflow.
[Integration documentation](https://oss-dbsv2.readthedocs.io/en/latest/lead_dbs.html).

## Horn laboratory software ecosystem

The following are documented components of the group's ecosystem, rather than a
claim that every study uses every tool. Most analysis explorers are part of the
MATLAB-centered Lead-DBS suite.

| Component | Role |
|---|---|
| Lead Connectome Mapper, also called Lead Mapper | Computes structural or functional connectivity from arbitrary image seeds or stimulation volumes; exports connectivity maps or matrices. |
| Lead Connectome | Processes diffusion and resting-state MRI for connectomic analysis, including applications outside DBS. |
| Lead Group | Supports cohort-level electrode, stimulation, and outcome analyses. |
| Sweetspot Explorer | Associates local stimulation distribution with clinical outcomes. |
| Fiber Filtering Explorer | Identifies streamlines whose stimulation engagement is associated with outcomes. |
| Network Mapping Explorer | Associates whole-brain connectivity patterns with outcomes and evaluates predictive models. |

See [Lead Mapper](https://netstim.gitbook.io/leaddbs/connectomics/lead-mapper),
[Lead Connectome](https://www.lead-dbs.org/about/lead-connectome/), the
[official walkthroughs](https://www.lead-dbs.org/helpsupport/knowledge-base/walkthrough-videos/),
and the [Lead-DBS methods paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10144063/).

Supporting components include SPM, ANTs, and FSL for image processing and spatial
transformations, and DSI Studio among the supported tractography engines.
[Coordinate-transform implementation](https://github.com/netstim/leaddbs/blob/master/ea_map_coords.m),
[tractography integration](https://www.lead-dbs.org/about/lead-connectome/).

The Python-enabled SlicerNetstim extension includes **WarpDrive**, which permits
manual refinement of nonlinear registration, and **Lead-OR**, which connects
surgical trajectories and anatomical imaging with intraoperative electrophysiology.
Its additional modules import atlases and stereotactic plans.
[SlicerNetstim documentation](https://github.com/netstim/SlicerNetstim).

## What EigenBrain adds

[Wan and colleagues' EigenBrain preprint](https://www.biorxiv.org/content/10.64898/2026.10.27.758642v1.full)
fits molecular reference maps using MRI-derived diffusion-map gradients. It uses
360 cortical parcels and transfers reference-model weights to aligned individual
gradients. Eight receptor/transporter maps exceeded the tested spatial null with
functional gradients: 5-HT1a, 5-HTT, D2, DAT, H3, VAChT, MOR, and KOR. Structural
and geometric single-modality models did not exceed that null. Individual
receptor-density differences remain unvalidated against individual receptor PET.

Its [released implementation](https://github.com/wanb-psych/EigenBrain_framework)
is already Python, using BrainSpace, neuromaps, statsmodels, and BrainSMASH. The
inspected snapshot contains research scripts with fixed paths and references to
external assets/helpers, rather than a packaged application. No license file was
present at commit `11490822d8d885d89b2b2b3342d9513a84835bff`; rlnm-python incorporates
none of its source code.

The [Dallas modeling script at that snapshot](https://github.com/wanb-psych/EigenBrain_framework/blob/11490822d8d885d89b2b2b3342d9513a84835bff/py07_dallas_modeling.py)
trains on the cohort's mean PET and evaluates its members. Our experimental
benchmark instead excludes each outer-test participant from the training PET
mean, selects component count within inner validation, and compares prediction
against the training-only atlas baseline.

Molecular-map association and molecular reconstruction answer different questions.
An atlas can be useful for spatial annotation even when its reconstruction from
MRI fails. Spatial nulls for map association also complement, rather than replace,
lesion-location and symptom-label nulls. See the
[neuromaps spatial-null documentation](https://netneurolab.github.io/neuromaps/user_guide/nulls.html).

## Implemented capabilities and validation

rlnm-python v0.2.0, implementation commit
`471255ae0f395147bfbbbb7b60a21798b61e2c3d`, provides:

- Functional projection, spatial-null sensitivity, selective specificity, strict
  NIfTI geometry checks, and reproducible provenance.
- `rlnm annotate`: continuous-map parcellation, a common valid-data domain,
  annotation-side BrainSMASH nulls, BH correction, and coverage/variogram reports.
  Direct neuromaps fetching currently supports MNI152 volumes.
- `rlnm benchmark-eigenbrain`: an independent FC-only experiment with nested
  leave-one-participant-out validation and an external connectivity reference.

All ten jobs in [CI run 42497088725](https://github.com/m9h/rlnm-python/actions/runs/42497088725)
passed, including the 35-test suite with optional dependencies on Linux, macOS,
and Windows, plus the original MATLAB arithmetic comparisons executed via Octave.
The synthetic molecular benchmark did not outperform its atlas baseline. The
small synthetic annotation example also exhibited poor variogram fits; its
p-values are not evidence of calibration. These outcomes are preserved in the
[validation record](validation-results/molecular-v02.json).

The package does not currently perform electrode reconstruction, registration,
electric-field simulation, or structural disconnectome generation. Configuration
and scientific limitations are documented in [molecular workflows](molecular.md),
[core methods](methods.md), and [validation](validation.md).

## Recommended next work

The [ARC lab validation plan](arc-lab-plan.md) develops these priorities into
patient-data benchmarks on the DGX Spark and NAS, and incorporates Pruckner and
colleagues' longitudinal quantitative tractography as an additional validation
direction.

1. Add explicit adapters for Lead Mapper outputs and stimulation derivatives,
   preserving coordinate systems, units, modality, and source provenance.
2. Add structural disconnection using a documented tractogram/reference dataset
   and explicit streamline-intersection rules, with independent validation.
3. Use standalone OSS-DBSv2 when electric fields or pathway activation are needed.
   Keep those outputs distinct from functional connectivity and tract-intersection
   probabilities in the data model.
4. Evaluate whether molecular features improve held-out symptom or stimulation
   outcome prediction beyond connectivity alone, using suitable paired data and
   grouped/external validation where sites, families, or repeated scans require it.

Optional software and downloaded atlases retain their own licenses. In particular,
neuromaps' terms apply even when installed as an optional dependency. The original
atlas publications should be cited alongside the software.
