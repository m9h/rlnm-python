"""Synthetic smoke tests and import of the separately obtained upstream example."""

import csv
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np

from . import UPSTREAM_COMMIT
from .io import image_data, load_image, sha256
from .pipeline import dump_json


def _empty_directory(output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Example output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _nifti(path, data, affine):
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float64), affine)
    image.header.set_xyzt_units("mm")
    image.set_qform(affine, 1)
    image.set_sform(affine, 1)
    nib.save(image, path)


def _manifest(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["participant_id", "symptom", "map"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)


def create_demo(output):
    output = _empty_directory(output)
    rng = np.random.default_rng(17)
    shape = (9, 10, 8)
    affine = np.array([[2.0, 0, 0, -8], [0, 2.0, 0, -10], [0, 0, 2.0, -8], [0, 0, 0, 1.0]])
    mask = np.zeros(shape, dtype=bool)
    mask[1:-1, 1:-1, 1:-1] = True
    _nifti(output / "brain_mask.nii.gz", mask, affine)
    grid = np.indices(shape)
    shared = np.sin(grid[0] / 2) * 0.3 + np.cos(grid[1] / 3) * 0.2
    rows = []
    for group, center in enumerate(([2, 3, 3], [6, 3, 3], [4, 7, 4])):
        distance = sum((grid[i] - center[i]) ** 2 for i in range(3))
        signal = np.exp(-distance / 3)
        for sample in range(6):
            name = f"group-{group}_sample-{sample}"
            array = (shared + signal + rng.normal(0, 0.1, shape)) * mask
            _nifti(output / f"{name}.nii.gz", array, affine)
            rows.append(
                {"participant_id": name, "symptom": f"synthetic-{group}", "map": f"{name}.nii.gz"}
            )
    _manifest(output / "participants.tsv", rows)
    dump_json(
        output / "specificity.json",
        {
            "analysis": "specificity",
            "manifest": "participants.tsv",
            "space": "synthetic",
            "brain_mask": "brain_mask.nii.gz",
            "n_permutations": 199,
            "seed": 17,
            "fdr": 0.05,
            "sided": "one",
            "jobs": 1,
        },
    )
    indices = np.flatnonzero(mask.ravel(order="F"))
    coords = np.array(np.unravel_index(indices, shape, order="F")).T
    distances = np.sum((coords[:, None] - coords[None, :]) ** 2, axis=2)
    fc = np.exp(-distances / 8) - 0.2
    np.save(output / "connectome.npy", fc)
    np.save(output / "indices.npy", indices)
    dump_json(
        output / "connectome.json",
        {
            "matrix": "connectome.npy",
            "indices": "indices.npy",
            "reference": "brain_mask.nii.gz",
            "space": "synthetic",
            "index_order": "nifti-fortran-zero",
        },
    )
    lesion_rows = []
    for number, xyz in enumerate(([2, 3, 3], [3, 3, 3], [2, 4, 3], [3, 4, 3])):
        lesion = np.zeros(shape)
        x, y, z = xyz
        lesion[x : x + 2, y, z] = 1
        name = f"lesion-{number}"
        _nifti(output / f"{name}.nii.gz", lesion, affine)
        lesion_rows.append(
            {"participant_id": name, "symptom": "synthetic", "map": f"{name}.nii.gz"}
        )
    _manifest(output / "lesions.tsv", lesion_rows)
    dump_json(
        output / "sensitivity.json",
        {
            "analysis": "sensitivity",
            "manifest": "lesions.tsv",
            "space": "synthetic",
            "brain_mask": "brain_mask.nii.gz",
            "connectome": "connectome.json",
            "n_permutations": 99,
            "seed": 17,
            "fdr": 0.05,
            "tail": "two-sided",
            "null_model": {"method": "reposition"},
        },
    )
    return output


def prepare_upstream(source, output):
    """Prepare C2 sample manifests; don't redistribute or modify source images."""
    source = Path(source).resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Expected upstream commit {UPSTREAM_COMMIT}; got {commit}")
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if status.strip():
        raise ValueError("Upstream tracked files have local modifications")
    output = _empty_directory(output)
    root = source / "C2_specificity_test/disease_data"
    template = source / "C1_sensitivity_test/template/norm_8mm.nii.gz"
    image = load_image(template)
    imported = []

    def import_map(path, relative):
        source_image = load_image(path)
        if source_image.shape != image.shape or not np.allclose(
            source_image.affine, image.affine, atol=1e-4, rtol=0
        ):
            raise ValueError(f"Upstream example does not match template geometry: {path}")
        unit = source_image.header.get_xyzt_units()[0]
        if unit not in ("mm", "unknown"):
            raise ValueError(f"Unexpected upstream spatial unit: {unit}")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        # This repair is specific to the pinned, geometry-matched upstream sample.
        # Generic run() never silently infers units or resamples inputs.
        source_image.header.set_xyzt_units("mm")
        nib.save(source_image, destination)
        imported.append(
            {
                "source": str(path),
                "source_sha256": sha256(path),
                "output": str(relative),
                "spatial_unit_before": unit,
                "spatial_unit_after": "mm",
                "output_sha256": sha256(destination),
            }
        )
        return str(destination)

    rows, sensitivity_masks = [], {}
    for folder in sorted((root / "Lesion_rmaps").iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        files = {}
        for path in sorted(folder.glob("*.nii*")):
            if path.name.endswith(".nii.gz"):
                files[path.name[:-7]] = path
            elif path.name.endswith(".nii"):
                files.setdefault(path.name[:-4], path)
        for stem, path in sorted(files.items(), key=lambda item: item[1].name):
            rows.append(
                {
                    "participant_id": f"{folder.name}/{stem}",
                    "symptom": folder.name,
                    "map": import_map(path, Path("maps") / folder.name / path.name),
                }
            )
        sensitivity_masks[folder.name] = import_map(
            root / "Lesion_sensitivitymap" / folder.name / "significant_8mm.nii.gz",
            Path("sensitivity_masks") / f"{folder.name}.nii.gz",
        )
    data = image_data(image)
    binary = nib.Nifti1Image((data != 0).astype(np.uint8), image.affine, image.header.copy())
    binary.set_data_dtype(np.uint8)
    nib.save(binary, output / "brain_mask.nii.gz")
    _manifest(output / "participants.tsv", rows)
    dump_json(
        output / "specificity.json",
        {
            "analysis": "specificity",
            "manifest": "participants.tsv",
            "space": "upstream-MNI-8mm",
            "brain_mask": "brain_mask.nii.gz",
            "n_permutations": 10000,
            "seed": 0,
            "fdr": 0.05,
            "sided": "one",
            "jobs": 1,
            "sensitivity_masks": sensitivity_masks,
            "input_provenance": "import_provenance.json",
        },
    )
    dump_json(
        output / "import_provenance.json",
        {
            "source_commit": commit,
            "source_template_sha256": sha256(template),
            "mask_conversion": "nonzero template voxels -> binary; affine and array axes preserved",
            "n_maps": len(rows),
            "source": str(source),
            "imported_maps": imported,
        },
    )
    return output
