"""Synthetic fixtures only; these arrays contain no participant imaging data."""

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial.distance import cdist

from .research import dump_json


def create_research_demo(output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(31)
    shape = (10, 6, 1)
    coordinates = np.indices(shape).reshape(3, -1).T.astype(float)
    n = len(coordinates)
    labels = np.arange(1, n + 1)
    blocks = (coordinates[:, 0] >= 5).astype(int)
    smooth = np.sin(coordinates[:, 0] / 2) + np.cos(coordinates[:, 1] / 2)
    other = np.cos(coordinates[:, 0] * 1.2) + 0.2 * rng.normal(size=n)

    def save(name, data):
        image = nib.Nifti1Image(
            np.asarray(data, dtype=float).reshape(shape), np.diag([2.0, 2.0, 2.0, 1.0])
        )
        image.header.set_xyzt_units("mm")
        nib.save(image, output / name)

    save("atlas.nii.gz", labels)
    save("mask.nii.gz", np.ones(n))
    save("target.nii.gz", smooth + 0.25 * rng.normal(size=n))
    save("annotation-a.nii.gz", smooth)
    save("annotation-b.nii.gz", other)
    np.savez_compressed(
        output / "distances.npz",
        parcel_ids=labels,
        blocks=blocks,
        distances=cdist(coordinates, coordinates),
    )
    dump_json(
        output / "annotate.json",
        {
            "map": "target.nii.gz",
            "mask": "mask.nii.gz",
            "atlas": "atlas.nii.gz",
            "distances": "distances.npz",
            "space": "synthetic-grid",
            "n_nulls": 99,
            "seed": 31,
            "annotations": [
                {
                    "id": "synthetic-smooth",
                    "map": "annotation-a.nii.gz",
                    "citation": "synthetic demo",
                },
                {
                    "id": "synthetic-other",
                    "map": "annotation-b.nii.gz",
                    "citation": "synthetic demo",
                },
            ],
        },
    )
    # A paired synthetic cohort for pipeline verification, not a molecular-accuracy claim.
    features = np.column_stack((smooth, other, coordinates[:, :2], rng.normal(size=(n, 8))))

    def connectivity(x):
        x = x - x.mean(1, keepdims=True)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x @ x.T

    reference = connectivity(features)
    fc, pet = [], []
    for _ in range(8):
        deviation = rng.normal(size=n)
        altered = features.copy()
        altered[:, 0] += 0.4 * deviation
        fc.append(connectivity(altered))
        pet.append(smooth + 0.3 * deviation + 0.2 * rng.normal(size=n))
    np.savez_compressed(
        output / "paired.npz",
        parcel_ids=labels,
        fc=fc,
        pet=pet,
        subject_ids=[f"synthetic:paired-{i}" for i in range(8)],
    )
    np.savez_compressed(
        output / "reference.npz",
        parcel_ids=labels,
        fc=reference,
        subject_ids=["synthetic:external-reference"],
    )
    dump_json(
        output / "benchmark.json",
        {
            "dataset": "paired.npz",
            "reference": "reference.npz",
            "space": "synthetic-grid",
            "marker": "synthetic molecular profile",
            "reference_citation": "synthetic demo",
            "components": [3, 6, 10],
            "sparsity": 0.5,
            "seed": 31,
        },
    )
    return output
