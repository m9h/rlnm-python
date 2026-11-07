"""Strict spatial I/O. No implicit resampling, template conversion, or clipping."""

import csv
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_image(path):
    image = nib.load(path)
    if len(image.shape) != 3:
        raise ValueError(f"Expected a 3-D NIfTI: {path}")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
        raise ValueError(f"Invalid spatial affine: {path}")
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    if qcode and scode and not np.allclose(qform, sform, atol=1e-4, rtol=0):
        raise ValueError(f"Conflicting qform/sform: {path}. Resolve geometry explicitly.")
    if not qcode and not scode:
        raise ValueError(f"Missing coded spatial transform: {path}")
    return image


def same_grid(image, reference, name="image"):
    if image.shape != reference.shape or not np.allclose(
        image.affine, reference.affine, atol=1e-4, rtol=0
    ):
        raise ValueError(f"Grid mismatch for {name}; register/resample explicitly before analysis")
    if image.header.get_xyzt_units()[0] != reference.header.get_xyzt_units()[0]:
        raise ValueError(f"Spatial unit mismatch for {name}")


def image_data(image, name="image", binary=False):
    data = image.get_fdata(dtype=np.float64)
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite values in {name}")
    if binary and not np.isin(data, [0, 1]).all():
        raise ValueError(f"{name} must contain only 0/1; threshold explicitly")
    return data.astype(bool) if binary else data


def load_mask(path, reference=None):
    image = load_image(path)
    if reference is not None:
        same_grid(image, reference, str(path))
    mask = image_data(image, str(path), binary=True)
    if not mask.any():
        raise ValueError(f"Empty analysis/placement mask: {path}")
    return image, mask


def read_manifest(path, *, require_symptoms=True):
    path = Path(path).resolve()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"participant_id", "map"} | ({"symptom"} if require_symptoms else set())
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Manifest needs columns: {', '.join(sorted(required))}")
        rows = list(reader)
    if not rows:
        raise ValueError("Empty participant manifest")
    seen_ids, seen_paths = set(), set()
    for row in rows:
        if any(not row[key] or not row[key].strip() for key in required):
            raise ValueError("Empty required manifest cell")
        filepath = (path.parent / row["map"]).resolve()
        if row["participant_id"] in seen_ids or filepath in seen_paths:
            raise ValueError("Duplicate participant ID or image path in manifest")
        if not filepath.is_file():
            raise ValueError(f"Missing input map: {filepath}")
        seen_ids.add(row["participant_id"])
        seen_paths.add(filepath)
        row["map"] = str(filepath)
    return rows


def load_maps(rows, reference, mask):
    vectors = []
    for row in rows:
        image = load_image(row["map"])
        same_grid(image, reference, row["map"])
        vectors.append(image_data(image, row["map"]).ravel(order="F")[mask.ravel(order="F")])
    return np.stack(vectors)


def write_map(path, values, reference, indices, *, outside=0):
    values, indices = np.asarray(values), np.asarray(indices)
    if values.shape != indices.shape or values.ndim != 1:
        raise ValueError("Map values and voxel indices must be matching vectors")
    array = np.full(np.prod(reference.shape), outside, dtype=np.float64)
    array[indices] = values
    header = reference.header.copy()
    header.set_data_dtype(np.float64)
    header.set_slope_inter(1, 0)
    image = nib.Nifti1Image(array.reshape(reference.shape, order="F"), reference.affine, header)
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(image, path)
