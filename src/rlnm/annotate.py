"""NIfTI-to-parcel annotation workflow; all coordinate changes are explicit."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from .annotation import association, spatial_surrogates, validate_distances
from .io import image_data, load_image, load_mask, same_grid
from .research import (
    dump_json,
    identifiers,
    integer,
    keys,
    load_npz,
    parcel_ids,
    research_run,
    text_setting,
    write_tsv,
)
from .stats import bh_fdr


def _config(config):
    required = {"map", "mask", "atlas", "distances", "space", "annotations"}
    keys(config, required | {"n_nulls", "seed", "min_coverage", "null_model"}, required)
    for key in required - {"annotations"}:
        text_setting(config[key], key)
    config.setdefault("n_nulls", 1000)
    config.setdefault("seed", 0)
    config.setdefault("min_coverage", 0.8)
    config.setdefault("null_model", {"kernel": "exp", "pv": 25, "nh": 25})
    integer(config["n_nulls"], "n_nulls", 1)
    integer(config["seed"], "seed")
    if not 0 < config["min_coverage"] <= 1:
        raise ValueError("min_coverage must be in (0,1]")
    keys(config["null_model"], {"kernel", "pv", "nh"}, set())
    if not isinstance(config["annotations"], list) or not config["annotations"]:
        raise ValueError("annotations must be a nonempty list")
    for annotation in config["annotations"]:
        keys(
            annotation,
            {"id", "map", "neuromaps", "citation", "resample", "zero_is_missing"},
            {"id", "citation"},
        )
        text_setting(annotation["id"], "annotation id")
        text_setting(annotation["citation"], "annotation citation")
        if ("map" in annotation) == ("neuromaps" in annotation):
            raise ValueError("Each annotation needs exactly one of map or neuromaps")
        if "map" in annotation:
            text_setting(annotation["map"], "annotation map")
        else:
            query = annotation["neuromaps"]
            keys(query, {"source", "desc", "space", "res"}, {"source", "desc", "space", "res"})
            for key, value in query.items():
                text_setting(value, key)
            if query["space"] != "MNI152" or config["space"] != "MNI152":
                raise ValueError("Direct neuromaps fetching currently supports MNI152 volumes only")
        for flag in ("resample", "zero_is_missing"):
            annotation.setdefault(flag, False)
            if type(annotation[flag]) is not bool:
                raise ValueError(f"{flag} must be boolean")
    identifiers([a["id"] for a in config["annotations"]], "annotation IDs")
    return config


def _fetch(query, cache):
    try:
        from neuromaps.datasets import fetch_annotation
    except ImportError as error:
        raise RuntimeError("Install rlnm-python[annotations] to fetch neuromaps data") from error
    result = fetch_annotation(**query, data_dir=cache, return_single=False, verbose=0)
    if not isinstance(result, dict) or len(result) != 1:
        raise ValueError("neuromaps query must select exactly one volume")
    value = next(iter(result.values()))
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("neuromaps query returned multiple images")
        value = value[0]
    return Path(value).resolve()


def _annotation_image(spec, base, output, reference, paths, transformations):
    path = (
        _fetch(spec["neuromaps"], output / "cache")
        if "neuromaps" in spec
        else (base / spec["map"]).resolve()
    )
    paths.append(path)
    image = load_image(path)
    if image.header.get_xyzt_units()[0] != "mm":
        raise ValueError(f"Annotation requires declared millimetre units: {path}")
    data = image.get_fdata()
    data[~np.isfinite(data)] = np.nan
    if spec["zero_is_missing"]:
        data[data == 0] = np.nan
    if spec["resample"]:
        # Missing voxels are NaN before interpolation; no zero-filled field of view.
        source = nib.Nifti1Image(data, image.affine, image.header)
        transformed = resample_from_to(source, reference, order=1, mode="constant", cval=np.nan)
        data = transformed.get_fdata()
        transformations.append(
            {
                "annotation": spec["id"],
                "source": str(path),
                "source_affine": image.affine.tolist(),
                "source_shape": list(image.shape),
                "operation": "linear resampling within declared template",
                "outside_field_of_view": "NaN; excluded from common domain",
            }
        )
    else:
        same_grid(image, reference, str(path))
    return data


def parcellate_common(target, atlas, mask, annotations, ids, min_coverage):
    """One common voxel domain for every comparison; coverage relative to full parcels."""
    arrays = [np.asarray(target, dtype=float), *[np.asarray(a, dtype=float) for a in annotations]]
    if any(a.shape != atlas.shape for a in arrays) or mask.shape != atlas.shape:
        raise ValueError("All arrays must share the atlas shape")
    valid = np.asarray(mask, dtype=bool).copy()
    for array in arrays:
        valid &= np.isfinite(array)
    vectors, counts, coverage = [], [], []
    for label in ids:
        parcel = atlas == label
        n_total = np.count_nonzero(parcel)
        if not n_total:
            raise ValueError(f"Parcel {label} is absent from atlas")
        sample = parcel & valid
        count = np.count_nonzero(sample)
        counts.append(count)
        coverage.append(count / n_total)
        vectors.append([a[sample].mean() if count else np.nan for a in arrays])
    coverage = np.asarray(coverage)
    keep = (coverage >= min_coverage) & (np.asarray(counts) > 0)
    if not keep.any():
        raise ValueError("No parcels meet common-domain coverage requirements")
    return np.asarray(vectors).T, np.asarray(counts), coverage, keep


def run_annotation(config_path, output):
    config_path = Path(config_path).resolve()
    config = _config(json.loads(config_path.read_text()))
    with research_run(config_path, output, "annotation", config) as (output, paths, record):
        base = config_path.parent
        files = {
            key: (base / config[key]).resolve() for key in ("map", "mask", "atlas", "distances")
        }
        paths.extend(files.values())
        reference = load_image(files["map"])
        if reference.header.get_xyzt_units()[0] != "mm":
            raise ValueError("Target map requires declared millimetre units")
        _, mask = load_mask(files["mask"], reference)
        atlas_image = load_image(files["atlas"])
        same_grid(atlas_image, reference, "atlas")
        atlas = image_data(atlas_image, "atlas")
        if np.any(atlas < 0) or np.any(atlas != np.floor(atlas)):
            raise ValueError("Atlas must contain nonnegative integer labels; zero is background")
        distance_data = load_npz(files["distances"], ("parcel_ids", "distances", "blocks"))
        ids = parcel_ids(distance_data["parcel_ids"])
        if set(ids) != set(np.unique(atlas[atlas > 0])):
            raise ValueError("Distance parcel_ids must exactly match nonzero atlas labels")
        distances, blocks = validate_distances(
            distance_data["distances"], distance_data["blocks"], len(ids)
        )
        transformations = []
        annotations = [
            _annotation_image(spec, base, output, reference, paths, transformations)
            for spec in config["annotations"]
        ]
        vectors, counts, coverage, keep = parcellate_common(
            reference.get_fdata(), atlas, mask, annotations, ids, config["min_coverage"]
        )
        distance_used = distances[np.ix_(keep, keep)]
        block_used = blocks[keep]
        validate_distances(distance_used, block_used, int(keep.sum()))
        record["transformations"] = transformations
        record["geometry"] = {
            "shape": list(reference.shape),
            "affine": reference.affine.tolist(),
            "space": config["space"],
            "spatial_unit": "mm",
        }
        record["rng"] = "BrainSMASH seeded per annotation and spatial block; n_jobs=1"
        names = [a["id"] for a in config["annotations"]]
        rows, null_correlations, diagnostics = [], [], []
        for i, spec in enumerate(config["annotations"]):
            seed = int(np.random.SeedSequence([config["seed"], i]).generate_state(1)[0])
            draws, qc = spatial_surrogates(
                vectors[i + 1, keep],
                distance_used,
                block_used,
                n_nulls=config["n_nulls"],
                seed=seed,
                **config["null_model"],
            )
            r, p, null_r = association(vectors[0, keep], vectors[i + 1, keep], draws)
            rows.append(
                {
                    "annotation": spec["id"],
                    "r": r,
                    "p_spatial": p,
                    "n_parcels": int(keep.sum()),
                    "citation": spec["citation"],
                    "worst_variogram_relative_rmse": max(q["relative_rmse"] for q in qc),
                }
            )
            null_correlations.append(null_r)
            diagnostics.append({"annotation": spec["id"], "blocks": qc})
        for row, q in zip(rows, bh_fdr([row["p_spatial"] for row in rows]), strict=True):
            row["q_bh"] = float(q)
        parcel_rows = []
        for i, label in enumerate(ids):
            row = {
                "parcel_id": int(label),
                "block": str(blocks[i]),
                "n_voxels": int(counts[i]),
                "coverage": float(coverage[i]),
                "included": bool(keep[i]),
            }
            for j, name in enumerate(["target", *[f"annotation:{x}" for x in names]]):
                row[name] = float(vectors[j, i]) if np.isfinite(vectors[j, i]) else ""
            parcel_rows.append(row)
        write_tsv(output / "parcels.tsv", parcel_rows)
        write_tsv(output / "associations.tsv", rows)
        dump_json(output / "null_diagnostics.json", diagnostics)
        np.savez_compressed(
            output / "parcel_data.npz",
            parcel_ids=ids[keep],
            target=vectors[0, keep],
            annotations=vectors[1:, keep],
            annotation_ids=names,
            blocks=block_used,
        )
        np.savez_compressed(
            output / "null_correlations.npz",
            annotation_ids=names,
            correlations=np.asarray(null_correlations),
        )
        summary = {
            "analysis": "annotation",
            "space": config["space"],
            "n_parcels": int(keep.sum()),
            "excluded_parcels": ids[~keep].tolist(),
            "n_nulls": config["n_nulls"],
            "minimum_possible_p": 1 / (1 + config["n_nulls"]),
            "null_fit_review_required": True,
            "fdr_family": "all requested annotations for this one target map",
            "test": "two-sided Pearson; annotation-side BrainSMASH spatial surrogates",
            "coverage": "common finite voxel domain; zeros retained unless explicitly excluded",
            "interpretation": "Spatial correspondence to reference maps, not receptor measurements",
            "null_caveat": "Independent blocks do not preserve cross-block covariance. "
            "Review variograms; generation does not certify null calibration.",
            "results": rows,
        }
        dump_json(output / "summary.json", summary)
        from .research_report import annotation_report

        annotation_report(output, summary, diagnostics)
    return summary
