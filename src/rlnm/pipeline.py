"""Config-driven NIfTI pipelines with explicit inputs and complete provenance."""

import importlib.metadata
import json
import platform
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from . import UPSTREAM_COMMIT, __version__
from .io import image_data, load_image, load_maps, load_mask, read_manifest, same_grid, sha256
from .io import write_map as save_map
from .report import write_report
from .sensitivity import mean_connectivity, sensitivity_from_nulls, signed_overlap
from .spatial import NullConfig, SpatialNulls
from .specificity import specificity


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def resolve(base, value):
    return (base / value).resolve()


def _common(config):
    shared = {
        "analysis",
        "manifest",
        "brain_mask",
        "space",
        "seed",
        "n_permutations",
        "jobs",
        "fdr",
        "input_provenance",
    }
    extra = (
        {"batch_size", "sided", "sensitivity_masks"}
        if config.get("analysis") == "specificity"
        else {"connectome", "null_model", "tail", "tail_fraction"}
    )
    if config.get("analysis") not in ("specificity", "sensitivity"):
        raise ValueError("analysis must be specificity or sensitivity")
    unknown = set(config) - shared - extra
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    for key in ("manifest", "brain_mask", "space"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"A nonempty {key} is required")
    config.setdefault("seed", 0)
    config.setdefault("n_permutations", 10000)
    config.setdefault("jobs", 1)
    config.setdefault("fdr", 0.05 if config["analysis"] == "specificity" else 0.005)
    for key in ("seed", "n_permutations", "jobs"):
        if type(config[key]) is not int or config[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid integer setting: {key}")
    if not np.isfinite(config["fdr"]) or not 0 < config["fdr"] <= 1:
        raise ValueError("fdr must be in (0,1]")
    return config


def run(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = _common(json.loads(config_path.read_text()))
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    provenance = {
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "package": "rlnm-python",
        "version": __version__,
        "upstream_commit": UPSTREAM_COMMIT,
        "python": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "config": config,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "scipy", "nibabel", "joblib", "matplotlib")
        },
        "inputs": [],
        "rng": "PCG64 SeedSequence([seed, draw_id])",
        "space_is_user_declared": True,
    }
    dump_json(output / "provenance.json", provenance)
    input_paths = [config_path]
    try:
        base = config_path.parent
        manifest = resolve(base, config["manifest"])
        brain_mask_path = resolve(base, config["brain_mask"])
        input_paths += [manifest, brain_mask_path]
        if "input_provenance" in config:
            input_paths.append(resolve(base, config["input_provenance"]))
        rows = read_manifest(manifest, require_symptoms=config["analysis"] == "specificity")
        input_paths += [Path(row["map"]) for row in rows]
        reference, brain_mask = load_mask(brain_mask_path)
        provenance["geometry"] = {
            "shape": list(reference.shape),
            "affine": reference.affine.tolist(),
            "spatial_unit": reference.header.get_xyzt_units()[0],
            "index_order": "zero-based NIfTI array, Fortran flattening",
            "space": config["space"],
        }
        if config["analysis"] == "specificity":
            summary, figures = _specificity(
                config, base, output, rows, reference, brain_mask, input_paths
            )
        else:
            summary, figures = _sensitivity(
                config, base, output, rows, reference, brain_mask, input_paths
            )
        provenance["inputs"] = [
            {"path": str(path), "sha256": sha256(path)} for path in dict.fromkeys(input_paths)
        ]
        provenance["elapsed_seconds"] = round(time.monotonic() - started, 3)
        summary["elapsed_seconds"] = provenance["elapsed_seconds"]
        dump_json(output / "summary.json", summary)
        write_report(output, summary, figures)
        provenance["outputs"] = [
            {"path": path.name, "sha256": sha256(path)}
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "provenance.json"
        ]
        provenance["status"] = "complete"
    except Exception as error:
        provenance["status"] = "failed"
        provenance["error"] = str(error)
        raise
    finally:
        dump_json(output / "provenance.json", provenance)
    return summary


def _specificity(config, base, output, rows, reference, brain_mask, input_paths):
    config.setdefault("sided", "one")
    config.setdefault("batch_size", 128)
    maps = load_maps(rows, reference, brain_mask)
    labels = [row["symptom"] for row in rows]
    result = specificity(
        maps,
        labels,
        **{
            key: config[key]
            for key in ("n_permutations", "seed", "jobs", "batch_size", "sided", "fdr")
        },
    )
    indices = np.flatnonzero(brain_mask.ravel(order="F"))
    sensitivity_masks = config.get("sensitivity_masks")
    shared_mask = np.ones(len(indices), dtype=bool)
    if sensitivity_masks is not None:
        if not isinstance(sensitivity_masks, dict) or set(sensitivity_masks) != set(
            result.symptoms
        ):
            raise ValueError("sensitivity_masks must explicitly map every symptom to a mask")
        for symptom in result.symptoms:
            path = resolve(base, sensitivity_masks[symptom])
            image = load_image(path)
            same_grid(image, reference, str(path))
            # Empty significance masks are valid outcomes.
            shared_mask &= image_data(image, str(path), binary=True).ravel(order="F")[indices]
            input_paths.append(path)
    symptom_means = np.stack(
        [maps[np.asarray(labels) == symptom].mean(0) for symptom in result.symptoms]
    )
    shared = np.maximum(symptom_means, 0).mean(0)
    shared[~shared_mask] = 0
    shared_name = (
        "shared_sensitivity_masked" if sensitivity_masks is not None else "shared_unmasked"
    )
    save_map(output / f"{shared_name}.nii.gz", shared, reference, indices)
    np.savez_compressed(
        output / "intermediates.npz",
        maps=maps,
        labels=labels,
        regressors=result.regressors,
        residuals=result.residuals,
        t=result.t,
        effect=result.effect,
        p=result.p,
        q=result.q,
        indices=indices,
        symptom_means=symptom_means,
        shared=shared,
    )
    figures = [
        (
            "Shared positive connectivity ("
            + ("sensitivity masked" if sensitivity_masks is not None else "unmasked")
            + ")",
            f"{shared_name}.nii.gz",
        )
    ]
    results = []
    for i, symptom in enumerate(result.symptoms):
        stem = f"symptom-{i + 1:03d}"
        for kind, values in (
            ("mean", symptom_means),
            ("t", result.t),
            ("effect", result.effect),
            ("p", result.p),
            ("q", result.q),
            ("significant", result.significant),
            ("specific", result.specific_maps),
        ):
            save_map(
                output / f"{stem}_{kind}.nii.gz",
                values[i],
                reference,
                indices,
                outside=1 if kind in ("p", "q") else 0,
            )
        figures += [
            (f"{symptom}: standardized effect", f"{stem}_effect.nii.gz"),
            (f"{symptom}: effect where q < {config['fdr']}", f"{stem}_specific.nii.gz"),
        ]
        results.append(
            {
                "symptom": symptom,
                "file_stem": stem,
                "n": int(result.counts[i]),
                "significant_voxels": int(result.significant[i].sum()),
                "minimum_q": float(result.q[i].min()),
            }
        )
    return {
        "analysis": "specificity",
        "space": config["space"],
        "n_lesions": len(rows),
        "domain_voxels": len(indices),
        "n_permutations": config["n_permutations"],
        "fdr": config["fdr"],
        "sided": config["sided"],
        "inference": "upstream fixed-residual label permutation; not a refit-permutation test",
        "fdr_family": "domain voxels, separately for each symptom",
        "shared_map_sensitivity_masked": sensitivity_masks is not None,
        "results": results,
    }, figures


def _sensitivity(config, base, output, rows, reference, brain_mask, input_paths):
    config.setdefault("tail", "upstream-sign")
    config.setdefault("tail_fraction", 0.25)
    if "connectome" not in config:
        raise ValueError("sensitivity requires a connectome metadata JSON path")
    meta_path = resolve(base, config["connectome"])
    meta = json.loads(meta_path.read_text())
    if set(meta) != {"matrix", "indices", "reference", "space", "index_order"}:
        raise ValueError("Connectome JSON needs matrix, indices, reference, space, index_order")
    if meta["space"] != config["space"] or meta["index_order"] != "nifti-fortran-zero":
        raise ValueError("Connectome space/index-order differs from the analysis declaration")
    fc_path, idx_path, ref_path = [
        resolve(meta_path.parent, meta[key]) for key in ("matrix", "indices", "reference")
    ]
    input_paths += [meta_path, fc_path, idx_path, ref_path]
    same_grid(load_image(ref_path), reference, "connectome reference")
    fc = np.load(fc_path, mmap_mode="r", allow_pickle=False)
    indices = np.load(idx_path, allow_pickle=False)
    if (
        indices.ndim != 1
        or not np.issubdtype(indices.dtype, np.integer)
        or indices.size == 0
        or np.any((indices < 0) | (indices >= brain_mask.size))
        or len(np.unique(indices)) != len(indices)
        or fc.shape != (len(indices), len(indices))
    ):
        raise ValueError("Invalid connectome index mapping or square matrix shape")
    lookup = np.full(brain_mask.size, -1, dtype=int)
    lookup[indices] = np.arange(len(indices))
    if np.any(lookup[brain_mask.ravel(order="F")] < 0):
        raise ValueError("Placement-mask voxels must all have corresponding connectome rows")
    lesions = []
    for row in rows:
        image = load_image(row["map"])
        same_grid(image, reference, row["map"])
        lesion = np.flatnonzero(image_data(image, row["map"], binary=True).ravel(order="F"))
        lesions.append(lesion)
    null_config = NullConfig(**config.get("null_model", {}))
    nulls = SpatialNulls(lesions, brain_mask, seed=config["seed"], config=null_config)
    observed_maps = mean_connectivity(fc, [lookup[lesion] for lesion in lesions])
    observed = signed_overlap(observed_maps, config["tail_fraction"])
    diagnostics = []

    def draw(draw_id):
        placed, diagnostic = nulls.draw(draw_id)
        maps = mean_connectivity(fc, [lookup[lesion] for lesion in placed])
        return signed_overlap(maps, config["tail_fraction"]), diagnostic

    def stream():
        with threadpool_limits(limits=1):
            results = Parallel(n_jobs=config["jobs"], prefer="threads", return_as="generator")(
                delayed(draw)(i) for i in range(config["n_permutations"])
            )
            for null_map, diagnostic in results:
                diagnostics.append(diagnostic)
                yield null_map

    result = sensitivity_from_nulls(observed, stream(), fdr=config["fdr"], tail=config["tail"])
    for kind, values in (
        ("overlap", observed),
        ("p", result.p),
        ("q", result.q),
        ("significant", result.significant),
        ("specific", np.where(result.significant, observed, 0)),
    ):
        save_map(
            output / f"sensitivity_{kind}.nii.gz",
            values,
            reference,
            indices,
            outside=1 if kind in ("p", "q") else 0,
        )
    np.savez_compressed(
        output / "intermediates.npz",
        observed_maps=observed_maps,
        observed=observed,
        p=result.p,
        q=result.q,
        indices=indices,
    )
    dump_json(output / "null_diagnostics.json", diagnostics)
    summary = {
        "analysis": "sensitivity",
        "space": config["space"],
        "n_lesions": len(rows),
        "domain_voxels": len(indices),
        "n_permutations": result.n_nulls,
        "tail": result.tail,
        "tail_fraction": config["tail_fraction"],
        "fdr": config["fdr"],
        "null_model": asdict(null_config),
        "minimum_q": float(result.q.min()),
        "significant_voxels": int(result.significant.sum()),
        "fdr_family": "all connectome target voxels",
        "ks_engine": "SciPy asymptotic KS; not claimed identical to MATLAB kstest2",
    }
    return summary, [
        ("Observed signed overlap", "sensitivity_overlap.nii.gz"),
        ("Signed overlap surviving FDR", "sensitivity_specific.nii.gz"),
    ]
