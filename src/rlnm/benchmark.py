"""CLI workflow for an external-reference, paired MRI/PET benchmark."""

import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .eigenbrain import _components, connectivity_gradients, nested_benchmark
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


def run_benchmark(config_path, output):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    required = {"dataset", "reference", "space", "marker", "reference_citation"}
    keys(config, required | {"components", "sparsity", "seed"}, required)
    for key in required:
        text_setting(config[key], key)
    config.setdefault("components", [5, 10, 20])
    config.setdefault("sparsity", 0.9)
    config.setdefault("seed", 0)
    integer(config["seed"], "seed")
    with research_run(config_path, output, "eigenbrain-benchmark", config) as (
        output,
        paths,
        record,
    ):
        dataset_path = (config_path.parent / config["dataset"]).resolve()
        reference_path = (config_path.parent / config["reference"]).resolve()
        paths.extend([dataset_path, reference_path])
        dataset = load_npz(dataset_path, ("subject_ids", "parcel_ids", "fc", "pet"))
        reference = load_npz(reference_path, ("subject_ids", "parcel_ids", "fc"))
        ids = parcel_ids(dataset["parcel_ids"])
        if not np.array_equal(ids, parcel_ids(reference["parcel_ids"])):
            raise ValueError("Reference and paired dataset parcel IDs/order must match exactly")
        subject_ids = identifiers(dataset["subject_ids"], "subject_ids")
        reference_ids = identifiers(reference["subject_ids"], "reference subject_ids")
        if set(subject_ids) & set(reference_ids):
            raise ValueError(
                "Reference participants must be disjoint from paired test participants"
            )
        if (
            len(subject_ids) < 4
            or dataset["fc"].shape != (len(subject_ids), len(ids), len(ids))
            or dataset["pet"].shape != (len(subject_ids), len(ids))
            or reference["fc"].shape != (len(ids), len(ids))
        ):
            raise ValueError("Array shapes must match IDs and include >=4 paired participants")
        counts = _components(config["components"], len(ids) - 2, len(ids))
        with threadpool_limits(limits=1):
            ref_basis = connectivity_gradients(
                reference["fc"],
                n_components=max(counts),
                sparsity=config["sparsity"],
                seed=config["seed"],
            )
            bases = np.stack(
                [
                    connectivity_gradients(
                        fc,
                        n_components=max(counts),
                        sparsity=config["sparsity"],
                        seed=config["seed"],
                    )
                    for fc in dataset["fc"]
                ]
            )
            result = nested_benchmark(ref_basis, bases, dataset["pet"], components=counts)
        for subject, row in zip(subject_ids, result["rows"], strict=True):
            row["subject_id"] = str(subject)
        write_tsv(output / "participants.tsv", result["rows"])
        folds = []
        for i, subject in enumerate(subject_ids):
            train_ids = subject_ids[np.arange(len(subject_ids)) != i].tolist()
            folds.append(
                {
                    "test_subject_id": str(subject),
                    "training_subject_ids": train_ids,
                    "inner_validation": "leave one of the training subjects out",
                    "candidate_inner_mse": dict(
                        zip(map(str, counts), result["inner_mse"][i].tolist(), strict=True)
                    ),
                    "selected_components": result["rows"][i]["components"],
                }
            )
        dump_json(output / "folds.json", folds)
        np.savez_compressed(
            output / "predictions.npz",
            subject_ids=subject_ids,
            parcel_ids=ids,
            observed=result["observed"],
            predicted=result["predictions"],
            training_atlas=result["baselines"],
            inner_mse=result["inner_mse"],
            components=result["components"],
        )
        np.savez_compressed(
            output / "bases.npz",
            reference=ref_basis,
            subjects=bases,
            parcel_ids=ids,
            subject_ids=subject_ids,
        )
        rows = result["rows"]
        summary = {
            "analysis": "eigenbrain-benchmark",
            "experimental": True,
            "space": config["space"],
            "marker": config["marker"],
            "n_subjects": len(subject_ids),
            "n_parcels": len(ids),
            "components": counts,
            "reference_citation": config["reference_citation"],
            "validation": "Nested leave-one-subject-out; inner-test MSE selects component count",
            "baseline": "Mean standardized PET map of outer-training subjects only",
            "mean_model_mse": float(np.mean([r["model_mse"] for r in rows])),
            "mean_baseline_mse": float(np.mean([r["baseline_mse"] for r in rows])),
            "mean_mse_improvement": float(np.mean([r["mse_improvement"] for r in rows])),
            "subjects_improved": sum(r["mse_improvement"] > 0 for r in rows),
            "interpretation": "Experimental prediction of within-map standardized molecular "
            "spatial profiles, not absolute receptor density. No clinical "
            "accuracy or statistical significance is implied by these scores.",
            "limits": "FC-only independent implementation; not full EigenBrain reproduction. "
            "Related participants and site effects need grouped/external validation. "
            "Reference independence is declared by supplied IDs, not independently verified.",
        }
        record["embedding"] = {
            "library": "BrainSpace",
            "approach": "diffusion maps",
            "kernel": "normalized_angle",
            "alpha": 0.5,
            "diffusion_time": 0,
            "diagonal": "zeroed",
            "alignment": "orthogonal Procrustes per K",
            "basis_scaling": "within-subject column z-score after alignment",
            "pet_scaling": "within-map z-score; spatial shape only",
        }
        dump_json(output / "summary.json", summary)
        from .research_report import benchmark_report

        benchmark_report(output, summary, rows)
    return summary
