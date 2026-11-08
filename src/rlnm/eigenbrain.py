"""Independent, experimental EigenBrain-style validation, not an author-code port.

An external reference basis anchors all subjects. Molecular weights and component
selection are trained with nested leave-one-subject-out evaluation. Outputs model
spatial profiles, not absolute tracer concentrations or receptor density.
"""

import warnings

import numpy as np
from scipy.linalg import orthogonal_procrustes

from .annotation import centered_unit
from .research import integer


def _components(components, available, parcels):
    values = list(components)
    if not values:
        raise ValueError("At least one component count is required")
    for value in values:
        integer(value, "component count", 1)
    if len(set(values)) != len(values) or max(values) > min(available, parcels - 2):
        raise ValueError("Component counts must be unique and <= available modes and parcels - 2")
    return sorted(values)


def _design(basis):
    mean, sd = basis.mean(0), basis.std(0)
    if np.any(sd <= 1e-12):
        raise ValueError("Basis has a constant or numerically constant component")
    return np.column_stack((np.ones(len(basis)), (basis - mean) / sd))


def _correlation(x, y):
    if np.linalg.norm(x - np.mean(x)) < 1e-12 or np.linalg.norm(y - np.mean(y)) < 1e-12:
        return None
    return float(np.clip(centered_unit(x) @ centered_unit(y), -1, 1))


def nested_benchmark(reference_basis, subject_bases, pet, *, components=(5, 10, 20)):
    """Nested LOSO selection by inner-test MSE; outer-test PET is never used to fit.

    Inputs: external reference (parcels, modes), individual unaligned bases
    (subjects, parcels, modes), paired PET (subjects, parcels). PET is standardized
    within each map: only relative cortical spatial patterns are evaluated.
    Procrustes alignment is repeated for each candidate dimensionality.
    """
    reference = np.asarray(reference_basis, dtype=float)
    subjects, pet = np.asarray(subject_bases, dtype=float), np.asarray(pet, dtype=float)
    if (
        reference.ndim != 2
        or subjects.ndim != 3
        or pet.ndim != 2
        or subjects.shape[1:] != reference.shape
        or subjects.shape[:2] != pet.shape
        or len(pet) < 4
        or not all(np.isfinite(x).all() for x in (reference, subjects, pet))
    ):
        raise ValueError(
            "Need finite matching reference/subject bases and paired PET for >=4 subjects"
        )
    counts = _components(components, reference.shape[1], reference.shape[0])
    y = centered_unit(pet, "PET") * np.sqrt(pet.shape[1])
    n = len(y)
    designs, coefficients = {}, {}
    for k in counts:
        ref = reference[:, :k]
        reference_design = _design(ref)
        if np.linalg.matrix_rank(reference_design) < k + 1:
            raise ValueError(f"Reference design is rank deficient at {k} components")
        designs[k] = np.stack(
            [
                _design(basis[:, :k] @ orthogonal_procrustes(basis[:, :k], ref)[0])
                for basis in subjects
            ]
        )
        # OLS is linear in y: averaging these coefficients is exactly fitting
        # to the training-only mean map, without repeatedly solving the same design.
        coefficients[k] = np.linalg.lstsq(reference_design, y.T, rcond=None)[0].T
    predictions, baselines, scores, rows = [], [], [], []
    for test in range(n):
        train = np.arange(n) != test
        baseline = y[train].mean(0)
        candidate_scores = []
        for k in counts:
            train_coefficients = coefficients[k][train]
            inner_means = (train_coefficients.sum(0) - train_coefficients) / (n - 2)
            inner_predictions = np.einsum("spk,sk->sp", designs[k][train], inner_means)
            candidate_scores.append(float(np.mean((inner_predictions - y[train]) ** 2)))
        selected = counts[int(np.argmin(candidate_scores))]
        prediction = designs[selected][test] @ coefficients[selected][train].mean(0)
        mse, baseline_mse = (
            float(np.mean((y[test] - prediction) ** 2)),
            float(np.mean((y[test] - baseline) ** 2)),
        )
        rows.append(
            {
                "test_index": test,
                "n_train": n - 1,
                "components": selected,
                "model_r": _correlation(prediction, y[test]),
                "baseline_r": _correlation(baseline, y[test]),
                "model_mse": mse,
                "baseline_mse": baseline_mse,
                "mse_improvement": baseline_mse - mse,
                "deviation_r": _correlation(prediction - baseline, y[test] - baseline),
            }
        )
        predictions.append(prediction)
        baselines.append(baseline)
        scores.append(candidate_scores)
    return {
        "rows": rows,
        "predictions": np.asarray(predictions),
        "baselines": np.asarray(baselines),
        "observed": y,
        "inner_mse": np.asarray(scores),
        "components": np.asarray(counts),
    }


def connectivity_gradients(connectivity, *, n_components=20, sparsity=0.9, seed=0):
    """BrainSpace diffusion-map embedding of a finite symmetric FC matrix.

    This uses the functional-connectivity branch only. It is not a geometric
    Laplacian eigenbasis and does not implement structural/geodesic modality fusion.
    """
    matrix = np.asarray(connectivity, dtype=float)
    integer(n_components, "n_components", 1)
    integer(seed, "seed")
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix, matrix.T, atol=1e-8, rtol=1e-8)
        or n_components >= len(matrix) - 1
        or not 0 <= sparsity < 1
    ):
        raise ValueError(
            "Need a finite symmetric FC matrix, components < parcels - 1, 0<=sparsity<1"
        )
    if int(len(matrix) * (1 - sparsity)) < 2:
        raise ValueError("Sparsity must retain at least two entries per FC row")
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0)
    try:
        from brainspace.gradient import GradientMaps
    except ImportError as error:
        raise RuntimeError("Install rlnm-python[eigenbrain] to compute FC gradients") from error
    random_state = int(np.random.SeedSequence(seed).generate_state(1)[0])
    model = GradientMaps(
        n_components=n_components,
        approach="dm",
        kernel="normalized_angle",
        random_state=random_state,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(matrix, sparsity=sparsity, alpha=0.5, diffusion_time=0)
    if any("not fully connected" in str(item.message) for item in caught):
        raise ValueError("Embedding graph is disconnected; inspect connectivity and sparsity")
    basis = np.asarray(model.gradients_)
    if basis.shape != (len(matrix), n_components) or not np.isfinite(basis).all():
        raise ValueError("Embedding returned invalid modes")
    return basis
