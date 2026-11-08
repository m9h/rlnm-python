"""Parcel-wise map associations with explicit annotation-side spatial surrogates.

This is an independent implementation. No EigenBrain source is incorporated.
The optional BrainSMASH backend fits variograms separately within declared blocks.
"""

import numpy as np

from .research import integer


def centered_unit(values, name="map"):
    values = np.asarray(values, dtype=float)
    if values.ndim not in (1, 2) or not np.isfinite(values).all() or values.shape[-1] < 3:
        raise ValueError(f"{name} must contain >=3 finite parcel values per map")
    centered = values - values.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(centered, axis=-1, keepdims=True)
    if np.any(norm <= 1e-12 * np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1)):
        raise ValueError(f"{name} contains a constant or numerically constant map")
    return centered / norm


def association(target, annotation, annotation_nulls):
    """Two-sided Pearson test. Null rows belong to annotation, never target.

    Zeros are valid data. The caller must supply an explicit common valid domain.
    Plus-one counting includes ties, with numerical tolerance at exact correlations.
    """
    x, y = centered_unit(target, "target"), centered_unit(annotation, "annotation")
    nulls = np.asarray(annotation_nulls, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or nulls.ndim != 2 or nulls.shape[1:] != x.shape:
        raise ValueError("Expected target/annotation vectors and nulls shaped (draws, parcels)")
    if len(nulls) < 1:
        raise ValueError("At least one null is required")
    r = float(np.clip(x @ y, -1, 1))
    null_r = np.clip(centered_unit(nulls, "annotation nulls") @ x, -1, 1)
    p = float((1 + np.count_nonzero(np.abs(null_r) >= abs(r) - 1e-12)) / (1 + len(nulls)))
    return r, p, null_r


def validate_distances(distances, blocks, n):
    distances, blocks = np.asarray(distances, dtype=float), np.asarray(blocks)
    if distances.shape != (n, n) or blocks.shape != (n,) or blocks.dtype.kind not in "iuUS":
        raise ValueError("Distances must be square with one integer/string block per parcel")
    if any(not str(x).strip() for x in blocks):
        raise ValueError("Empty spatial block label")
    for block in np.unique(blocks):
        indices = np.flatnonzero(blocks == block)
        d = distances[np.ix_(indices, indices)]
        if (
            len(indices) < 20
            or not np.isfinite(d).all()
            or not np.allclose(d, d.T, atol=1e-8, rtol=1e-8)
            or not np.allclose(np.diag(d), 0, atol=1e-8)
            or np.any(d[~np.eye(len(d), dtype=bool)] <= 0)
        ):
            raise ValueError(
                "Each spatial block needs >=20 parcels and finite symmetric distances "
                "with zero diagonal and positive off-diagonal entries"
            )
    return distances, blocks


def spatial_surrogates(
    annotation, distances, blocks, *, n_nulls=1000, seed=0, kernel="exp", pv=25, nh=25
):
    """Generate BrainSMASH surrogates and save variogram-fit diagnostics.

    Cross-block covariance is not preserved. Blocks should represent the intended
    null hypothesis (typically separate hemispheres). Diagnostics require review;
    successful generation alone does not certify spatial-null calibration.
    """
    integer(n_nulls, "n_nulls", 1)
    integer(seed, "seed")
    integer(nh, "nh", 3)
    if kernel not in ("exp", "gaussian", "invdist", "uniform") or not 0 < pv <= 100:
        raise ValueError("Invalid BrainSMASH kernel or pv")
    y = np.asarray(annotation, dtype=float)
    if y.ndim != 1:
        raise ValueError("Annotation must be a vector")
    centered_unit(y, "annotation")
    distances, blocks = validate_distances(distances, blocks, len(y))
    try:
        from brainsmash.mapgen.base import Base
    except ImportError as error:
        raise RuntimeError("Install rlnm-python[annotations] to generate spatial nulls") from error
    surrogates = np.empty((n_nulls, len(y)))
    diagnostics = []
    for number, block in enumerate(np.unique(blocks)):
        indices = np.flatnonzero(blocks == block)
        centered_unit(y[indices], f"annotation in block {block}")
        block_seed = int(np.random.SeedSequence([seed, number]).generate_state(1)[0])
        generator = Base(
            y[indices],
            distances[np.ix_(indices, indices)],
            seed=block_seed,
            kernel=kernel,
            pv=pv,
            nh=nh,
            n_jobs=1,
        )
        draws = np.asarray(generator(n=n_nulls)).reshape(n_nulls, len(indices))
        # BrainSMASH's non-resampled output is demeaned. Restore the observed
        # block mean so a bilateral mean difference is not erased by the null.
        draws = draws - draws.mean(axis=1, keepdims=True) + y[indices].mean()
        centered_unit(draws, "spatial nulls")
        surrogates[:, indices] = draws
        empirical = generator.compute_smooth_variogram(y[indices])
        # A deterministic subset makes QC tractable for large permutation counts.
        sample = draws[np.linspace(0, n_nulls - 1, min(n_nulls, 100), dtype=int)]
        variograms = np.stack([generator.compute_smooth_variogram(row) for row in sample])
        mean = variograms.mean(0)
        if not np.isfinite(variograms).all() or not np.isfinite(empirical).all():
            raise ValueError("Non-finite variogram fit; inspect distances and map smoothness")
        diagnostics.append(
            {
                "block": str(block),
                "n_parcels": len(indices),
                "seed": block_seed,
                "distance": generator.h.tolist(),
                "empirical": empirical.tolist(),
                "surrogate_mean": mean.tolist(),
                "surrogate_025": np.quantile(variograms, 0.025, axis=0).tolist(),
                "surrogate_975": np.quantile(variograms, 0.975, axis=0).tolist(),
                "relative_rmse": float(
                    np.sqrt(np.mean((mean - empirical) ** 2))
                    / max(np.mean(np.abs(empirical)), 1e-15)
                ),
                "qc_draws": len(sample),
            }
        )
    return surrogates, diagnostics
